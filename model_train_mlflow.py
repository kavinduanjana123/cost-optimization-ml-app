"""
model_train_mlflow.py
Train, evaluate, and version a cost optimization model using MLflow.
Automatically promotes the best-performing model to Production.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# =====================================================
# 1️⃣ Load Dataset
# =====================================================
print("📥 Loading dataset...")
df = pd.read_excel("unified_company_dataset_with_subscriptions.xlsx")

target = "OptimizedCost_LKR"
features = [
    "Year", "Month", "Department", "Role", "CostType", "CostCategory", "Headcount",
    "BasicSalary_LKR", "Employer_EPF_LKR", "Employer_ETF_LKR", "MonthlyTotalEmploymentCost_LKR",
    "ResourceUtilization_%", "InflationRate_%", "Depreciation_LKR", "ProjectRevenue_LKR",
    "SoftwareSubscriptionCost_LKR"
]

X = df[features].copy()
y = df[target].copy()

# Encode categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in cat_cols:
    X[col] = encoder.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data ready: {X_train.shape[0]} train, {X_test.shape[0]} test samples.")

# =====================================================
# 2️⃣ Model & Params
# =====================================================
params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}
model = XGBRegressor(**params)

# =====================================================
# 3️⃣ MLflow Setup
# =====================================================
mlflow.set_experiment("Cost_Optimization_Model")
client = MlflowClient()
MODEL_NAME = "cost_optimizer_v2"

# Create model name if not exists
try:
    client.get_registered_model(MODEL_NAME)
except MlflowException:
    client.create_registered_model(MODEL_NAME)
    print(f"✅ Created MLflow Registered Model: {MODEL_NAME}")

# =====================================================
# 4️⃣ Train + Evaluate + Log
# =====================================================
with mlflow.start_run() as run:
    print(f"🚀 MLflow Run ID: {run.info.run_id}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"📊 Metrics → R²={r2:.3f}, MAE={mae:,.2f}, RMSE={rmse:,.2f}")

    mlflow.log_params(params)
    mlflow.log_metrics({"R2": r2, "MAE": mae, "RMSE": rmse})

    mlflow.sklearn.log_model(model, "model", input_example=X_test.iloc[:1])
    model_uri = f"runs:/{run.info.run_id}/model"

    joblib.dump(model, "cost_optimizer.pkl")

    # Register new model version
    result = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run.info.run_id
    )
    print(f"✅ Registered '{MODEL_NAME}' version {result.version}")

    # Compare with current Production version
    try:
        current_prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        prod_r2 = client.get_run(current_prod.run_id).data.metrics.get("R2", 0)
        print(f"ℹ️ Current Production R² = {prod_r2:.3f}")

        if r2 > prod_r2:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=result.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"🚀 Promoted version {result.version} to Production (R² improved)")
        else:
            print("⚠️ New model not better. Keeping old Production model.")
    except IndexError:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=result.version,
            stage="Production"
        )
        print(f"🏆 Promoted version {result.version} to Production (first model)")

print("✅ Training & MLflow logging complete!")
