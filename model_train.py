# -----------------------------
# COST OPTIMIZATION MODEL TRAINING
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt


# 1️⃣ Load dataset
df = pd.read_excel("unified_company_dataset_with_subscriptions.xlsx")

# 2️⃣ Define target variable
target = "OptimizedCost_LKR"

# 3️⃣ Select useful features (we exclude identifiers and the target)
features = [
    "Year", "Month", "Department", "Role", "CostType", "CostCategory", "Headcount",
    "BasicSalary_LKR", "Employer_EPF_LKR", "Employer_ETF_LKR", "MonthlyTotalEmploymentCost_LKR",
    "ResourceUtilization_%", "InflationRate_%", "Depreciation_LKR", "ProjectRevenue_LKR",
    "SoftwareSubscriptionCost_LKR"
]

X = df[features]
y = df[target]

# 4️⃣ Encode categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in cat_cols:
    X[col] = encoder.fit_transform(X[col])

# 5️⃣ Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Initialize and train model
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 7️⃣ Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Performance:")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.3f}")

# 8️⃣ Save model
joblib.dump(model, "cost_optimizer.pkl")
print("💾 Model saved as cost_optimizer.pkl")

# 9️⃣ Visualize Actual vs Predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Optimized Cost")
plt.ylabel("Predicted Optimized Cost")
plt.title("Actual vs Predicted Cost")
plt.grid(True)
plt.show()




import matplotlib.pyplot as plt
from xgboost import plot_importance

# Feature Importance Visualization
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='gain')
plt.title("Feature Importance - Cost Optimization Model")
plt.show()



import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)