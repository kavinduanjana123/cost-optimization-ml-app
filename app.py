from flask import Flask, render_template, request
import pandas as pd
import mlflow.sklearn
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn.preprocessing import LabelEncoder
from mlflow.tracking import MlflowClient
import joblib


app = Flask(__name__)
MODEL_NAME = "cost_optimizer_v2"

# Load the latest production model
try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    print("✅ Loaded Production model from MLflow")
except Exception:
    print("⚠️ Could not load Production model. Using local model.")
    model = joblib.load("cost_optimizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "No file uploaded!", 400

    # Read the uploaded Excel file
    df = pd.read_excel(file)
    X = df.copy()

    # ✅ Keep only features used during training
    trained_features = [
        "Year", "Month", "Department", "Role", "CostType", "CostCategory", "Headcount",
        "BasicSalary_LKR", "Employer_EPF_LKR", "Employer_ETF_LKR", "MonthlyTotalEmploymentCost_LKR",
        "ResourceUtilization_%", "InflationRate_%", "Depreciation_LKR", "ProjectRevenue_LKR",
        "SoftwareSubscriptionCost_LKR"
    ]

    # Check for missing columns
    missing_cols = [col for col in trained_features if col not in X.columns]
    if missing_cols:
        return f"❌ Missing columns in your file: {missing_cols}", 400

    X = X[trained_features]

    # ✅ Encode categorical columns
  
    cat_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # ✅ Ensure numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # ✅ Predict
    y_pred = model.predict(X)
    print("✅ Prediction sample:", y_pred[:5])
    print("📊 Total predicted cost:", y_pred.sum())

    df['Predicted_OptimizedCost_LKR'] = y_pred
    total_pred = float(df['Predicted_OptimizedCost_LKR'].sum())

    # ✅ Chart


    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['Predicted_OptimizedCost_LKR'], name='Predicted Cost'))
    fig.update_layout(
        title='Predicted Optimized Costs per Record',
        xaxis_title='Record Index',
        yaxis_title='Predicted Cost (LKR)',
        template='plotly_white'
    )
    chart = pyo.plot(fig, output_type='div')

    return render_template(
        'result.html',
        tables=[df.head(20).to_html(classes='data', index=False)],
        chart=chart,
        total_pred=total_pred
    )



@app.route('/insights')
def insights():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    model_data = []
    for v in versions:
        run_metrics = client.get_run(v.run_id).data.metrics
        model_data.append({
            "Version": v.version,
            "Stage": v.current_stage,
            "R2": run_metrics.get("R2", None),
            "MAE": run_metrics.get("MAE", None),
            "RMSE": run_metrics.get("RMSE", None),
            "Run_ID": v.run_id
        })

    df = pd.DataFrame(model_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Version"], y=df["R2"], mode="lines+markers", name="R² Score"))
    fig.update_layout(title="Model Performance by Version", xaxis_title="Version", yaxis_title="R² Score")
    chart = pyo.plot(fig, output_type="div")

    return render_template('model_insights.html', chart=chart, tables=[df.to_html(classes='data', index=False)])





if __name__ == "__main__":
    app.run(debug=True)
