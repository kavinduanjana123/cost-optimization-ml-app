import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_excel("unified_company_dataset_with_subscriptions.xlsx")

# Aggregate by Year-Month
df['YearMonth'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
df_month = df.groupby('YearMonth')['ActualCost_LKR'].sum().reset_index()
df_month.columns = ['ds', 'y']

# Train Prophet model
m = Prophet()
m.fit(df_month)

# Forecast next 12 months
future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)

# Plot forecast
fig1 = m.plot(forecast)
plt.title("12-Month Forecast of Total Company Cost")
plt.show()

# Optional: Save forecast for dashboard
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('cost_forecast.csv', index=False)
