import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("beijing_aqi_2022_2024_combined.csv")
df = df[['date', 'AQI']]
df['AQI_t-1'] = df['AQI'].shift(1)
df['AQI_t-2'] = df['AQI'].shift(2)
df.dropna(inplace=True)

split_index = int(len(df) * 0.8)
X_train = df[['AQI_t-1', 'AQI_t-2']].iloc[:split_index].values
X_test = df[['AQI_t-1', 'AQI_t-2']].iloc[split_index:].values
y_train = df['AQI'].iloc[:split_index].values
y_test = df['AQI'].iloc[split_index:].values
date_test = pd.to_datetime(df['date'].iloc[split_index:].values)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = np.corrcoef(y_test, y_pred)[0, 1] ** 2

print("\nRF：")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R2: {r2:.4f}")


results = pd.DataFrame({
    'date': date_test,
    'True AQI': y_test,
    'Predicted AQI': y_pred
})
results.to_csv("rf_selfreg_aqi.csv", index=False, encoding='utf-8-sig')

plt.figure(figsize=(14, 6))
plt.plot(date_test, y_test, label='True AQI', marker='o')
plt.plot(date_test, y_pred, label='Predicted AQI (RF)', linestyle='--', marker='x')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.title("RF Predicted vs True AQI")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_selfreg_aqi_plot.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='black', s=60)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.title(f'RF $R^2$ = {r2:.4f}')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_selfreg_r2_scatter.png", dpi=300)
plt.show()
