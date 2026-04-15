import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


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
date_test = pd.to_datetime(df['date'].iloc[split_index:])


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = np.corrcoef(y_test, y_pred)[0, 1] ** 2

print("\nLR ")
print(f"L w1 = {model.coef_[0]:.4f}, w2 = {model.coef_[1]:.4f}")
print(f"L b = {model.intercept_:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R2: {r2:.4f}")


results = pd.DataFrame({
    'date': date_test,
    'True AQI': y_test,
    'Predicted AQI': y_pred
})
results.to_csv("lr_selfreg_aqi.csv", index=False, encoding='utf-8-sig')


plt.figure(figsize=(14, 6))
plt.plot(date_test, y_test, label='True AQI', marker='o')
plt.plot(date_test, y_pred, label='Predicted AQI (LR)', linestyle='--', marker='x')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.title("Linear Regression Predicted vs True AQI")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lr_selfreg_aqi_plot.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='black', s=60)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.title(f'LR $R^2$ = {r2:.4f}')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.grid(True)
plt.tight_layout()
plt.savefig("lr_selfreg_r2_scatter.png", dpi=1200)
plt.show()
