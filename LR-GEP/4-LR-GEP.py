import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import matplotlib.dates as mdates


raw_df = pd.read_csv("beijing_aqi_2022_2024_combined.csv")
df = pd.read_csv("F-AQI-Zscore-LR.csv")

features = ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']
target = 'AQI'


df.dropna(subset=features + [target], inplace=True)
df.reset_index(drop=True, inplace=True)

raw_df.dropna(subset=features + [target], inplace=True)
raw_df.reset_index(drop=True, inplace=True)


split_index = int(len(df) * 0.8)
X_train = df[features].iloc[:split_index]
X_test = df[features].iloc[split_index:]
y_train = df[target].iloc[:split_index]
y_test = df[target].iloc[split_index:]


test_indices = df.iloc[split_index:].index
y_test_real = raw_df.loc[test_indices, target].values
date_test = raw_df.loc[test_indices, 'date'].values


def protected_div(x1, x2):
    return np.where(np.abs(x2) > 1e-6, x1 / x2, 0.0)

division = make_function(function=protected_div, name='div', arity=2)


best_model = None
best_mse = float('inf')
patience = 5
patience_counter = 0
max_generations = 300
results = []


aqi_scaler = MinMaxScaler()
aqi_scaler.fit(raw_df[[target]])

# Step 7: 迭代训练并监控性能
for gen in range(1, max_generations + 1):
    model = SymbolicRegressor(
        population_size=500,
        generations=gen,
        function_set=('add', 'sub', 'mul', division),
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=0,
        random_state=gen,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    y_pred_norm = model.predict(X_test)
    y_pred_real = aqi_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()


    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_real, y_pred_real)
    r2 = np.corrcoef(y_test_real, y_pred_real)[0, 1] ** 2

    results.append((gen, mse, rmse, mape, r2))

    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_y_pred_real = y_pred_real
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"ealy {gen} 代。")
        break


results_df = pd.DataFrame({
    'date': date_test,
    'True AQI': y_test_real,
    'Predicted AQI (LR-GEP)': best_y_pred_real
})
results_df.to_csv("LR-GEP_predicted_vs_true_aqi_dynamic.csv", index=False, encoding='utf-8-sig')


date_test = pd.to_datetime(date_test)
plt.figure(figsize=(14, 7))
plt.plot(date_test, y_test_real, label="True AQI", marker='o')
plt.plot(date_test, best_y_pred_real, label="Predicted AQI (LR-GEP)", linestyle='--', marker='x')

plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.title("LR-GEP Predicted vs True AQI")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lr-gep_vs_true_aqi_dynamic.png", dpi=1200)
plt.show()


gen, mse, rmse, mape, r2 = results[-1]
print(f"\n LR-GEP  {gen} ）：")
print(f"(MSE): {mse:.4f}")
print(f"(RMSE): {rmse:.4f}")
print(f" (MAPE): {mape:.4f}")
print(f"(R2): {r2:.4f}")
print("LR-GEP_predicted_vs_true_aqi_dynamic.csv")


plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, best_y_pred_real, alpha=0.7, edgecolor='black', s=60)
plt.plot([min(y_test_real), max(y_test_real)], [min(y_test_real), max(y_test_real)], 'k--', lw=2)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI (LR-GEP)")
plt.title(f"LR-GEP-R2 = {r2:.4f}", fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.savefig("lr-gep-r2_scatter_plot_gep.png", dpi=1200)
plt.show()
