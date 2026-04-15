import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv("beijing_aqi_2022_2024_combined.csv", encoding='utf-8')
if 'date' not in df.columns:
    raise ValueError("Missing 'date' column. Cannot preserve time information.")
dates = df['date'].copy()

pollutants = ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']
target = 'AQI'

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=pollutants + [target], inplace=True)

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[pollutants + [target]])
normalized_df = pd.DataFrame(normalized_data, columns=pollutants + [target])

normalized_df.insert(0, 'date', dates.loc[normalized_df.index].values)

print("First 10 rows after normalization:\n")
print(normalized_df.head(10))
print("\nTotal sample count:", len(normalized_df))
normalized_df.to_csv("normalized_aqi_samples.csv", index=False, encoding='utf-8')
print(" Normalized sample Data saved as normalized_aqi_samples.csv")

X = normalized_df[pollutants]
y = normalized_df[target]
lr_model = LinearRegression()
lr_model.fit(X, y)

coefficients = lr_model.coef_
intercept = lr_model.intercept_

print("\nMultivariate Linear Regression Equation:")
for feature, coef in zip(pollutants, coefficients):
    print(f"{feature} × {coef:.4f}", end=" + ")
print(f"Intercept = {intercept:.4f}")

coef_df = pd.DataFrame({'Feature': pollutants, 'Coefficient': coefficients})
coef_df['Intercept'] = intercept
coef_df.to_csv("linear_regression_coefficients.csv", index=False, encoding='utf-8')
print("Linear regression coefficients saved as linear_regression_coefficients.csv")

y_pred = lr_model.predict(X)
R = np.corrcoef(y, y_pred)[0, 1]
print(f"\nPearson correlation coefficient R: {R:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.6, color='mediumseagreen', edgecolor='black')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('AQI')
plt.ylabel('Feature')
plt.title(f'Feature AQI\nR = {R:.4f}', fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.savefig("model_r_scatter.png", dpi=300)
plt.show()

correlation_df = pd.DataFrame(columns=['Feature', 'Correlation'])
for feature in pollutants:
    corr_value = normalized_df[feature].corr(normalized_df[target])
    correlation_df = correlation_df.append({'Feature': feature, 'Correlation': corr_value}, ignore_index=True)

correlation_df['AbsCorrelation'] = correlation_df['Correlation'].abs()
correlation_df.sort_values(by='AbsCorrelation', ascending=False, inplace=True)

print("\nFeature-AQI Pearson correlation coefficients (sorted by absolute value):")
print(correlation_df[['Feature', 'Correlation']])
correlation_df.to_csv("feature_correlation_with_AQI.csv", index=False, encoding='utf-8')
print(" Correlation table saved as feature_correlation_with_AQI.csv")

colors = ['blue', 'green', 'orange', 'purple', 'teal', 'goldenrod']
for i, feature in enumerate(pollutants):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X[feature], y,
        color=colors[i % len(colors)],
        s=60,
        alpha=0.8
    )
    plt.xlabel(f'{feature} Concentration')
    plt.ylabel('Normalized AQI')
    plt.title(f'{feature} vs AQI', fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Linear_Regression_{feature}_vs_AQI.png", dpi=300)
    plt.show()
