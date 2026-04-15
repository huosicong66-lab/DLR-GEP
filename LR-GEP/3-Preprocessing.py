import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("normalized_aqi_samples.csv")

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=pollutants, inplace=True)
df_original = df.copy()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_original[pollutants])
plt.title("Before Data Preprocessing")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot_before_pollutants.png", dpi=300)
plt.close()

def detect_z_outliers(data, col, threshold=3):
    if data[col].std() == 0:
        return pd.Series([False] * len(data))
    z_scores = np.abs(stats.zscore(data[col]))
    return z_scores > threshold

before_counts = {}
after_counts = {}

for col in pollutants:
    is_outlier = detect_z_outliers(df, col)
    before_counts[col] = is_outlier.sum()

    train_df = df.loc[~is_outlier].copy()
    predict_df = df.loc[is_outlier].copy()

    if train_df.empty or predict_df.empty:
        after_counts[col] = 0
        continue

    X_train = train_df.drop(columns=[col])
    y_train = train_df[col]
    X_predict = predict_df.drop(columns=[col])

    non_numeric_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    constant_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
    drop_cols = non_numeric_cols + constant_cols
    X_train = X_train.drop(columns=drop_cols)
    X_predict = X_predict.drop(columns=drop_cols)

    if X_train.shape[1] == 0:
        after_counts[col] = is_outlier.sum()
        continue

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_predict)

    df.loc[is_outlier, col] = predicted_values
    after_counts[col] = detect_z_outliers(df, col).sum()

comparison_df = pd.DataFrame({
    'Feature': pollutants,
    'Z_Outliers_Before': [before_counts[col] for col in pollutants],
    'Z_Outliers_After': [after_counts[col] for col in pollutants],
    'Fixed_Count': [before_counts[col] - after_counts[col] for col in pollutants]
})
comparison_df.to_csv("Z_outlier_comparison_LR_pollutants.csv", index=False, encoding='utf-8-sig')

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[pollutants])
plt.title("After Data Preprocessing")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot_after_pollutants_LR.png", dpi=300)
plt.close()

df.to_csv("F-AQI-Zscore-LR-pollutants.csv", index=False, encoding='utf-8-sig')
print("AQI remains unchanged. Pollutant features have been corrected. File saved: F-AQI-Zscore-LR-pollutants.csv")