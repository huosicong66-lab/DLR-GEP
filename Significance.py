import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

# =========================
# 1. 读取数据
# =========================

# 你的主模型（DLR-GEP）
dlr_gep = pd.read_csv("LR-GEP_predicted_vs_true_aqi_dynamic.csv")

# 其他模型
models = {
    "GNN": pd.read_csv("gnn_selfreg_aqi.csv"),
    "LR": pd.read_csv("lr_selfreg_aqi.csv"),
    "GP": pd.read_csv("gp_selfreg_aqi.csv"),
    "GA": pd.read_csv("ga_selfreg_aqi.csv"),
    "GEP": pd.read_csv("gep_selfreg_aqi.csv"),
    "RF": pd.read_csv("rf_selfreg_aqi.csv"),
    "LR-GA": pd.read_csv("lr_ga_nextday_results.csv"),
    "LR-GP": pd.read_csv("lr_gp_nextday_results.csv"),
}

# =========================
# 2. 自动识别列名（防止不一致）
# =========================

def get_columns(df):
    cols = df.columns
    # 尝试匹配
    true_col = [c for c in cols if "true" in c.lower() or "actual" in c.lower()][0]
    pred_col = [c for c in cols if "pred" in c.lower()][0]
    return true_col, pred_col

# DLR-GEP列
true_col, pred_col = get_columns(dlr_gep)

y_true = dlr_gep[true_col].values
y_pred_main = dlr_gep[pred_col].values

# 主模型误差
error_main = np.abs(y_true - y_pred_main)

# =========================
# 3. 统计检验
# =========================

results = []

for name, df in models.items():
    t_col, p_col = get_columns(df)

    y_pred = df[p_col].values

    # 保证长度一致
    min_len = min(len(y_true), len(y_pred))
    e1 = error_main[:min_len]
    e2 = np.abs(y_true[:min_len] - y_pred[:min_len])

    # 配对t检验
    t_stat, p_value = ttest_rel(e1, e2)

    results.append([name, p_value])

# =========================
# 4. 输出结果
# =========================

results_df = pd.DataFrame(results, columns=["Model", "p-value"])

# 显著性标记
def significance(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

results_df["Significance"] = results_df["p-value"].apply(significance)

print(results_df)

# 保存
results_df.to_csv("statistical_significance_results.csv", index=False)

print("\n 已生成统计显著性结果表：statistical_significance_results.csv")