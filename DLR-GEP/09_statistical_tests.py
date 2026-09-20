from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

SCRIPT_DIR = Path(__file__).resolve().parent
CANONICAL_ROOT = SCRIPT_DIR / "results" / "Classification" / "canonical_predictions"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "Statistical_Tests"
CITIES = ["Beijing", "Nanning"]
REFERENCE_MODEL = "DLR-GEP"
BASELINES = ["LR", "DLR", "RF", "ARIMA", "BiLSTM", "GEP"]


def canonical_path(city, model):
    return CANONICAL_ROOT / f"{city}_{model.replace('-', '_')}.csv"


def read_prediction(city, model):
    path = canonical_path(city, model)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 08_classification_evaluation.py first.")
    frame = pd.read_csv(path, usecols=["Target_Date", "Observed_AQI", "Predicted_AQI"])
    frame["Target_Date"] = pd.to_datetime(frame["Target_Date"], errors="coerce")
    for c in ["Observed_AQI", "Predicted_AQI"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame.dropna().drop_duplicates("Target_Date").sort_values("Target_Date")


def align(reference, baseline):
    merged = reference.merge(baseline, on="Target_Date", how="inner", suffixes=("_ref", "_base"))
    if len(merged) < 10:
        raise ValueError("Fewer than 10 common test dates.")
    if not np.allclose(merged["Observed_AQI_ref"], merged["Observed_AQI_base"], atol=1e-8):
        raise ValueError("Observed AQI values differ after date alignment.")
    return merged


def dm_test(y, pred_ref, pred_base, loss_name="Squared_Error", horizon=1):
    err_ref, err_base = y - pred_ref, y - pred_base
    if loss_name == "Squared_Error":
        d = err_ref ** 2 - err_base ** 2
    elif loss_name == "Absolute_Error":
        d = np.abs(err_ref) - np.abs(err_base)
    else:
        raise ValueError(loss_name)
    d = np.asarray(d, dtype=float); n = len(d); mean_d = float(np.mean(d))
    centered = d - mean_d
    gamma0 = float(np.dot(centered, centered) / n)
    long_run = gamma0
    for lag in range(1, min(horizon, n)):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run += 2.0 * gamma
    if long_run <= 0 or not np.isfinite(long_run):
        return np.nan, np.nan, mean_d
    dm = mean_d / np.sqrt(long_run / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_hln = float(dm * correction)
    p_value = float(2 * student_t.sf(abs(dm_hln), df=n - 1))
    return dm_hln, p_value, mean_d


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float); adjusted = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]
    if not len(valid): return adjusted
    order = valid[np.argsort(p[valid])]; m = len(order); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx]); adjusted[idx] = min(running, 1.0)
    return adjusted


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for city in CITIES:
        print(f"\n{'=' * 72}\nDM tests: {city}\n{'=' * 72}")
        reference = read_prediction(city, REFERENCE_MODEL)
        for baseline_name in BASELINES:
            try:
                baseline = read_prediction(city, baseline_name)
                data = align(reference, baseline)
            except Exception as exc:
                print(f"Skip {baseline_name}: {exc}"); continue
            y = data["Observed_AQI_ref"].to_numpy(float)
            pr = data["Predicted_AQI_ref"].to_numpy(float)
            pb = data["Predicted_AQI_base"].to_numpy(float)
            for loss in ["Squared_Error", "Absolute_Error"]:
                dm, raw_p, diff = dm_test(y, pr, pb, loss, 1)
                ref_loss = (y-pr)**2 if loss == "Squared_Error" else np.abs(y-pr)
                base_loss = (y-pb)**2 if loss == "Squared_Error" else np.abs(y-pb)
                rows.append({"City": city, "Reference_Model": REFERENCE_MODEL,
                             "Baseline_Model": baseline_name, "Loss": loss, "N": len(y),
                             "Reference_Mean_Loss": np.mean(ref_loss), "Baseline_Mean_Loss": np.mean(base_loss),
                             "Mean_Loss_Difference": diff, "DM_Statistic": dm, "Raw_P_Value": raw_p})
    if not rows: raise RuntimeError("No comparisons were available.")
    result = pd.DataFrame(rows)
    result["Holm_Adjusted_P_Value"] = np.nan
    for (_, _), idx in result.groupby(["City", "Loss"]).groups.items():
        result.loc[idx, "Holm_Adjusted_P_Value"] = holm_adjust(result.loc[idx, "Raw_P_Value"].to_numpy())
    result["Significant_0.05"] = result["Holm_Adjusted_P_Value"] < 0.05
    result["Better_Model"] = np.where(result["Mean_Loss_Difference"] < 0, REFERENCE_MODEL,
                                       np.where(result["Mean_Loss_Difference"] > 0, result["Baseline_Model"], "Equal"))
    result.to_csv(OUTPUT_ROOT / "statistical_tests_summary.csv", index=False, encoding="utf-8-sig")
    for city in CITIES:
        part = result[result.City == city]
        if len(part): part.to_csv(OUTPUT_ROOT / f"{city}_DM_tests.csv", index=False, encoding="utf-8-sig")
    print("\nInterpretation: negative loss difference means DLR-GEP has lower loss.")
    print(result.to_string(index=False))
    print(f"\nSaved: {OUTPUT_ROOT / 'statistical_tests_summary.csv'}")


if __name__ == "__main__":
    main()
