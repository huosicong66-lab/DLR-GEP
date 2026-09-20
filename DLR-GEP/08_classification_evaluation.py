from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR / "results"
OUTPUT_ROOT = RESULT_ROOT / "Classification"
CITIES = ["Beijing", "Nanning"]
MODELS = ["LR", "DLR", "RF", "ARIMA", "BiLSTM", "GEP", "DLR-GEP"]

CLASS_LABELS = list(range(6))
CLASS_NAMES = ["Excellent", "Good", "Lightly polluted", "Moderately polluted",
               "Heavily polluted", "Severely polluted"]
AQI_BOUNDS = [-np.inf, 50, 100, 150, 200, 300, np.inf]

DATE_ALIASES = ["Target_Date", "Date", "target_date"]
TRUE_ALIASES = ["Observed_AQI", "Actual_AQI", "True_AQI", "Target_AQI", "y_true", "Observed"]
PRED_ALIASES = {
    "LR": ["LR_Prediction", "Predicted_AQI", "Prediction", "y_pred"],
    "DLR": ["DLR_Prediction", "DLR_Holdout_Prediction", "Predicted_AQI", "Prediction", "y_pred"],
    "RF": ["RF_Prediction", "Predicted_AQI", "Prediction", "y_pred"],
    "ARIMA": ["ARIMA_Prediction", "Predicted_AQI", "Prediction", "y_pred"],
    "BiLSTM": ["BiLSTM_Prediction", "Predicted_AQI", "Prediction", "y_pred"],
    "GEP": ["Improved_GEP_Prediction", "GEP_Prediction", "Predicted_AQI", "Prediction", "y_pred"],
    "DLR-GEP": ["DLR_GEP_Prediction", "Improved_GEP_Prediction", "Predicted_AQI", "Prediction", "y_pred"],
}


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(frame, aliases):
    lookup = {norm(c): c for c in frame.columns}
    for alias in aliases:
        if norm(alias) in lookup:
            return lookup[norm(alias)]
    return None


def explicit_candidates(city, model):
    table = {
        "LR": [
            ("LR", "test_predictions.csv"),
            ("LR", "LR_test_predictions.csv"),
        ],

        "DLR": [
            ("DLR", "test_predictions.csv"),
            ("DLR", "DLR_holdout_test_predictions.csv"),
        ],

        "RF": [
            ("RF-84", "test_predictions.csv"),
            ("RF", "test_predictions.csv"),
            ("RF", "RF_ensemble_test_predictions.csv"),
            ("RF", "RF_test_predictions.csv"),
        ],

        "ARIMA": [
            ("ARIMA", "test_predictions.csv"),
            ("ARIMA", "ARIMA_test_predictions.csv"),
        ],

        "BiLSTM": [
            ("BiLSTM", "test_predictions.csv"),
            ("BiLSTM", "BiLSTM_ensemble_test_predictions.csv"),
            ("BiLSTM", "BiLSTM_test_predictions.csv"),
        ],

        "GEP": [
            ("GEP", "GEP_ensemble_test_predictions.csv"),
            ("GEP", "test_predictions.csv"),
        ],

        "DLR-GEP": [
            ("DLR-GEP-Nested", "test_predictions.csv"),
            ("DLR-GEP", "test_predictions.csv"),
        ],
    }

    return [
        RESULT_ROOT / folder / city / filename
        for folder, filename in table[model]
    ]

def path_matches(path, city, model):


    try:
        relative = path.relative_to(RESULT_ROOT)
    except ValueError:
        return False

    parts = list(relative.parts)
    if len(parts) < 3:
        return False

    result_folder = norm(parts[0])
    city_folder = norm(parts[1])
    file_name = norm(path.name)
    s = norm(str(relative))

    if city_folder != norm(city) or "test" not in file_name or "predict" not in file_name:
        return False
    if any(x in s for x in ["classification", "statistical", "ablation", "validation"]):
        return False

    allowed_folders = {
        "LR": {"lr"},
        "DLR": {"dlr"},
        "RF": {"rf"},
        "ARIMA": {"arima", "arimabaseline"},
        "BiLSTM": {"bilstm", "bilstmbaseline"},
        "GEP": {"gep"},
        "DLR-GEP": {"dlrgep", "dlrgepnested"},
    }
    return result_folder in allowed_folders[model]


def discover_files(city, model):
    for path in explicit_candidates(city, model):
        if path.exists():
            return [path]
    paths = [p for p in RESULT_ROOT.rglob("*.csv") if path_matches(p, city, model)]
    if not paths:
        return []
    non_seed = [p for p in paths if "seed" not in norm(p.name)]
    if non_seed:
        non_seed.sort(key=lambda p: ("ensemble" not in norm(p.name), "final" not in norm(p.name), len(str(p))))
        return [non_seed[0]]
    return sorted(paths)


def prepared_test(city):
    folder = SCRIPT_DIR / "prepared_data" / city.lower()
    for name in ["test.csv", "test_scaled.csv", "testing.csv"]:
        path = folder / name
        if path.exists():
            return pd.read_csv(path)
    return None


def read_one(path, city, model):
    frame = pd.read_csv(path)
    date_col = find_column(frame, DATE_ALIASES)
    true_col = find_column(frame, TRUE_ALIASES)
    pred_col = find_column(frame, PRED_ALIASES[model])
    fallback = prepared_test(city)
    if pred_col is None:
        raise ValueError(f"{path.name} does not contain a prediction column. Available columns: {list(frame.columns)}")
    out = pd.DataFrame({"Predicted_AQI": pd.to_numeric(frame[pred_col], errors="coerce")})
    if date_col:
        out["Target_Date"] = pd.to_datetime(frame[date_col], errors="coerce")
    elif fallback is not None and len(fallback) == len(frame):
        c = find_column(fallback, DATE_ALIASES)
        out["Target_Date"] = pd.to_datetime(fallback[c], errors="coerce") if c else np.arange(len(frame))
    else:
        out["Target_Date"] = np.arange(len(frame))
    if true_col:
        out["Observed_AQI"] = pd.to_numeric(frame[true_col], errors="coerce")
    elif fallback is not None and len(fallback) == len(frame):
        c = find_column(fallback, TRUE_ALIASES)
        if c is None: raise ValueError(f"{path.name} and the prepared test set do not contain an observed AQI column")
        out["Observed_AQI"] = pd.to_numeric(fallback[c], errors="coerce")
    else:
        raise ValueError(f"{path.name} does not contain an observed AQI column")
    return out.dropna().drop_duplicates("Target_Date").sort_values("Target_Date")


def load_prediction(city, model):
    paths = discover_files(city, model)
    if not paths:
        return None, []
    frames = [read_one(p, city, model) for p in paths]
    if len(frames) == 1:
        return frames[0], paths
    base = frames[0].rename(columns={"Predicted_AQI": "p0"})
    for i, frame in enumerate(frames[1:], 1):
        part = frame[["Target_Date", "Observed_AQI", "Predicted_AQI"]].rename(
            columns={"Observed_AQI": f"y{i}", "Predicted_AQI": f"p{i}"})
        base = base.merge(part, on="Target_Date", how="inner")
    pred_cols = [c for c in base if c.startswith("p")]
    return pd.DataFrame({"Target_Date": base["Target_Date"], "Observed_AQI": base["Observed_AQI"],
                         "Predicted_AQI": base[pred_cols].median(axis=1)}), paths


def aqi_class(values):
    clipped = np.maximum(np.asarray(values, dtype=float), 0)
    categories = pd.cut(
        clipped,
        bins=AQI_BOUNDS,
        labels=CLASS_LABELS,
        right=True,
        include_lowest=True,
    )


    return np.asarray(categories, dtype=int)


def metrics(y_true, y_pred):
    observed = np.unique(y_true)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=observed,
                                                     average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=observed,
                                                     average="weighted", zero_division=0)
    return {"Accuracy": accuracy_score(y_true, y_pred), "Macro_Precision": mp,
            "Macro_Recall": mr, "Macro_F1": mf, "Weighted_Precision": wp,
            "Weighted_Recall": wr, "Weighted_F1": wf,
            "Observed_Class_Count": len(observed)}


def plot_matrix(matrix, title, path, normalized=False):
    values = matrix.astype(float)
    if normalized:
        den = values.sum(axis=1, keepdims=True)
        values = np.divide(values, den, out=np.zeros_like(values), where=den != 0)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(6), yticks=np.arange(6), xlabel="Predicted AQI category",
           ylabel="Observed AQI category", title=title)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    threshold = values.max() / 2 if values.size else 0
    for i in range(6):
        for j in range(6):
            text = f"{values[i, j]:.2f}" if normalized else str(int(values[i, j]))
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if values[i, j] > threshold else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    canonical = OUTPUT_ROOT / "canonical_predictions"; canonical.mkdir(exist_ok=True)
    rows, per_class_rows = [], []
    for city in CITIES:
        print(f"\n{'=' * 72}\nClassification evaluation: {city}\n{'=' * 72}")
        for model in MODELS:
            try:
                data, paths = load_prediction(city, model)
                if data is None:
                    print(f"Skip {model}: no test prediction file found"); continue
                y_true, y_pred = aqi_class(data["Observed_AQI"]), aqi_class(data["Predicted_AQI"])
                result = {"City": city, "Model": model, "N": len(data), **metrics(y_true, y_pred)}
                rows.append(result)
                data.assign(Observed_Category=y_true, Predicted_Category=y_pred).to_csv(
                    canonical / f"{city}_{model.replace('-', '_')}.csv", index=False, encoding="utf-8-sig")
                cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
                p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_LABELS, zero_division=0)
                for k in CLASS_LABELS:
                    per_class_rows.append({"City": city, "Model": model, "Category": CLASS_NAMES[k],
                                           "Precision": p[k], "Recall": r[k], "F1": f[k], "Support": int(support[k])})
                if model == "DLR-GEP":
                    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
                        OUTPUT_ROOT / f"{city}_DLR_GEP_confusion_matrix.csv", encoding="utf-8-sig")
                    plot_matrix(cm, f"{city} - DLR-GEP confusion matrix",
                                OUTPUT_ROOT / f"{city}_DLR_GEP_confusion_matrix.png")
                    plot_matrix(cm, f"{city} - DLR-GEP normalized confusion matrix",
                                OUTPUT_ROOT / f"{city}_DLR_GEP_confusion_matrix_normalized.png", True)
                print(f"{model}: N={len(data)}, Accuracy={result['Accuracy']:.4f}, Macro-F1={result['Macro_F1']:.4f}")
                print("  source:", ", ".join(str(p) for p in paths))
            except Exception as exc:
                print(f"Skip {model}: {exc}")
    if not rows:
        raise RuntimeError("No valid prediction files were found.")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_ROOT / "classification_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class_rows).to_csv(OUTPUT_ROOT / "classification_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT_ROOT / 'classification_metrics.csv'}\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
