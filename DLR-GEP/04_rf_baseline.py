from __future__ import annotations

import json
import math
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "prepared_data"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "RF-84"

CITY_DIRECTORIES = {
    "Beijing": DATA_ROOT / "beijing",
    "Nanning": DATA_ROOT / "nanning",
}

SEEDS = [42, 52, 62, 72, 82]


N_ESTIMATORS = 500
MAX_DEPTH_VALUES = [None, 8, 12, 16]
MIN_SAMPLES_LEAF_VALUES = [1, 2, 4]
MAX_FEATURES_VALUES = ["sqrt", 0.5]
N_JOBS = -1
EPS = 1e-8


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    valid = np.abs(y_true) > EPS
    mape = (
        float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100)
        if np.any(valid)
        else float("nan")
    )
    return {
        "MSE": float(mse),
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE(%)": mape,
        "R2": float(r2_score(y_true, y_pred)),
    }


def feature_names(city_dir: Path) -> List[str]:
    with (city_dir / "feature_names.json").open("r", encoding="utf-8") as handle:
        content = json.load(handle)
    if isinstance(content, list):
        return list(content)
    if isinstance(content, dict):
        names = content.get("feature_names") or content.get("features") or content.get("columns")
        if names:
            return list(names)
    raise ValueError("Unable to read feature_names.json")


def read_X(path: Path, names: List[str]) -> np.ndarray:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.loc[:, ~frame.columns.astype(str).str.lower().str.startswith("unnamed")]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}  is missing features: {missing}")
    array = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path.name}  contains missing or nonnumeric values")
    return array


def read_y(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "Target_AQI" not in frame.columns:
        raise ValueError(f"{path.name}  is missing  Target_AQI")
    array = pd.to_numeric(frame["Target_AQI"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path.name} contains invalid targets")
    return array.reshape(-1)


def read_dates(city_dir: Path, split: str, n: int) -> np.ndarray:
    path = city_dir / f"{split}.csv"
    if path.exists():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "Target_Date" in frame.columns and len(frame) == n:
            return frame["Target_Date"].astype(str).to_numpy()
    return np.arange(n).astype(str)


def load_city(city_dir: Path) -> Tuple:
    names = feature_names(city_dir)
    X_train = read_X(city_dir / "X_train.csv", names)
    y_train = read_y(city_dir / "y_train.csv")
    X_val = read_X(city_dir / "X_validation.csv", names)
    y_val = read_y(city_dir / "y_validation.csv")
    X_test = read_X(city_dir / "X_test.csv", names)
    y_test = read_y(city_dir / "y_test.csv")
    if not (len(X_train) == len(y_train) and len(X_val) == len(y_val) and len(X_test) == len(y_test)):
        raise ValueError("X and y have different sample counts")
    forbidden = {"target_aqi", "targetaqi", "aqi_t+1", "aqit1"}
    if any(name.lower() in forbidden for name in names):
        raise ValueError("Future AQI target features detected; execution stopped")
    return (
        X_train, y_train, X_val, y_val, X_test, y_test, names,
        read_dates(city_dir, "validation", len(y_val)),
        read_dates(city_dir, "test", len(y_test)),
    )


def new_model(max_depth, min_leaf: int, max_features, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        max_features=max_features,
        random_state=seed,
        n_jobs=N_JOBS,
        bootstrap=True,
    )


def select_parameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Dict, pd.DataFrame]:
    rows = []
    for depth, leaf, max_features in product(
        MAX_DEPTH_VALUES,
        MIN_SAMPLES_LEAF_VALUES,
        MAX_FEATURES_VALUES,
    ):
        model = new_model(depth, leaf, max_features, seed=42)
        model.fit(X_train, y_train)
        prediction = np.maximum(model.predict(X_val), 0.0)
        result = metrics(y_val, prediction)
        rows.append({
            "n_estimators": N_ESTIMATORS,
            "max_depth": "None" if depth is None else depth,
            "min_samples_leaf": leaf,
            "max_features": max_features,
            **result,
        })
        print(
            f"  depth={str(depth):>4}, leaf={leaf}, features={str(max_features):>4}, "
            f"validation RMSE={result['RMSE']:.6f}"
        )
    table = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    best = table.iloc[0]
    depth_value = None if str(best["max_depth"]) == "None" else int(float(best["max_depth"]))
    max_features_value = best["max_features"]
    if str(max_features_value) != "sqrt":
        max_features_value = float(max_features_value)
    return {
        "max_depth": depth_value,
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": max_features_value,
    }, table


def process_city(city: str, city_dir: Path) -> Tuple[List[Dict], Dict]:
    print("\n" + "=" * 74)
    print(f"Processing city: {city}")
    print("=" * 74)
    X_train, y_train, X_val, y_val, X_test, y_test, names, val_dates, test_dates = load_city(city_dir)
    print(f"Features: {len(names)}; training: {len(y_train)}; validation: {len(y_val)}; testing: {len(y_test)}")
    output = OUTPUT_ROOT / city
    output.mkdir(parents=True, exist_ok=True)

    print("\nValidation hyperparameter search:")
    selected, search_table = select_parameters(X_train, y_train, X_val, y_val)
    search_table.to_csv(output / "RF_validation_search.csv", index=False, encoding="utf-8-sig")
    print(f"\nSelected parameters: {selected}")

    run_rows, val_predictions, test_predictions, importances = [], [], [], []
    for run, seed in enumerate(SEEDS, 1):
        started = time.perf_counter()
        model = new_model(
            selected["max_depth"],
            selected["min_samples_leaf"],
            selected["max_features"],
            seed,
        )
        model.fit(X_train, y_train)
        val_pred = np.maximum(model.predict(X_val), 0.0)
        test_pred = np.maximum(model.predict(X_test), 0.0)
        result = metrics(y_test, test_pred)
        run_rows.append({
            "City": city, "Model": "RF", "Run": run, "Seed": seed,
            **selected, **result, "Time_seconds": time.perf_counter() - started,
        })
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)
        importances.append(model.feature_importances_)
        print(f"  Run {run}/5 seed={seed}: RMSE={result['RMSE']:.6f}, R2={result['R2']:.6f}")


    val_ensemble = np.mean(np.column_stack(val_predictions), axis=1)
    test_ensemble = np.mean(np.column_stack(test_predictions), axis=1)
    validation_result = metrics(y_val, val_ensemble)
    test_result = metrics(y_test, test_ensemble)

    pd.DataFrame(run_rows).to_csv(output / "RF_independent_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "Target_Date": val_dates,
        "Observed_AQI": y_val,
        "Predicted_AQI": val_ensemble,
        "Error": y_val - val_ensemble,
    }).to_csv(output / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "Target_Date": test_dates,
        "Observed_AQI": y_test,
        "Predicted_AQI": test_ensemble,
        "Error": y_test - test_ensemble,
    }).to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "Feature": names,
        "Mean_Importance": np.mean(np.vstack(importances), axis=0),
        "Std_Importance": np.std(np.vstack(importances), axis=0),
    }).sort_values("Mean_Importance", ascending=False).to_csv(
        output / "RF_feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    metric_names = ["MSE", "RMSE", "MAE", "MAPE(%)", "R2"]
    stability = []
    run_frame = pd.DataFrame(run_rows)
    for metric_name in metric_names:
        stability.append({
            "City": city, "Model": "RF", "Metric": metric_name,
            "Mean": float(run_frame[metric_name].mean()),
            "Std": float(run_frame[metric_name].std(ddof=0)),
            "Minimum": float(run_frame[metric_name].min()),
            "Maximum": float(run_frame[metric_name].max()),
            "Number_of_Runs": len(SEEDS),
        })
    pd.DataFrame(stability).to_csv(output / "RF_stability_summary.csv", index=False, encoding="utf-8-sig")

    final_row = {
        "City": city, "Model": "RF-Ensemble", "Number_of_Features": len(names),
        "n_estimators": N_ESTIMATORS, **selected, **test_result,
    }
    print("\nValidation ensemble:", validation_result)
    print("Test ensemble:", test_result)
    return stability, final_row


def main() -> None:
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Input directory: {DATA_ROOT}")
    print(f"Output directory: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stability_rows, final_rows = [], []
    for city, city_dir in CITY_DIRECTORIES.items():
        if not city_dir.exists():
            print(f"Skipping  {city}: not found: {city_dir}")
            continue
        city_stability, final_row = process_city(city, city_dir)
        stability_rows.extend(city_stability)
        final_rows.append(final_row)
    stability_frame = pd.DataFrame(stability_rows)
    final_frame = pd.DataFrame(final_rows)
    stability_frame.to_csv(OUTPUT_ROOT / "RF_all_runs_summary.csv", index=False, encoding="utf-8-sig")
    final_frame.to_csv(OUTPUT_ROOT / "RF_final_summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 74)
    print("RF final summary")
    print("=" * 74)
    print(final_frame.to_string(index=False))


if __name__ == "__main__":
    main()
