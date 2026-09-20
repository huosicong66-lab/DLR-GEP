from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "prepared_data"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "ARIMA"
CITY_DIRECTORIES = {
    "Beijing": DATA_ROOT / "beijing",
    "Nanning": DATA_ROOT / "nanning",
}

ORDER_CANDIDATES = [
    (0, 1, 1), (1, 0, 0), (1, 0, 1), (2, 0, 0), (2, 0, 1),
    (3, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1),
]
EPS = 1e-8


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
        "MSE": float(mse), "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE(%)": mape, "R2": float(r2_score(y_true, y_pred)),
    }


def read_target(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "Target_AQI" not in frame.columns:
        raise ValueError(f"{path.name} is missing Target_AQI")
    values = pd.to_numeric(frame["Target_AQI"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path.name} contains invalid targets")
    return values.reshape(-1)


def read_dates(city_dir: Path, split: str, n: int) -> np.ndarray:
    path = city_dir / f"{split}.csv"
    if path.exists():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "Target_Date" in frame.columns and len(frame) == n:
            return frame["Target_Date"].astype(str).to_numpy()
    return np.arange(n).astype(str)


def load_city(city_dir: Path) -> Tuple:
    train = read_target(city_dir / "y_train.csv")
    validation = read_target(city_dir / "y_validation.csv")
    test = read_target(city_dir / "y_test.csv")
    return (
        train, validation, test,
        read_dates(city_dir, "validation", len(validation)),
        read_dates(city_dir, "test", len(test)),
    )


def fit_arima(history: np.ndarray, order: Tuple[int, int, int]):
    from statsmodels.tsa.arima.model import ARIMA
    return ARIMA(
        history,
        order=order,
        trend="n" if order[1] > 0 else "c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()


def walk_forward(fitted, observations: np.ndarray) -> Tuple[np.ndarray, object]:
    predictions = []
    state = fitted
    for observed in observations:
        prediction = float(np.asarray(state.forecast(steps=1)).reshape(-1)[0])
        predictions.append(max(prediction, 0.0))

        state = state.append([float(observed)], refit=False)
    return np.asarray(predictions), state


def select_order(train: np.ndarray, validation: np.ndarray) -> Tuple[Tuple[int, int, int], pd.DataFrame]:
    rows: List[Dict] = []
    best_order, best_rmse = None, np.inf
    for order in ORDER_CANDIDATES:
        try:
            fitted = fit_arima(train, order)
            prediction, _ = walk_forward(fitted, validation)
            result = calculate_metrics(validation, prediction)
            rows.append({"p": order[0], "d": order[1], "q": order[2], "Status": "Success", **result})
            print(f"  ARIMA{order}: validation RMSE={result['RMSE']:.6f}")
            if result["RMSE"] < best_rmse:
                best_rmse, best_order = result["RMSE"], order
        except Exception as error:
            rows.append({"p": order[0], "d": order[1], "q": order[2], "Status": f"Failed: {error}"})
            print(f"  ARIMA{order}: failed")
    if best_order is None:
        raise RuntimeError("All ARIMA candidate models failed to fit")
    return best_order, pd.DataFrame(rows)


def process_city(city: str, city_dir: Path) -> Dict:
    print("\n" + "=" * 72)
    print(f"Processing city: {city}")
    print("=" * 72)
    train, validation, test, val_dates, test_dates = load_city(city_dir)
    print(f"Training={len(train)}, validation={len(validation)}, testing={len(test)}")
    output = OUTPUT_ROOT / city
    output.mkdir(parents=True, exist_ok=True)

    selected_order, search = select_order(train, validation)
    search.to_csv(output / "ARIMA_validation_search.csv", index=False, encoding="utf-8-sig")
    print(f"Selected order: ARIMA{selected_order}")


    fitted = fit_arima(train, selected_order)
    validation_prediction, state_after_validation = walk_forward(fitted, validation)
    test_prediction, _ = walk_forward(state_after_validation, test)
    validation_metrics = calculate_metrics(validation, validation_prediction)
    test_metrics = calculate_metrics(test, test_prediction)

    pd.DataFrame({
        "Target_Date": val_dates, "Observed_AQI": validation,
        "Predicted_AQI": validation_prediction, "Error": validation - validation_prediction,
    }).to_csv(output / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "Target_Date": test_dates, "Observed_AQI": test,
        "Predicted_AQI": test_prediction, "Error": test - test_prediction,
    }).to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")

    print("Validation metrics:", validation_metrics)
    print("Test metrics:", test_metrics)
    return {
        "City": city, "Model": f"ARIMA{selected_order}",
        "p": selected_order[0], "d": selected_order[1], "q": selected_order[2],
        "Protocol": "One-step walk-forward, fixed parameters", **test_metrics,
    }


def main() -> None:
    try:
        import statsmodels
    except ImportError as exc:
        raise ImportError("statsmodels is not installed. Run: pip install statsmodels") from exc
    print(f"Input directory: {DATA_ROOT}")
    print(f"Output directory: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for city, city_dir in CITY_DIRECTORIES.items():
        if not city_dir.exists():
            print(f"Skipping {city}: not found: {city_dir}")
            continue
        rows.append(process_city(city, city_dir))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_ROOT / "ARIMA_final_summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 72)
    print("ARIMA final summary")
    print("=" * 72)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
