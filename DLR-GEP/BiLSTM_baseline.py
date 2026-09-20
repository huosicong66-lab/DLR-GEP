from __future__ import annotations

import json
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "prepared_data"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "BiLSTM"
CITY_DIRECTORIES = {
    "Beijing": DATA_ROOT / "beijing",
    "Nanning": DATA_ROOT / "nanning",
}

VARIABLES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "AQI"]
WINDOW = 7
SEEDS = [42, 52, 62, 72, 82]
EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 25
LEARNING_RATE = 0.001
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
        "MSE": float(mse),
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE(%)": mape,
        "R2": float(r2_score(y_true, y_pred)),
    }


def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame.loc[:, ~frame.columns.astype(str).str.lower().str.startswith("unnamed")]


def sequence_columns() -> List[str]:

    return [f"{variable}_lag{lag}" for lag in range(WINDOW - 1, -1, -1) for variable in VARIABLES]


def read_sequence(path: Path) -> np.ndarray:
    frame = read_frame(path)
    columns = sequence_columns()
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing BiLSTM sequence features: {missing}")
    flat = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"{path.name} contains missing or nonnumeric values")
    return flat.reshape(len(flat), WINDOW, len(VARIABLES))


def read_target(path: Path) -> np.ndarray:
    frame = read_frame(path)
    if "Target_AQI" not in frame.columns:
        raise ValueError(f"{path.name} is missing Target_AQI")
    target = pd.to_numeric(frame["Target_AQI"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(target)):
        raise ValueError(f"{path.name} contains invalid targets")
    return target.reshape(-1)


def read_dates(city_dir: Path, split: str, n: int) -> np.ndarray:
    path = city_dir / f"{split}.csv"
    if path.exists():
        frame = read_frame(path)
        if "Target_Date" in frame.columns and len(frame) == n:
            return frame["Target_Date"].astype(str).to_numpy()
    return np.arange(n).astype(str)


def load_city(city_dir: Path) -> Tuple:
    X_train = read_sequence(city_dir / "X_train.csv")
    y_train = read_target(city_dir / "y_train.csv")
    X_val = read_sequence(city_dir / "X_validation.csv")
    y_val = read_target(city_dir / "y_validation.csv")
    X_test = read_sequence(city_dir / "X_test.csv")
    y_test = read_target(city_dir / "y_test.csv")
    if not (len(X_train) == len(y_train) and len(X_val) == len(y_val) and len(X_test) == len(y_test)):
        raise ValueError("X and y have different sample counts")
    return (
        X_train, y_train, X_val, y_val, X_test, y_test,
        read_dates(city_dir, "validation", len(y_val)),
        read_dates(city_dir, "test", len(y_test)),
    )


def set_seed(seed: int, tf) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_model(tf, input_shape: Tuple[int, int]):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(inputs)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=False)
    )(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs, name="BiLSTM_AQI")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def process_city(city: str, city_dir: Path, tf) -> Tuple[List[Dict], Dict]:
    print("\n" + "=" * 74)
    print(f"Processing city: {city}")
    print("=" * 74)
    X_train, y_train, X_val, y_val, X_test, y_test, val_dates, test_dates = load_city(city_dir)
    print(f"Sequence shape: {X_train.shape[1:]} (7 days x 7 variables)")
    print(f"Training={len(y_train)}, validation={len(y_val)}, testing={len(y_test)}")
    output = OUTPUT_ROOT / city
    output.mkdir(parents=True, exist_ok=True)


    y_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
    val_predictions, test_predictions, run_rows = [], [], []

    for run, seed in enumerate(SEEDS, 1):
        print(f"\nRun {run}/5, seed={seed}")
        tf.keras.backend.clear_session()
        set_seed(seed, tf)
        model = build_model(tf, X_train.shape[1:])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE,
                restore_best_weights=True, min_delta=1e-6,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8,
                min_lr=1e-5, verbose=0,
            ),
        ]
        started = time.perf_counter()
        history = model.fit(
            X_train, y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            shuffle=False, verbose=0, callbacks=callbacks,
        )
        val_scaled = model.predict(X_val, verbose=0)
        test_scaled = model.predict(X_test, verbose=0)
        val_pred = np.maximum(y_scaler.inverse_transform(val_scaled).reshape(-1), 0.0)
        test_pred = np.maximum(y_scaler.inverse_transform(test_scaled).reshape(-1), 0.0)
        result = calculate_metrics(y_test, test_pred)
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)
        run_rows.append({
            "City": city, "Model": "BiLSTM", "Run": run, "Seed": seed,
            "Epochs_Trained": len(history.history["loss"]),
            **result, "Time_seconds": time.perf_counter() - started,
        })
        pd.DataFrame(history.history).to_csv(
            output / f"training_history_seed_{seed}.csv", index=False, encoding="utf-8-sig"
        )
        print(f"  epochs={len(history.history['loss'])}, RMSE={result['RMSE']:.6f}, R2={result['R2']:.6f}")


    val_ensemble = np.mean(np.column_stack(val_predictions), axis=1)
    test_ensemble = np.mean(np.column_stack(test_predictions), axis=1)
    final_metrics = calculate_metrics(y_test, test_ensemble)
    run_frame = pd.DataFrame(run_rows)
    run_frame.to_csv(output / "BiLSTM_independent_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "Target_Date": val_dates, "Observed_AQI": y_val,
        "Predicted_AQI": val_ensemble, "Error": y_val - val_ensemble,
    }).to_csv(output / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "Target_Date": test_dates, "Observed_AQI": y_test,
        "Predicted_AQI": test_ensemble, "Error": y_test - test_ensemble,
    }).to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")
    stability = []
    for metric_name in ["MSE", "RMSE", "MAE", "MAPE(%)", "R2"]:
        stability.append({
            "City": city, "Model": "BiLSTM", "Metric": metric_name,
            "Mean": float(run_frame[metric_name].mean()),
            "Std": float(run_frame[metric_name].std(ddof=0)),
            "Minimum": float(run_frame[metric_name].min()),
            "Maximum": float(run_frame[metric_name].max()),
            "Number_of_Runs": len(SEEDS),
        })
    final_row = {
        "City": city, "Model": "BiLSTM-Ensemble", "Window": WINDOW,
        "Sequence_Features": len(VARIABLES), "Runs": len(SEEDS), **final_metrics,
    }
    print("\nBiLSTM ensemble:", final_metrics)
    return stability, final_row


def main() -> None:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is not installed. Run: pip install tensorflow"
        ) from exc
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Input directory: {DATA_ROOT}")
    print(f"Output directory: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stability_rows, final_rows = [], []
    for city, city_dir in CITY_DIRECTORIES.items():
        if not city_dir.exists():
            print(f"Skipping {city}: not found: {city_dir}")
            continue
        stability, final_row = process_city(city, city_dir, tf)
        stability_rows.extend(stability)
        final_rows.append(final_row)
    stability_frame = pd.DataFrame(stability_rows)
    final_frame = pd.DataFrame(final_rows)
    stability_frame.to_csv(OUTPUT_ROOT / "BiLSTM_stability_summary.csv", index=False, encoding="utf-8-sig")
    final_frame.to_csv(OUTPUT_ROOT / "BiLSTM_final_summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 74)
    print("BiLSTM final summary")
    print("=" * 74)
    print(final_frame.to_string(index=False))


if __name__ == "__main__":
    main()
