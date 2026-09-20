from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "prepared_data"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "LR"

CITY_DIRECTORIES = {
    "Beijing": DATA_ROOT / "beijing",
    "Nanning": DATA_ROOT / "nanning",
}


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    y_true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(-1)

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = math.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    valid_mask = np.abs(y_true) > 1e-8

    if np.any(valid_mask):
        mape = np.mean(
            np.abs(
                (
                    y_true[valid_mask]
                    - y_pred[valid_mask]
                )
                / y_true[valid_mask]
            )
        ) * 100.0
    else:
        mape = np.nan

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "R2": r2,
    }


def read_feature_names(
    city_directory: Path,
) -> List[str]:
    feature_file = (
        city_directory / "feature_names.json"
    )

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature-name file not found: {feature_file}"
        )

    with open(
        feature_file,
        "r",
        encoding="utf-8",
    ) as file:
        content = json.load(file)

    if isinstance(content, list):
        feature_names = content

    elif isinstance(content, dict):
        feature_names = (
            content.get("feature_names")
            or content.get("features")
            or content.get("columns")
        )

    else:
        feature_names = None

    if not feature_names:
        raise ValueError(
            f"Unable to read feature names from {feature_file}."
        )

    return feature_names


def read_X(
    file_path: Path,
    feature_names: List[str],
) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
    )

    dataframe = dataframe.loc[
        :,
        ~dataframe.columns.astype(str)
        .str.lower()
        .str.startswith("unnamed")
    ]

    missing_features = [
        feature
        for feature in feature_names
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"{file_path.name} is missing "
            f"{len(missing_features)} features: \n"
            f"{missing_features}"
        )

    X = dataframe[
        feature_names
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.all(np.isfinite(X)):
        raise ValueError(
            f"{file_path.name} contains missing or nonnumeric values."
        )

    return X


def read_y(file_path: Path) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Target file not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
    )

    dataframe = dataframe.loc[
        :,
        ~dataframe.columns.astype(str)
        .str.lower()
        .str.startswith("unnamed")
    ]

    if "Target_AQI" in dataframe.columns:
        target_column = "Target_AQI"
    else:
        numeric_columns = dataframe.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if not numeric_columns:
            raise ValueError(
                f"{file_path.name} does not contain a numeric target column."
            )

        target_column = numeric_columns[-1]

        print(
            f"Warning: {file_path.name} does not contain Target_AQI; "
            f"using {target_column} as the target."
        )

    y = pd.to_numeric(
        dataframe[target_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.all(np.isfinite(y)):
        raise ValueError(
            f"{file_path.name} contains missing or nonnumeric target values."
        )

    return y.reshape(-1)


def read_target_dates(
    city_directory: Path,
    split: str,
    expected_length: int,
) -> np.ndarray:


    combined_file = (
        city_directory / f"{split}.csv"
    )

    if not combined_file.exists():
        return np.arange(expected_length).astype(str)

    dataframe = pd.read_csv(
        combined_file,
        encoding="utf-8-sig",
    )

    if (
        "Target_Date" in dataframe.columns
        and len(dataframe) == expected_length
    ):
        return dataframe[
            "Target_Date"
        ].astype(str).to_numpy()

    return np.arange(expected_length).astype(str)


def load_city_data(
    city_directory: Path,
) -> Tuple:
    feature_names = read_feature_names(
        city_directory
    )

    X_train = read_X(
        city_directory / "X_train.csv",
        feature_names,
    )

    y_train = read_y(
        city_directory / "y_train.csv"
    )

    X_validation = read_X(
        city_directory / "X_validation.csv",
        feature_names,
    )

    y_validation = read_y(
        city_directory / "y_validation.csv"
    )

    X_test = read_X(
        city_directory / "X_test.csv",
        feature_names,
    )

    y_test = read_y(
        city_directory / "y_test.csv"
    )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training X and y have different sample counts."
        )

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "Validation X and y have different sample counts."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test X and y have different sample counts."
        )

    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            "The training feature count does not match feature_names.json."
        )

    validation_dates = read_target_dates(
        city_directory,
        "validation",
        len(y_validation),
    )

    test_dates = read_target_dates(
        city_directory,
        "test",
        len(y_test),
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_names,
        validation_dates,
        test_dates,
    )


def process_city(
    city: str,
    city_directory: Path,
) -> List[dict]:
    print("\n" + "=" * 70)
    print(f"Processing city: {city}")
    print("=" * 70)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_names,
        validation_dates,
        test_dates,
    ) = load_city_data(city_directory)

    print(
        f"Number of input features: "
        f"{X_train.shape[1]}"
    )
    print(f"Training samples: {len(y_train)}")
    print(
        f"Validation samples: {len(y_validation)}"
    )
    print(f"Testing samples: {len(y_test)}")


    forbidden_features = [
        feature
        for feature in feature_names
        if feature.lower() in {
            "target_aqi",
            "targetaqi",
            "aqi_t+1",
            "aqi_t1",
        }
    ]

    if forbidden_features:
        raise ValueError(
            "Future AQI target features detected: "
            f"{forbidden_features}"
        )


    historical_aqi_features = [
        feature
        for feature in feature_names
        if feature.startswith("AQI_")
    ]

    print(
        f"Historical AQI features: "
        f"{len(historical_aqi_features)}"
    )


    model = LinearRegression()
    model.fit(
        X_train,
        y_train,
    )

    validation_predictions = model.predict(
        X_validation
    )

    test_predictions = model.predict(
        X_test
    )


    validation_predictions = np.maximum(
        validation_predictions,
        0.0,
    )

    test_predictions = np.maximum(
        test_predictions,
        0.0,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    print("\nValidation metrics:")

    for metric, value in validation_metrics.items():
        print(f"  {metric}: {value:.6f}")

    print("\nTest metrics:")

    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")

    city_output_directory = (
        OUTPUT_ROOT / city
    )

    city_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_output = pd.DataFrame({
        "Target_Date": validation_dates,
        "Observed_AQI": y_validation,
        "Predicted_AQI": validation_predictions,
        "Error": (
            y_validation
            - validation_predictions
        ),
    })

    validation_output.to_csv(
        city_output_directory
        / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_output = pd.DataFrame({
        "Target_Date": test_dates,
        "Observed_AQI": y_test,
        "Predicted_AQI": test_predictions,
        "Error": y_test - test_predictions,
    })

    test_output.to_csv(
        city_output_directory
        / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coefficient_table = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_,
        "Absolute_Coefficient":
            np.abs(model.coef_),
    }).sort_values(
        "Absolute_Coefficient",
        ascending=False,
    )

    coefficient_table.to_csv(
        city_output_directory
        / "feature_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_information = {
        "City": city,
        "Model": "LR",
        "Number_of_Features":
            len(feature_names),
        "Intercept": float(model.intercept_),
        "Training_Samples":
            len(y_train),
        "Validation_Samples":
            len(y_validation),
        "Testing_Samples":
            len(y_test),
        "Target": "AQI on day t+1",
        "Historical_AQI_Features":
            historical_aqi_features,
        "Test_Data_Used_for_Training": False,
    }

    with open(
        city_output_directory
        / "model_information.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model_information,
            file,
            ensure_ascii=False,
            indent=2,
        )

    rows = [
        {
            "City": city,
            "Model": "LR",
            "Dataset": "Validation",
            "Number_of_Features":
                len(feature_names),
            **validation_metrics,
        },
        {
            "City": city,
            "Model": "LR",
            "Dataset": "Test",
            "Number_of_Features":
                len(feature_names),
            **test_metrics,
        },
    ]

    print(
        f"\nResults saved to: "
        f"{city_output_directory}"
    )

    return rows


def main():
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Input directory: {DATA_ROOT}")
    print(f"Output directory: {OUTPUT_ROOT}")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    for city, city_directory in (
        CITY_DIRECTORIES.items()
    ):
        if not city_directory.exists():
            print(
                f"\nSkipping {city}: data directory not found"
            )
            print(city_directory)
            continue

        try:
            rows = process_city(
                city,
                city_directory,
            )

            all_rows.extend(rows)

        except Exception as error:
            print(f"\n{city} failed: {error}")

    if not all_rows:
        print("\nNo LR results were generated.")
        return

    summary = pd.DataFrame(all_rows)

    summary_file = (
        OUTPUT_ROOT
        / "LR_all_cities_metrics.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("LR baseline summary")
    print("=" * 70)
    print(summary.to_string(index=False))

    print(
        f"\nAll-city summary saved to: "
        f"{summary_file}"
    )


if __name__ == "__main__":
    main()
