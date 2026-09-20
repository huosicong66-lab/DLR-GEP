from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "prepared_data"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "DLR"

CITY_DIRECTORIES = {
    "Beijing": DATA_ROOT / "beijing",
    "Nanning": DATA_ROOT / "nanning",
}


RIDGE_ALPHA = 1.0


FORGETTING_FACTOR_CANDIDATES = [
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
    1.0,
]


COVARIANCE_SCALE_CANDIDATES = [
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    5.0,
]

EPSILON = 1e-10
MAX_ABSOLUTE_COEFFICIENT = 1e5
MAX_COVARIANCE_VALUE = 1e8


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
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
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE(%)": float(mape),
        "R2": float(r2),
    }


def read_feature_names(
    city_directory: Path,
) -> List[str]:
    file_path = (
        city_directory / "feature_names.json"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature-name file not found: {file_path}"
        )

    with open(
        file_path,
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
            f"Unable to read feature names: {file_path}"
        )

    return list(feature_names)


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
            f"{file_path.name} is missing features: "
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
            f"{file_path.name} contains invalid feature values."
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

    if "Target_AQI" not in dataframe.columns:
        raise ValueError(
            f"{file_path.name} is missing Target_AQI."
        )

    y = pd.to_numeric(
        dataframe["Target_AQI"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.all(np.isfinite(y)):
        raise ValueError(
            f"{file_path.name} contains invalid AQI targets."
        )

    return y.reshape(-1)


def read_target_dates(
    city_directory: Path,
    split: str,
    expected_length: int,
) -> np.ndarray:
    file_path = city_directory / f"{split}.csv"

    if not file_path.exists():
        return np.arange(
            expected_length
        ).astype(str)

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
    )

    if (
        "Target_Date" in dataframe.columns
        and len(dataframe) == expected_length
    ):
        return dataframe[
            "Target_Date"
        ].astype(str).to_numpy()

    return np.arange(
        expected_length
    ).astype(str)


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
            "The feature count does not match feature_names.json."
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


def initialize_rls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    covariance_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:


    ridge = Ridge(
        alpha=RIDGE_ALPHA,
        fit_intercept=True,
    )

    ridge.fit(
        X_train,
        y_train,
    )

    beta = np.concatenate([
        [float(ridge.intercept_)],
        np.asarray(
            ridge.coef_,
            dtype=float,
        ),
    ])

    number_of_parameters = len(beta)

    covariance = (
        covariance_scale
        * np.eye(
            number_of_parameters,
            dtype=float,
        )
    )

    return beta, covariance


def rls_predict(
    X: np.ndarray,
    y: Optional[np.ndarray],
    initial_beta: np.ndarray,
    initial_covariance: np.ndarray,
    forgetting_factor: float,
    update: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:


    beta = initial_beta.copy()
    covariance = initial_covariance.copy()

    predictions = []
    coefficient_history = []
    gain_norm_history = []

    for sample_index in range(len(X)):
        x_vector = np.concatenate([
            [1.0],
            X[sample_index],
        ])

        coefficient_history.append(
            beta.copy()
        )

        raw_prediction = float(
            np.dot(x_vector, beta)
        )


        prediction = max(
            raw_prediction,
            0.0,
        )

        predictions.append(prediction)

        if update:
            if y is None:
                raise ValueError(
                    "y is required when update=True."
                )

            covariance_x = (
                covariance @ x_vector
            )

            denominator = (
                forgetting_factor
                + np.dot(
                    x_vector,
                    covariance_x,
                )
            )

            denominator = max(
                float(denominator),
                EPSILON,
            )

            gain = (
                covariance_x / denominator
            )

            prediction_error = float(
                y[sample_index] - prediction
            )

            beta = (
                beta
                + gain * prediction_error
            )


            covariance = (
                covariance
                - np.outer(
                    gain,
                    x_vector,
                ) @ covariance
            ) / forgetting_factor


            covariance = 0.5 * (
                covariance + covariance.T
            )

            beta = np.clip(
                beta,
                -MAX_ABSOLUTE_COEFFICIENT,
                MAX_ABSOLUTE_COEFFICIENT,
            )

            covariance = np.clip(
                covariance,
                -MAX_COVARIANCE_VALUE,
                MAX_COVARIANCE_VALUE,
            )

            gain_norm_history.append(
                float(np.linalg.norm(gain))
            )

        else:
            gain_norm_history.append(0.0)

    return (
        np.asarray(
            predictions,
            dtype=float,
        ),
        beta,
        covariance,
        np.asarray(
            coefficient_history,
            dtype=float,
        ),
    )


def fixed_predict(
    X: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    X_design = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    prediction = X_design @ beta

    return np.maximum(
        prediction,
        0.0,
    )


def select_rls_parameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Tuple[
    str,
    float,
    float,
    pd.DataFrame,
]:


    search_rows = []


    fixed_beta, _ = initialize_rls(
        X_train,
        y_train,
        covariance_scale=1.0,
    )

    fixed_validation_prediction = fixed_predict(
        X_validation,
        fixed_beta,
    )

    fixed_metrics = calculate_metrics(
        y_validation,
        fixed_validation_prediction,
    )

    search_rows.append({
        "Mode": "Fixed",
        "Forgetting_Factor": 1.0,
        "Covariance_Scale": 0.0,
        **fixed_metrics,
    })

    best_mode = "Fixed"
    best_forgetting_factor = 1.0
    best_covariance_scale = 0.0
    best_rmse = fixed_metrics["RMSE"]


    for covariance_scale in (
        COVARIANCE_SCALE_CANDIDATES
    ):
        initial_beta, initial_covariance = (
            initialize_rls(
                X_train,
                y_train,
                covariance_scale,
            )
        )

        for forgetting_factor in (
            FORGETTING_FACTOR_CANDIDATES
        ):
            (
                validation_prediction,
                _,
                _,
                _,
            ) = rls_predict(
                X=X_validation,
                y=y_validation,
                initial_beta=initial_beta,
                initial_covariance=initial_covariance,
                forgetting_factor=forgetting_factor,
                update=True,
            )

            metrics = calculate_metrics(
                y_validation,
                validation_prediction,
            )

            search_rows.append({
                "Mode": "RLS-DLR",
                "Forgetting_Factor":
                    forgetting_factor,
                "Covariance_Scale":
                    covariance_scale,
                **metrics,
            })

            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_mode = "RLS-DLR"
                best_forgetting_factor = (
                    forgetting_factor
                )
                best_covariance_scale = (
                    covariance_scale
                )

    search_results = pd.DataFrame(
        search_rows
    ).sort_values(
        by=["RMSE", "MAE"],
        ascending=True,
    ).reset_index(drop=True)

    return (
        best_mode,
        float(best_forgetting_factor),
        float(best_covariance_scale),
        search_results,
    )


def feature_contribution_table(
    X: np.ndarray,
    coefficient_history: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    feature_coefficients = (
        coefficient_history[:, 1:]
    )

    contribution = np.abs(
        feature_coefficients * X
    )

    table = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_Contribution":
            np.mean(contribution, axis=0),
        "Mean_Absolute_Coefficient":
            np.mean(
                np.abs(feature_coefficients),
                axis=0,
            ),
        "Coefficient_Standard_Deviation":
            np.std(
                feature_coefficients,
                axis=0,
            ),
    })

    return table.sort_values(
        "Mean_Absolute_Contribution",
        ascending=False,
    ).reset_index(drop=True)


def process_city(
    city: str,
    city_directory: Path,
) -> List[dict]:
    print("\n" + "=" * 74)
    print(f"Processing city: {city}")
    print("=" * 74)

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

    city_output_directory = (
        OUTPUT_ROOT / city
    )

    city_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        selected_mode,
        selected_forgetting_factor,
        selected_covariance_scale,
        parameter_search,
    ) = select_rls_parameters(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    print(f"Selected mode: {selected_mode}")
    print(
        "Selected forgetting factor: "
        f"{selected_forgetting_factor}"
    )
    print(
        "Selected covariance scale: "
        f"{selected_covariance_scale}"
    )

    parameter_search.to_csv(
        city_output_directory
        / "rls_parameter_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )


    actual_covariance_scale = (
        selected_covariance_scale
        if selected_mode == "RLS-DLR"
        else 1.0
    )

    initial_beta, initial_covariance = (
        initialize_rls(
            X_train,
            y_train,
            actual_covariance_scale,
        )
    )


    if selected_mode == "RLS-DLR":
        (
            validation_predictions,
            beta_after_validation,
            covariance_after_validation,
            validation_beta_history,
        ) = rls_predict(
            X=X_validation,
            y=y_validation,
            initial_beta=initial_beta,
            initial_covariance=initial_covariance,
            forgetting_factor=(
                selected_forgetting_factor
            ),
            update=True,
        )

    else:
        validation_predictions = fixed_predict(
            X_validation,
            initial_beta,
        )

        beta_after_validation = (
            initial_beta.copy()
        )

        covariance_after_validation = (
            initial_covariance.copy()
        )

        validation_beta_history = np.tile(
            initial_beta,
            (len(X_validation), 1),
        )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )


    holdout_predictions = fixed_predict(
        X_test,
        beta_after_validation,
    )

    holdout_metrics = calculate_metrics(
        y_test,
        holdout_predictions,
    )


    if selected_mode == "RLS-DLR":
        (
            online_predictions,
            beta_after_online,
            covariance_after_online,
            online_beta_history,
        ) = rls_predict(
            X=X_test,
            y=y_test,
            initial_beta=beta_after_validation,
            initial_covariance=(
                covariance_after_validation
            ),
            forgetting_factor=(
                selected_forgetting_factor
            ),
            update=True,
        )

    else:
        online_predictions = (
            holdout_predictions.copy()
        )

        beta_after_online = (
            beta_after_validation.copy()
        )

        covariance_after_online = (
            covariance_after_validation.copy()
        )

        online_beta_history = np.tile(
            beta_after_validation,
            (len(X_test), 1),
        )

    online_metrics = calculate_metrics(
        y_test,
        online_predictions,
    )


    print("\nValidation metrics:")

    for metric, value in validation_metrics.items():
        print(f"  {metric}: {value:.6f}")

    print("\nDLR-Holdout test metrics:")

    for metric, value in holdout_metrics.items():
        print(f"  {metric}: {value:.6f}")

    print("\nDLR-Online test metrics:")

    for metric, value in online_metrics.items():
        print(f"  {metric}: {value:.6f}")


    validation_output = pd.DataFrame({
        "Target_Date": validation_dates,
        "Observed_AQI": y_validation,
        "Predicted_AQI":
            validation_predictions,
        "Error":
            y_validation
            - validation_predictions,
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
        "DLR_Holdout_Prediction":
            holdout_predictions,
        "DLR_Online_Prediction":
            online_predictions,
        "Holdout_Error":
            y_test - holdout_predictions,
        "Online_Error":
            y_test - online_predictions,
    })

    test_output.to_csv(
        city_output_directory
        / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )


    coefficient_columns = [
        "Intercept",
        *feature_names,
    ]

    validation_coefficient_table = pd.DataFrame(
        validation_beta_history,
        columns=coefficient_columns,
    )

    validation_coefficient_table.insert(
        0,
        "Target_Date",
        validation_dates,
    )

    validation_coefficient_table.to_csv(
        city_output_directory
        / "validation_coefficient_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    online_coefficient_table = pd.DataFrame(
        online_beta_history,
        columns=coefficient_columns,
    )

    online_coefficient_table.insert(
        0,
        "Target_Date",
        test_dates,
    )

    online_coefficient_table.to_csv(
        city_output_directory
        / "online_coefficient_history.csv",
        index=False,
        encoding="utf-8-sig",
    )


    contribution_table = (
        feature_contribution_table(
            X_validation,
            validation_beta_history,
            feature_names,
        )
    )

    contribution_table.to_csv(
        city_output_directory
        / "feature_contribution_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )


    model_information = {
        "City": city,
        "Model": "RLS-DLR",
        "Selected_Mode": selected_mode,
        "Number_of_Features":
            len(feature_names),
        "Historical_AQI_Feature_Count":
            len(historical_aqi_features),
        "Historical_AQI_Features":
            historical_aqi_features,
        "Ridge_Alpha":
            RIDGE_ALPHA,
        "Selected_Forgetting_Factor":
            selected_forgetting_factor,
        "Selected_Covariance_Scale":
            selected_covariance_scale,
        "Forgetting_Factor_Candidates":
            FORGETTING_FACTOR_CANDIDATES,
        "Covariance_Scale_Candidates":
            COVARIANCE_SCALE_CANDIDATES,
        "Validation_Updates":
            selected_mode == "RLS-DLR",
        "Holdout_Test_Updates": False,
        "Online_Test_Updates":
            selected_mode == "RLS-DLR",
        "Test_Used_for_Parameter_Selection":
            False,
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
            "Model": "DLR-Validation",
            "Protocol":
                "Sequential validation",
            "Selected_Mode":
                selected_mode,
            "Number_of_Features":
                len(feature_names),
            "Forgetting_Factor":
                selected_forgetting_factor,
            "Covariance_Scale":
                selected_covariance_scale,
            **validation_metrics,
        },
        {
            "City": city,
            "Model": "DLR-Holdout",
            "Protocol":
                "Independent holdout",
            "Selected_Mode":
                selected_mode,
            "Number_of_Features":
                len(feature_names),
            "Forgetting_Factor":
                selected_forgetting_factor,
            "Covariance_Scale":
                selected_covariance_scale,
            **holdout_metrics,
        },
        {
            "City": city,
            "Model": "DLR-Online",
            "Protocol":
                "Sequential online update",
            "Selected_Mode":
                selected_mode,
            "Number_of_Features":
                len(feature_names),
            "Forgetting_Factor":
                selected_forgetting_factor,
            "Covariance_Scale":
                selected_covariance_scale,
            **online_metrics,
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
            raise

    if not all_rows:
        print("\nNo DLR results were generated.")
        return

    summary = pd.DataFrame(all_rows)

    summary_file = (
        OUTPUT_ROOT
        / "DLR_all_cities_metrics.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 74)
    print("RLS-DLR summary")
    print("=" * 74)
    print(summary.to_string(index=False))

    print(
        f"\nAll-city summary saved to: "
        f"{summary_file}"
    )


if __name__ == "__main__":
    main()
