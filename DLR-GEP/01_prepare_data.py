from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "prepared_data"

CITY_FILES = {
    "Beijing": SCRIPT_DIR / "beijing_aqi_2022_2024_combined.csv",
    "Nanning": SCRIPT_DIR / "Nanning_aqi_2022_2024_combined.csv",
}

POLLUTANT_COLUMNS = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
]

AQI_COLUMN = "AQI"
DATE_COLUMN = "Date"

HISTORY_WINDOW = 7
ROLLING_WINDOWS = [3, 7]

TEST_RATIO = 0.20
VALIDATION_RATIO_WITHIN_DEVELOPMENT = 0.20


def clean_column_name(name: str) -> str:

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace("₂", "2")
        .replace("₁", "1")
        .replace("₀", "0")
    )


def read_csv_safely(file_path: Path) -> Tuple[pd.DataFrame, str]:

    encodings = [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
        "gb2312",
    ]

    last_error = None

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(
                file_path,
                encoding=encoding,
            )

            dataframe = dataframe.loc[
                :,
                ~dataframe.columns.astype(str)
                .str.lower()
                .str.startswith("unnamed")
            ]

            return dataframe, encoding

        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        f"Unable to read file: {file_path}\n"
        f"Last encoding error: {last_error}"
    )


def standardize_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:


    alias_mapping = {
        "date": "Date",
        "日期": "Date",
        "time": "Date",
        "datetime": "Date",

        "pm25": "PM2.5",
        "pm2.5": "PM2.5",
        "pm₂₅": "PM2.5",

        "pm10": "PM10",
        "pm₁₀": "PM10",

        "so2": "SO2",
        "so₂": "SO2",

        "no2": "NO2",
        "no₂": "NO2",

        "co": "CO",

        "o3": "O3",
        "o₃": "O3",

        "aqi": "AQI",
        "aqi指数": "AQI",
        "空气质量指数": "AQI",
    }

    rename_mapping = {}

    for column in dataframe.columns:
        original_text = str(column).strip()
        normalized = clean_column_name(original_text)

        matched_name = None

        for alias, standard_name in alias_mapping.items():
            if clean_column_name(alias) == normalized:
                matched_name = standard_name
                break

        if matched_name is not None:
            rename_mapping[column] = matched_name

    return dataframe.rename(columns=rename_mapping)


def check_required_columns(dataframe: pd.DataFrame) -> None:
    required_columns = (
        [DATE_COLUMN]
        + POLLUTANT_COLUMNS
        + [AQI_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Raw data are missing required columns: "
            f"{missing_columns}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


def remove_invalid_rows(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:


    statistics = {
        "raw_records": len(dataframe),
        "duplicates_removed": 0,
        "invalid_dates_removed": 0,
        "nonnumeric_rows_removed": 0,
        "negative_rows_removed": 0,
    }

    dataframe = dataframe.copy()


    dataframe[DATE_COLUMN] = pd.to_datetime(
        dataframe[DATE_COLUMN],
        errors="coerce",
    )

    invalid_date_count = int(
        dataframe[DATE_COLUMN].isna().sum()
    )

    statistics["invalid_dates_removed"] = (
        invalid_date_count
    )

    dataframe = dataframe.dropna(
        subset=[DATE_COLUMN]
    )


    numeric_columns = (
        POLLUTANT_COLUMNS + [AQI_COLUMN]
    )

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    nonnumeric_mask = dataframe[
        numeric_columns
    ].isna().any(axis=1)

    statistics["nonnumeric_rows_removed"] = int(
        nonnumeric_mask.sum()
    )

    dataframe = dataframe.loc[
        ~nonnumeric_mask
    ].copy()


    negative_mask = (
        dataframe[numeric_columns] < 0
    ).any(axis=1)

    statistics["negative_rows_removed"] = int(
        negative_mask.sum()
    )

    dataframe = dataframe.loc[
        ~negative_mask
    ].copy()


    dataframe = dataframe.sort_values(
        DATE_COLUMN
    ).reset_index(drop=True)

    before_duplicates = len(dataframe)

    dataframe = dataframe.drop_duplicates(
        subset=[DATE_COLUMN],
        keep="first",
    ).reset_index(drop=True)

    statistics["duplicates_removed"] = (
        before_duplicates - len(dataframe)
    )

    return dataframe, statistics


def construct_features(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:


    feature_dataframe = dataframe[
        [DATE_COLUMN]
        + POLLUTANT_COLUMNS
        + [AQI_COLUMN]
    ].copy()

    feature_names = []


    for pollutant in POLLUTANT_COLUMNS:
        for lag in range(HISTORY_WINDOW):
            feature_name = f"{pollutant}_lag{lag}"

            feature_dataframe[feature_name] = (
                feature_dataframe[pollutant].shift(lag)
            )

            feature_names.append(feature_name)


    for pollutant in POLLUTANT_COLUMNS:
        for window in ROLLING_WINDOWS:
            mean_name = (
                f"{pollutant}_roll{window}_mean"
            )

            std_name = (
                f"{pollutant}_roll{window}_std"
            )

            feature_dataframe[mean_name] = (
                feature_dataframe[pollutant]
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .mean()
            )

            feature_dataframe[std_name] = (
                feature_dataframe[pollutant]
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .std(ddof=0)
            )

            feature_names.extend([
                mean_name,
                std_name,
            ])


    for pollutant in POLLUTANT_COLUMNS:
        difference_name = f"{pollutant}_diff1"

        feature_dataframe[difference_name] = (
            feature_dataframe[pollutant].diff(1)
        )

        feature_names.append(difference_name)


    for lag in range(HISTORY_WINDOW):
        feature_name = f"AQI_lag{lag}"

        feature_dataframe[feature_name] = (
            feature_dataframe[AQI_COLUMN].shift(lag)
        )

        feature_names.append(feature_name)


    for window in ROLLING_WINDOWS:
        mean_name = f"AQI_roll{window}_mean"
        std_name = f"AQI_roll{window}_std"

        feature_dataframe[mean_name] = (
            feature_dataframe[AQI_COLUMN]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

        feature_dataframe[std_name] = (
            feature_dataframe[AQI_COLUMN]
            .rolling(
                window=window,
                min_periods=window,
            )
            .std(ddof=0)
        )

        feature_names.extend([
            mean_name,
            std_name,
        ])


    feature_dataframe["AQI_diff1"] = (
        feature_dataframe[AQI_COLUMN].diff(1)
    )

    feature_names.append("AQI_diff1")


    feature_dataframe["Target_Date"] = (
        feature_dataframe[DATE_COLUMN].shift(-1)
    )

    feature_dataframe["Target_AQI"] = (
        feature_dataframe[AQI_COLUMN].shift(-1)
    )

    return feature_dataframe, feature_names


def retain_consecutive_samples(
    feature_dataframe: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:


    dataframe = feature_dataframe.copy()


    input_date = dataframe[DATE_COLUMN]


    earliest_date = dataframe[
        DATE_COLUMN
    ].shift(HISTORY_WINDOW - 1)

    target_date = dataframe["Target_Date"]


    historical_window_is_consecutive = (
        input_date - earliest_date
        == pd.Timedelta(
            days=HISTORY_WINDOW - 1
        )
    )


    target_is_next_day = (
        target_date - input_date
        == pd.Timedelta(days=1)
    )

    complete_values = dataframe[
        feature_names + ["Target_AQI"]
    ].notna().all(axis=1)

    valid_mask = (
        historical_window_is_consecutive
        & target_is_next_day
        & complete_values
    )

    result = dataframe.loc[
        valid_mask,
        [
            DATE_COLUMN,
            "Target_Date",
            *feature_names,
            "Target_AQI",
        ],
    ].copy()

    result = result.rename(
        columns={DATE_COLUMN: "Input_Date"}
    )

    result = result.sort_values(
        "Target_Date"
    ).reset_index(drop=True)

    return result


def chronological_split(
    samples: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:


    number_of_samples = len(samples)

    number_of_test = int(
        np.ceil(
            number_of_samples * TEST_RATIO
        )
    )

    number_of_development = (
        number_of_samples - number_of_test
    )

    number_of_validation = int(
        round(
            number_of_development
            * VALIDATION_RATIO_WITHIN_DEVELOPMENT
        )
    )

    number_of_training = (
        number_of_development
        - number_of_validation
    )

    training = samples.iloc[
        :number_of_training
    ].copy()

    validation = samples.iloc[
        number_of_training:
        number_of_training + number_of_validation
    ].copy()

    testing = samples.iloc[
        number_of_training + number_of_validation:
    ].copy()

    return training, validation, testing


def dataframe_from_scaled_features(
    original_split: pd.DataFrame,
    scaled_features: np.ndarray,
    feature_names: List[str],
    scaled_target: np.ndarray,
) -> pd.DataFrame:


    output = pd.DataFrame(
        scaled_features,
        columns=feature_names,
    )

    output.insert(
        0,
        "Target_Date",
        original_split["Target_Date"].dt.strftime(
            "%Y-%m-%d"
        ).to_numpy(),
    )

    output.insert(
        0,
        "Input_Date",
        original_split["Input_Date"].dt.strftime(
            "%Y-%m-%d"
        ).to_numpy(),
    )

    output["Target_AQI"] = (
        original_split["Target_AQI"]
        .to_numpy(dtype=float)
    )

    output["Target_AQI_scaled"] = (
        scaled_target.reshape(-1)
    )

    return output


def save_split_files(
    city_output_dir: Path,
    split_name: str,
    raw_split: pd.DataFrame,
    feature_names: List[str],
    scaled_features: np.ndarray,
    scaled_target: np.ndarray,
) -> None:


    lower_split = split_name.lower()

    X_raw = raw_split[
        feature_names
    ].to_numpy(dtype=float)

    y_raw = raw_split[
        "Target_AQI"
    ].to_numpy(dtype=float)


    pd.DataFrame(
        scaled_features,
        columns=feature_names,
    ).to_csv(
        city_output_dir / f"X_{lower_split}.csv",
        index=False,
        encoding="utf-8-sig",
    )


    pd.DataFrame({
        "Target_AQI": y_raw,
    }).to_csv(
        city_output_dir / f"y_{lower_split}.csv",
        index=False,
        encoding="utf-8-sig",
    )


    pd.DataFrame(
        X_raw,
        columns=feature_names,
    ).to_csv(
        city_output_dir
        / f"X_{lower_split}_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )


    pd.DataFrame(
        scaled_features,
        columns=feature_names,
    ).to_csv(
        city_output_dir
        / f"X_{lower_split}_scaled.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({
        "Target_AQI_scaled":
            scaled_target.reshape(-1),
    }).to_csv(
        city_output_dir
        / f"y_{lower_split}_scaled.csv",
        index=False,
        encoding="utf-8-sig",
    )


    combined_scaled = dataframe_from_scaled_features(
        raw_split,
        scaled_features,
        feature_names,
        scaled_target,
    )

    combined_scaled.to_csv(
        city_output_dir / f"{lower_split}.csv",
        index=False,
        encoding="utf-8-sig",
    )


    raw_output = raw_split[
        [
            "Input_Date",
            "Target_Date",
            *feature_names,
            "Target_AQI",
        ]
    ].copy()

    raw_output["Input_Date"] = (
        raw_output["Input_Date"]
        .dt.strftime("%Y-%m-%d")
    )

    raw_output["Target_Date"] = (
        raw_output["Target_Date"]
        .dt.strftime("%Y-%m-%d")
    )

    raw_output.to_csv(
        city_output_dir
        / f"{lower_split}_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_scaler_parameters(
    city_output_dir: Path,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    feature_names: List[str],
) -> None:
    scaler_information = {
        "feature_names": feature_names,
        "feature_minimum": (
            feature_scaler.data_min_.tolist()
        ),
        "feature_maximum": (
            feature_scaler.data_max_.tolist()
        ),
        "feature_scale": (
            feature_scaler.scale_.tolist()
        ),
        "target_minimum": float(
            target_scaler.data_min_[0]
        ),
        "target_maximum": float(
            target_scaler.data_max_[0]
        ),
        "target_scale": float(
            target_scaler.scale_[0]
        ),
        "fitted_using": "training set only",
    }

    with open(
        city_output_dir / "scaler_parameters.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            scaler_information,
            file,
            ensure_ascii=False,
            indent=2,
        )


def process_city(
    city: str,
    file_path: Path,
) -> None:
    print("\n" + "=" * 65)
    print(f"Processing city: {city}")
    print("=" * 65)

    dataframe, encoding = read_csv_safely(
        file_path
    )

    dataframe = standardize_column_names(
        dataframe
    )

    check_required_columns(dataframe)

    dataframe, cleaning_statistics = (
        remove_invalid_rows(dataframe)
    )

    feature_dataframe, feature_names = (
        construct_features(dataframe)
    )

    samples = retain_consecutive_samples(
        feature_dataframe,
        feature_names,
    )

    training, validation, testing = (
        chronological_split(samples)
    )

    city_output_dir = (
        OUTPUT_ROOT / city.lower()
    )

    city_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    feature_scaler = MinMaxScaler()

    X_train_raw = training[
        feature_names
    ].to_numpy(dtype=float)

    X_validation_raw = validation[
        feature_names
    ].to_numpy(dtype=float)

    X_test_raw = testing[
        feature_names
    ].to_numpy(dtype=float)

    X_train_scaled = feature_scaler.fit_transform(
        X_train_raw
    )

    X_validation_scaled = feature_scaler.transform(
        X_validation_raw
    )

    X_test_scaled = feature_scaler.transform(
        X_test_raw
    )


    target_scaler = MinMaxScaler()

    y_train_raw = training[
        ["Target_AQI"]
    ].to_numpy(dtype=float)

    y_validation_raw = validation[
        ["Target_AQI"]
    ].to_numpy(dtype=float)

    y_test_raw = testing[
        ["Target_AQI"]
    ].to_numpy(dtype=float)

    y_train_scaled = target_scaler.fit_transform(
        y_train_raw
    )

    y_validation_scaled = target_scaler.transform(
        y_validation_raw
    )

    y_test_scaled = target_scaler.transform(
        y_test_raw
    )


    save_split_files(
        city_output_dir=city_output_dir,
        split_name="train",
        raw_split=training,
        feature_names=feature_names,
        scaled_features=X_train_scaled,
        scaled_target=y_train_scaled,
    )

    save_split_files(
        city_output_dir=city_output_dir,
        split_name="validation",
        raw_split=validation,
        feature_names=feature_names,
        scaled_features=X_validation_scaled,
        scaled_target=y_validation_scaled,
    )

    save_split_files(
        city_output_dir=city_output_dir,
        split_name="test",
        raw_split=testing,
        feature_names=feature_names,
        scaled_features=X_test_scaled,
        scaled_target=y_test_scaled,
    )

    with open(
        city_output_dir / "feature_names.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_names,
            file,
            ensure_ascii=False,
            indent=2,
        )

    save_scaler_parameters(
        city_output_dir,
        feature_scaler,
        target_scaler,
        feature_names,
    )


    complete_output = samples.copy()

    complete_output["Input_Date"] = (
        complete_output["Input_Date"]
        .dt.strftime("%Y-%m-%d")
    )

    complete_output["Target_Date"] = (
        complete_output["Target_Date"]
        .dt.strftime("%Y-%m-%d")
    )

    complete_output.to_csv(
        city_output_dir / "all_valid_samples_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )


    print(f"CSV encoding: {encoding}")
    print(
        f"Raw records: "
        f"{cleaning_statistics['raw_records']}"
    )
    print(
        "Duplicate dates removed: "
        f"{cleaning_statistics['duplicates_removed']}"
    )
    print(
        "Invalid dates removed: "
        f"{cleaning_statistics['invalid_dates_removed']}"
    )
    print(
        "Nonnumeric rows removed: "
        f"{cleaning_statistics['nonnumeric_rows_removed']}"
    )
    print(
        "Negative rows removed: "
        f"{cleaning_statistics['negative_rows_removed']}"
    )
    print(
        f"Unique valid observations: {len(dataframe)}"
    )
    print(f"Historical window: {HISTORY_WINDOW} days")
    print(
        f"Number of input features: "
        f"{len(feature_names)}"
    )
    print(
        "  Pollutant historical features: 72"
    )
    print(
        "  Historical AQI features: 12"
    )
    print(
        f"Valid one-day-ahead samples: "
        f"{len(samples)}"
    )
    print(f"Training samples: {len(training)}")
    print(
        f"Validation samples: {len(validation)}"
    )
    print(f"Testing samples: {len(testing)}")

    print("Target-date ranges:")

    print(
        "  Training: "
        f"{training['Target_Date'].iloc[0].date()} "
        "-> "
        f"{training['Target_Date'].iloc[-1].date()}"
    )

    print(
        "  Validation: "
        f"{validation['Target_Date'].iloc[0].date()} "
        "-> "
        f"{validation['Target_Date'].iloc[-1].date()}"
    )

    print(
        "  Testing: "
        f"{testing['Target_Date'].iloc[0].date()} "
        "-> "
        f"{testing['Target_Date'].iloc[-1].date()}"
    )

    print(f"Output directory: {city_output_dir}")


def main():
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Output directory: {OUTPUT_ROOT}")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    available_city_count = 0

    for city, file_path in CITY_FILES.items():
        if not file_path.exists():
            print(f"\nSkipping {city}：file not found")
            print(file_path)
            continue

        process_city(
            city=city,
            file_path=file_path,
        )

        available_city_count += 1

    if available_city_count == 0:
        raise FileNotFoundError(
            "No raw CSV file was found for Beijing or Nanning.\n"
            "Place the raw CSV files in the same directory as 01_prepare_data.py."
        )

    print("\nHistorical feature construction completed for all cities.")
    print(
        "AQI_lag0 through AQI_lag6 contain only AQI observations available before the target date and "
        "exclude Target_AQI."
    )


if __name__ == "__main__":
    main()
