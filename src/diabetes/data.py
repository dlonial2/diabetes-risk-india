from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

_INDIA_TARGET_COLUMN = "Diabetes_Status"
_INDIA_NUMERIC_COLUMNS = {
    "Age",
    "BMI",
    "Cholesterol_Level",
    "Fasting_Blood_Sugar",
    "Postprandial_Blood_Sugar",
    "HBA1C",
    "Heart_Rate",
    "Waist_Hip_Ratio",
    "Glucose_Tolerance_Test_Result",
    "Vitamin_D_Level",
    "C_Protein_Level",
    "Pregnancies",
}
_INDIA_STRING_NORMALISE_COLUMNS = {
    "Gender",
    "Family_History",
    "Physical_Activity",
    "Diet_Type",
    "Smoking_Status",
    "Alcohol_Intake",
    "Stress_Level",
    "Hypertension",
    "Urban_Rural",
    "Health_Insurance",
    "Regular_Checkups",
    "Medication_For_Chronic_Conditions",
    "Polycystic_Ovary_Syndrome",
    "Thyroid_Condition",
    _INDIA_TARGET_COLUMN,
}
_INDIA_VALUE_NORMALISATIONS = {
    "Polycystic_Ovary_Syndrome": {"0": "No", 0: "No", "1": "Yes", 1: "Yes"},
    "Alcohol_Intake": {"": pd.NA},
}
_INDIA_TARGET_MAPPING = {"yes": 1, "no": 0}

_INDIA_CLINIC_TARGET_COLUMN = "class"


def load_dataset(
    csv_path: str | Path | None,
    *,
    dataset: str = "india",
    expected_columns: Iterable[str] | None = None,
    download_if_missing: bool = False,
    cache_download: bool = False,
) -> pd.DataFrame:
    """Load a diabetes dataset from ``csv_path``.

    ``dataset`` selects the schema-specific cleaning rules. ``download_if_missing`` and
    ``cache_download`` are retained for backwards compatibility and remain no-ops.
    """

    if csv_path is None:
        raise ValueError("csv_path must be provided")

    dataset = dataset.lower()

    if dataset == "india":
        return _load_india_dataset(csv_path, expected_columns)
    if dataset == "india_clinic":
        return _load_india_clinic_dataset(csv_path, expected_columns)

    raise ValueError(f"Unsupported dataset '{dataset}'")


def _load_india_dataset(
    csv_path: str | Path,
    expected_columns: Iterable[str] | None,
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    if expected_columns is not None:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    df = df.copy()

    for column, replacements in _INDIA_VALUE_NORMALISATIONS.items():
        if column in df.columns:
            df[column] = df[column].replace(replacements)

    for column in _INDIA_STRING_NORMALISE_COLUMNS & set(df.columns):
        series = df[column]
        mask = series.notna()
        df.loc[mask, column] = series[mask].astype(str).str.strip()

    for column in _INDIA_NUMERIC_COLUMNS & set(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if _INDIA_TARGET_COLUMN in df.columns:
        target_series = df[_INDIA_TARGET_COLUMN].str.lower()
        df[_INDIA_TARGET_COLUMN] = target_series.map(_INDIA_TARGET_MAPPING)
        if df[_INDIA_TARGET_COLUMN].isna().any():
            bad_values = (
                target_series[df[_INDIA_TARGET_COLUMN].isna()].dropna().unique().tolist()
            )
            raise ValueError(
                "Target column contains unexpected values: " + ", ".join(bad_values)
            )

    return df


def _load_india_clinic_dataset(
    csv_path: str | Path,
    expected_columns: Iterable[str] | None,
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    df = df.rename(
        columns=lambda col: col.strip().lower().replace(" ", "_") if isinstance(col, str) else col
    )

    if expected_columns is not None:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    df = df.copy()

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    categorical_columns = [
        column
        for column in df.columns
        if column != _INDIA_CLINIC_TARGET_COLUMN and df[column].dtype == object
    ]
    for column in categorical_columns:
        df[column] = df[column].astype(str).str.strip().str.title()

    if _INDIA_CLINIC_TARGET_COLUMN in df.columns:
        target_series = df[_INDIA_CLINIC_TARGET_COLUMN].astype(str).str.strip().str.title()
        mapping = {"Positive": 1, "Negative": 0}
        df[_INDIA_CLINIC_TARGET_COLUMN] = target_series.map(mapping)
        if df[_INDIA_CLINIC_TARGET_COLUMN].isna().any():
            bad_values = (
                target_series[df[_INDIA_CLINIC_TARGET_COLUMN].isna()].dropna().unique().tolist()
            )
            raise ValueError(
                "Target column contains unexpected values: " + ", ".join(bad_values)
            )

    return df
