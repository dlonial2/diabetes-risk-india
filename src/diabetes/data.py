from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

_TARGET_COLUMN = "Diabetes_Status"
_NUMERIC_COLUMNS = {
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
_STRING_NORMALISE_COLUMNS = {
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
    _TARGET_COLUMN,
}
_VALUE_NORMALISATIONS = {
    "Polycystic_Ovary_Syndrome": {"0": "No", 0: "No", "1": "Yes", 1: "Yes"},
    "Alcohol_Intake": {"" : pd.NA},
}
_TARGET_MAPPING = {"yes": 1, "no": 0}


def load_dataset(
    csv_path: str | Path | None,
    *,
    expected_columns: Iterable[str] | None = None,
    download_if_missing: bool = False,
    cache_download: bool = False,
) -> pd.DataFrame:
    """Load the Diabetes Prediction (India) dataset from a CSV file.

    ``download_if_missing`` and ``cache_download`` are kept for backwards compatibility but
    are no-ops for this dataset. An explicit CSV path is required.
    """
    if csv_path is None:
        raise ValueError("csv_path must be provided for the India dataset")

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    if expected_columns is not None:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    df = df.copy()

    for column, replacements in _VALUE_NORMALISATIONS.items():
        if column in df.columns:
            df[column] = df[column].replace(replacements)

    # Clean up categorical string columns
    for column in _STRING_NORMALISE_COLUMNS & set(df.columns):
        series = df[column]
        mask = series.notna()
        df.loc[mask, column] = series[mask].astype(str).str.strip()

    # Convert numeric columns
    for column in _NUMERIC_COLUMNS & set(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if _TARGET_COLUMN in df.columns:
        target_series = df[_TARGET_COLUMN].str.lower()
        df[_TARGET_COLUMN] = target_series.map(_TARGET_MAPPING)
        if df[_TARGET_COLUMN].isna().any():
            bad_values = (
                target_series[df[_TARGET_COLUMN].isna()].dropna().unique().tolist()
            )
            raise ValueError(
                "Target column contains unexpected values: " + ", ".join(bad_values)
            )

    return df
