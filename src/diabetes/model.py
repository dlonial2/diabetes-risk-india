from __future__ import annotations

from typing import Mapping, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


_DEFAULT_NUMERIC_FEATURES: Sequence[str] = (
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
)

_DEFAULT_CATEGORICAL_FEATURES: Sequence[str] = (
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
)


def build_pipeline(
    *,
    features: Sequence[str] | None = None,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    model_type: str = "logistic_regression",
    model_kwargs: Mapping[str, object] | None = None,
) -> tuple[Pipeline, Sequence[str]]:
    """Create an sklearn pipeline for diabetes prediction.

    Returns the pipeline together with the resolved feature order.
    """
    if features is not None:
        use_numeric = tuple(features)
        use_categorical: tuple[str, ...] = ()
    else:
        use_numeric = (
            tuple(numeric_features)
            if numeric_features is not None
            else tuple(_DEFAULT_NUMERIC_FEATURES)
        )
        use_categorical = (
            tuple(categorical_features)
            if categorical_features is not None
            else tuple(_DEFAULT_CATEGORICAL_FEATURES)
        )

    model_kwargs = dict(model_kwargs or {})

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_type == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if use_numeric:
        transformers.append(("numeric", Pipeline(steps=numeric_steps), list(use_numeric)))

    if features is not None:
        # ``features`` provided implies purely numeric (backwards compatibility).
        use_categorical = ()
    elif use_categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(use_categorical)))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_type == "logistic_regression":
        defaults = {
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 500,
            "solver": "lbfgs",
        }
        defaults.update(model_kwargs)
        classifier = LogisticRegression(**defaults)
    elif model_type == "random_forest":
        defaults = {
            "n_estimators": 200,
            "random_state": None,
            "n_jobs": -1,
        }
        defaults.update(model_kwargs)
        classifier = RandomForestClassifier(**defaults)
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    resolved_features = tuple(use_numeric) + tuple(use_categorical)
    return pipeline, resolved_features
