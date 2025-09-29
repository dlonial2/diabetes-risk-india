"""Streamlit demo for the diabetes risk model (India dataset)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/diabetes_india_pipeline.joblib")

NUMERIC_FIELDS = [
    ("Age", 45.0),
    ("BMI", 27.5),
    ("Cholesterol_Level", 190.0),
    ("Fasting_Blood_Sugar", 110.0),
    ("Postprandial_Blood_Sugar", 155.0),
    ("HBA1C", 6.2),
    ("Heart_Rate", 80.0),
    ("Waist_Hip_Ratio", 0.9),
    ("Glucose_Tolerance_Test_Result", 120.0),
    ("Vitamin_D_Level", 25.0),
    ("C_Protein_Level", 5.0),
    ("Pregnancies", 0.0),
]

CATEGORICAL_FIELDS = [
    ("Gender", ["Female", "Male", "Other"], "Female"),
    ("Family_History", ["No", "Yes"], "Yes"),
    ("Physical_Activity", ["Low", "Medium", "High"], "Medium"),
    ("Diet_Type", ["Vegetarian", "Non-Vegetarian", "Vegan"], "Vegetarian"),
    ("Smoking_Status", ["Never", "Former", "Current"], "Never"),
    ("Alcohol_Intake", ["None", "Moderate", "High"], "None"),
    ("Stress_Level", ["Low", "Medium", "High"], "Medium"),
    ("Hypertension", ["No", "Yes"], "No"),
    ("Urban_Rural", ["Urban", "Rural"], "Urban"),
    ("Health_Insurance", ["No", "Yes"], "No"),
    ("Regular_Checkups", ["No", "Yes"], "Yes"),
    ("Medication_For_Chronic_Conditions", ["No", "Yes"], "No"),
    ("Polycystic_Ovary_Syndrome", ["No", "Yes"], "No"),
    ("Thyroid_Condition", ["No", "Yes"], "No"),
]


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        st.error(
            "Model artefact not found at %s. Run the training script first to produce it." % path
        )
        return None
    return joblib.load(path)


def main() -> None:
    st.title("Diabetes Risk (India Dataset)")
    st.caption(
        "Predict the probability of diabetes using the model trained on the India diabetes dataset."
    )

    model = load_model(MODEL_PATH)
    if model is None:
        st.stop()

    threshold = st.sidebar.slider(
        "Decision threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05
    )
    st.sidebar.write(
        "Predictions above this probability are flagged as at-risk (label = 1)."
    )

    numeric_cols = st.columns(4)
    numeric_inputs: dict[str, float] = {}
    for idx, (feature, default) in enumerate(NUMERIC_FIELDS):
        column = numeric_cols[idx % len(numeric_cols)]
        with column:
            value = st.number_input(
                feature,
                value=float(default),
                step=0.5 if feature not in {"Waist_Hip_Ratio", "HBA1C"} else 0.1,
                min_value=0.0,
            )
            numeric_inputs[feature] = float(value)

    categorical_cols = st.columns(3)
    categorical_inputs: dict[str, str] = {}
    for idx, (feature, options, default) in enumerate(CATEGORICAL_FIELDS):
        column = categorical_cols[idx % len(categorical_cols)]
        with column:
            selection = st.selectbox(feature, options, index=options.index(default))
            categorical_inputs[feature] = selection

    if st.button("Predict", type="primary"):
        data = {**numeric_inputs, **categorical_inputs}
        df = pd.DataFrame([data])
        prob = float(model.predict_proba(df)[0, 1])
        label = int(prob >= threshold)
        st.metric("Risk Probability", f"{prob:.2%}")
        st.metric("Predicted Label", label)
        st.progress(prob)
        st.caption(
            "Outputs are for educational purposes only and must not be used as medical advice."
        )


if __name__ == "__main__":
    main()
