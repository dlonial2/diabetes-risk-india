"""Streamlit demo for the diabetes risk model (Indian clinical dataset)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
import streamlit as st

import os
from urllib.request import urlretrieve

MODEL_URL = "https://huggingface.co/dlonial/diabetesmodel/resolve/main/diabetes_india_clinic_pipeline.joblib"
MODEL_PATH = Path("diabetes_india_clinic_pipeline.joblib")

# download model at startup if not present
if not MODEL_PATH.exists():
    urlretrieve(MODEL_URL, MODEL_PATH)

METRICS_PATH = Path("diabetes_india_clinic_metrics.json")
NUMERIC_FIELDS = [
    ("age", "Age", 45.0, 10.0, 100.0, 1.0),
]

CATEGORICAL_FIELDS = [
    ("gender", "Gender", ["Female", "Male"], "Female"),
    ("polyuria", "Polyuria", ["No", "Yes"], "No"),
    ("polydipsia", "Polydipsia", ["No", "Yes"], "No"),
    ("sudden_weight_loss", "Sudden Weight Loss", ["No", "Yes"], "No"),
    ("weakness", "Weakness", ["No", "Yes"], "No"),
    ("polyphagia", "Polyphagia", ["No", "Yes"], "No"),
    ("genital_thrush", "Genital Thrush", ["No", "Yes"], "No"),
    ("visual_blurring", "Visual Blurring", ["No", "Yes"], "No"),
    ("itching", "Itching", ["No", "Yes"], "No"),
    ("irritability", "Irritability", ["No", "Yes"], "No"),
    ("delayed_healing", "Delayed Healing", ["No", "Yes"], "No"),
    ("partial_paresis", "Partial Paresis", ["No", "Yes"], "No"),
    ("muscle_stiffness", "Muscle Stiffness", ["No", "Yes"], "No"),
    ("alopecia", "Alopecia", ["No", "Yes"], "No"),
    ("obesity", "Obesity", ["No", "Yes"], "No"),
]


@st.cache_resource
def load_model(path: Path):
    try:
        if not path.exists():
            st.error("Model download failed or file missing." )
            return None
        model = joblib.load(path)
    except Exception as exc:  # pragma: no cover - Streamlit surface
        st.error(f"Failed to load model: {exc}")
        return None
    else:
        return model


@st.cache_data
def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _prepare_inputs(raw_values: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([raw_values])


def _summarise_importances(metrics: dict | None, top_n: int = 5) -> Iterable[str]:
    if not metrics:
        return []
    if "feature_importances" in metrics:
        rows = metrics["feature_importances"]
        scores = [(row["feature"], row["importance"]) for row in rows]
    elif "coefficients" in metrics:
        rows = metrics["coefficients"]
        scores = [(row["feature"], abs(row["coefficient"])) for row in rows]
    else:
        return []

    aggregated: dict[str, float] = defaultdict(float)
    for name, score in scores:
        if score is None:
            continue
        label = str(name)
        if "__" in label:
            label = label.split("__", 1)[1]
        if label.endswith("_Yes") or label.endswith("_No"):
            label = label.rsplit("_", 1)[0]
        aggregated[label] += float(score)

    filtered = [(name, value) for name, value in aggregated.items() if value is not None]
    if not filtered:
        return []
    filtered.sort(key=lambda item: item[1], reverse=True)
    total = sum(score for _, score in filtered)
    for name, score in filtered[:top_n]:
        share = (score / total) * 100 if total > 0 else 0.0
        yield f"{name.replace('_', ' ').title()}: {share:.1f}% importance"


def main() -> None:
    st.title("Diabetes Risk (Indian Clinical Cohort)")
    st.caption(
        "Predict diabetes probability using a random-forest model trained on the Early Stage Diabetes Risk dataset collected in India."
    )

    model = load_model(MODEL_PATH)
    metrics = load_metrics(METRICS_PATH)
    if model is None:
        st.stop()

    if metrics:
        drivers = list(_summarise_importances(metrics))
        if drivers:
            st.sidebar.subheader("Key risk drivers")
            for line in drivers:
                st.sidebar.write(f"- {line}")

    threshold = st.sidebar.slider(
        "Decision threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05
    )
    st.sidebar.write(
        "Predictions above this probability are flagged as at-risk (label = 1)."
    )

    with st.form("predict_form"):
        numeric_cols = st.columns(2)
        numeric_inputs: dict[str, float] = {}
        for idx, (field_key, label, default, min_value, max_value, step) in enumerate(NUMERIC_FIELDS):
            column = numeric_cols[idx % len(numeric_cols)]
            with column:
                value = st.number_input(
                    label,
                    value=float(default),
                    min_value=float(min_value),
                    max_value=float(max_value),
                    step=float(step),
                )
                numeric_inputs[field_key] = float(value)

        categorical_inputs: dict[str, str] = {}
        cat_cols = st.columns(3)
        for idx, (field_key, label, options, default) in enumerate(CATEGORICAL_FIELDS):
            column = cat_cols[idx % len(cat_cols)]
            with column:
                selection = st.selectbox(label, options, index=options.index(default))
                categorical_inputs[field_key] = selection

        submitted = st.form_submit_button("Predict")

    if submitted:
        if not hasattr(model, "predict_proba"):
            st.error("Loaded model does not support probability predictions (predict_proba missing).")
            return

        prepped = _prepare_inputs({**numeric_inputs, **categorical_inputs})
        try:
            prob = float(model.predict_proba(prepped)[0, 1])
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.exception(exc)
            return

        label = int(prob >= threshold)
        st.metric("Risk Probability", f"{prob:.2%}")
        st.metric("Predicted Label", label)
        st.progress(int(prob * 100))
        st.caption(
            "Outputs are for educational purposes only and must not be used as medical advice."
        )


if __name__ == "__main__":
    main()
