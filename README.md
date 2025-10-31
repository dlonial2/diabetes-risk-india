# Diabetes Risk Demo

A lightweight training script for predicting diabetes risk from an Indian lifestyle survey, plus an optional Streamlit demo built around the Early Stage Diabetes clinical symptoms dataset.

## What it does
- Cleans the survey file at `data/diabetes.csv`, coercing numeric lab results, tidying string columns, and mapping `Diabetes_Status` to 0/1.
- Trains a scikit-learn pipeline (median imputation + one-hot encoding + configurable classifier) with a stratified 80/20 split (`src/train.py --dataset india`).
- Stores the fitted pipeline and metrics in the directory you choose (defaults to `models/`).
- (Optional) The Streamlit app in `app.py` can still serve the clinical symptoms model if you train it with `--dataset india_clinic`.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=$PWD/src python3 -m src.train \
  --dataset india \
  --data-path data/diabetes.csv \
  --model random_forest \
  --output-dir models_india \
  --report-prefix diabetes_india

# optional: refresh the clinical symptoms model used by Streamlit
# PYTHONPATH=$PWD/src python3 -m src.train --dataset india_clinic --data-path data/diabetes_india_clinic.csv --output-dir models_india_clinic --report-prefix diabetes_india_clinic

# optional Streamlit demo (uses the india_clinic artefacts)
# streamlit run app.py
```

## Results
- After training, inspect `<output-dir>/<report-prefix>_metrics.json` for accuracy, precision/recall/F1, ROC-AUC, and feature importances.
- The repo includes example artefacts for the clinical model in `models_india_clinic/`, including ROC curve and confusion-matrix PNGs.
