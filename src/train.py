from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from diabetes import build_pipeline, load_dataset

DATASET_CONFIGS = {
    "india": {
        "target": "Diabetes_Status",
        "numeric_features": (
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
        ),
        "categorical_features": (
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
        ),
    },
    "india_clinic": {
        "target": "class",
        "numeric_features": ("age",),
        "categorical_features": (
            "gender",
            "polyuria",
            "polydipsia",
            "sudden_weight_loss",
            "weakness",
            "polyphagia",
            "genital_thrush",
            "visual_blurring",
            "itching",
            "irritability",
            "delayed_healing",
            "partial_paresis",
            "muscle_stiffness",
            "alopecia",
            "obesity",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a diabetes prediction model.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DATASET_CONFIGS.keys()),
        default="india",
        help="Dataset schema to use for training.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/diabetes.csv"),
        help="Path to the diabetes dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory where the trained model and metrics will be stored.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/validation split.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest"],
        help="Type of classifier to train.",
    )
    parser.add_argument(
        "--penalty",
        type=str,
        default="l2",
        choices=["l2", "none"],
        help="Logistic regression penalty (limited to solvers supported by lbfgs).",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularisation strength used by logistic regression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum number of iterations for the solver.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=200,
        help="Number of trees for the random forest classifier.",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Maximum depth for trees in the random forest.",
    )
    parser.add_argument(
        "--rf-min-samples-split",
        type=int,
        default=2,
        help="Minimum number of samples required to split an internal node in the random forest.",
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=1,
        help="Minimum number of samples required to be at a leaf node in the random forest.",
    )
    parser.add_argument(
        "--rf-class-weight",
        type=str,
        default=None,
        help="Class weight strategy for the random forest (e.g. 'balanced').",
    )
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="diabetes",
        help="Filename prefix used when saving the trained artefacts and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_config = DATASET_CONFIGS[args.dataset]
    target_column: str = dataset_config["target"]
    configured_numeric: Sequence[str] = dataset_config["numeric_features"]
    configured_categorical: Sequence[str] = dataset_config["categorical_features"]

    expected_columns = set(configured_numeric) | set(configured_categorical) | {
        target_column
    }
    df = load_dataset(
        args.data_path,
        dataset=args.dataset,
        expected_columns=expected_columns,
        download_if_missing=False,
        cache_download=False,
    )

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing from dataset")

    numeric_features = tuple(col for col in configured_numeric if col in df.columns)
    categorical_features = tuple(
        col for col in configured_categorical if col in df.columns
    )

    feature_columns = numeric_features + categorical_features

    X = df.loc[:, feature_columns]
    y = df.loc[:, target_column]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    if args.model == "logistic":
        model_type = "logistic_regression"
        penalty = None if args.penalty == "none" else args.penalty
        model_kwargs = {
            "penalty": penalty,
            "C": args.c,
            "max_iter": args.max_iter,
            "solver": "lbfgs",
        }
    else:
        model_type = "random_forest"
        model_kwargs = {
            "n_estimators": args.rf_n_estimators,
            "max_depth": args.rf_max_depth,
            "min_samples_split": args.rf_min_samples_split,
            "min_samples_leaf": args.rf_min_samples_leaf,
            "class_weight": args.rf_class_weight,
            "random_state": args.random_state,
            "n_jobs": -1,
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    pipeline, resolved_features = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_type=model_type,
        model_kwargs=model_kwargs,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    y_proba = pipeline.predict_proba(X_valid)[:, 1]

    accuracy = accuracy_score(y_valid, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_valid, y_pred, pos_label=1, average="binary"
    )
    roc_auc = roc_auc_score(y_valid, y_proba)
    cm = confusion_matrix(y_valid, y_pred)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    try:
        feature_names_out = list(preprocessor.get_feature_names_out())
    except AttributeError:
        feature_names_out = list(resolved_features)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "features": list(resolved_features),
        "dataset": args.dataset,
        "model_type": args.model,
        "random_state": args.random_state,
    }

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        names = feature_names_out
        if len(names) != len(importances):
            names = [f"feature_{idx}" for idx in range(len(importances))]
        metrics["feature_importances"] = [
            {"feature": str(feature), "importance": float(importance)}
            for feature, importance in zip(names, importances)
        ]
    elif hasattr(classifier, "coef_"):
        coef = classifier.coef_
        if coef.ndim == 2 and coef.shape[1] == len(feature_names_out):
            metrics["coefficients"] = [
                {"feature": str(feature), "coefficient": float(weight)}
                for feature, weight in zip(feature_names_out, coef[0])
            ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / f"{args.report_prefix}_pipeline.joblib"
    metrics_path = args.output_dir / f"{args.report_prefix}_metrics.json"
    roc_path = args.output_dir / f"{args.report_prefix}_roc_curve.png"
    cm_path = args.output_dir / f"{args.report_prefix}_confusion_matrix.png"

    joblib.dump(pipeline, model_path)
    metrics.update(
        {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "roc_curve_path": str(roc_path),
            "confusion_matrix_path": str(cm_path),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))

    _save_roc_curve(y_valid, y_proba, roc_auc, roc_path)
    _save_confusion_matrix(cm, cm_path)

    print("Validation metrics:")
    print(f"  Accuracy        : {accuracy:.3f}")
    print(f"  Precision       : {precision:.3f}")
    print(f"  Recall          : {recall:.3f}")
    print(f"  F1-score        : {f1:.3f}")
    print(f"  ROC-AUC         : {roc_auc:.3f}")
    print("  Confusion matrix:")
    for row in cm:
        print(f"    {row[0]:3d} {row[1]:3d}")
    print(f"Model saved to   : {model_path}")
    print(f"Metrics saved to : {metrics_path}")
    print(f"ROC curve saved  : {roc_path}")
    print(f"Confusion matrix : {cm_path}")


def _save_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, roc_auc: float, path: Path) -> None:
    """Persist a ROC curve plot to ``path``."""
    plt.switch_backend("Agg")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    """Persist a normalised confusion matrix heatmap to ``path``."""
    plt.switch_backend("Agg")
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Percentage"},
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix (Normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
