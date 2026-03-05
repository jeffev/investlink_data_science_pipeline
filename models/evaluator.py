"""
Sprint 3 — Model evaluation: cross-validation, confusion matrix, feature importance.

Can be run standalone after training to generate a full evaluation report:
    python models/evaluator.py
    python models/evaluator.py --dataset data/training_dataset.parquet --cv 10
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import argparse

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from models.trainer import DATASET_PATH, MODELS_DIR, FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Core evaluation ───────────────────────────────────────────────────────────

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    cv_folds: int = 5,
) -> dict:
    """
    Runs stratified k-fold cross-validation using cross_val_predict.
    This gives out-of-fold predictions that act as an unbiased estimate
    of real-world performance (unlike fitting + predicting on the same data).

    Returns dict with all evaluation metrics.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    sample_weights = compute_sample_weight("balanced", y)

    # cross_val_predict generates OOF predictions
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")

    f1_weighted = f1_score(y, y_pred, average="weighted")
    f1_macro    = f1_score(y, y_pred, average="macro")
    f1_per_class = f1_score(y, y_pred, average=None, labels=range(len(label_names)))

    try:
        roc_auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = None

    report = classification_report(y, y_pred, target_names=label_names)
    cm      = confusion_matrix(y, y_pred)

    results = {
        "cv_folds":        cv_folds,
        "f1_weighted":     round(f1_weighted, 4),
        "f1_macro":        round(f1_macro, 4),
        "f1_per_class":    {label_names[i]: round(float(f1_per_class[i]), 4) for i in range(len(label_names))},
        "roc_auc_weighted": round(roc_auc, 4) if roc_auc else None,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y)),
        "class_distribution": {label_names[i]: int((y == i).sum()) for i in range(len(label_names))},
    }

    logger.info(
        f"\n{'─'*50}\n"
        f"Cross-validation results ({cv_folds}-fold OOF)\n"
        f"  F1 weighted : {f1_weighted:.4f}\n"
        f"  F1 macro    : {f1_macro:.4f}\n"
        f"  ROC-AUC OVR : {roc_auc:.4f if roc_auc else 'N/A'}\n"
        f"  Per class   : {results['f1_per_class']}\n"
        f"{'─'*50}"
    )
    logger.info(f"\n{report}")

    return results, y_pred, y_proba


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    output_path: str | None = None,
) -> None:
    """Plots a normalized confusion matrix heatmap."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=".0f" if data is cm else ".2f",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info(f"Confusion matrix saved: {output_path}")
    plt.show()


def plot_feature_importance(
    model,
    feature_cols: list[str],
    top_n: int = 20,
    output_path: str | None = None,
) -> None:
    """Plots top-N feature importances. Works for sklearn and XGBoost models (and Pipelines)."""
    try:
        # Unwrap Pipeline to access the classifier step's importances
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        importances = clf.feature_importances_
    except AttributeError:
        logger.warning("Model does not expose feature_importances_, skipping plot")
        return

    feat_imp = (
        pd.Series(importances, index=feature_cols)
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    feat_imp.plot(kind="bar", ax=ax, color="steelblue", alpha=0.8)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13)
    ax.set_ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info(f"Feature importance plot saved: {output_path}")
    plt.show()

    logger.info(f"\nTop 10 features:\n{feat_imp.head(10).round(4).to_string()}")


def plot_probability_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
    output_path: str | None = None,
) -> None:
    """
    Plots probability distributions per predicted class.
    Helps understand if the model is well-calibrated.
    """
    colors = {"BARATA": "#2ecc71", "NEUTRA": "#3498db", "CARA": "#e74c3c"}

    fig, axes = plt.subplots(1, len(label_names), figsize=(15, 4))

    for i, (ax, name) in enumerate(zip(axes, label_names)):
        correct   = y_proba[y_true == i, i]
        incorrect = y_proba[y_true != i, i]
        ax.hist(correct,   bins=20, alpha=0.6, color=colors.get(name, "steelblue"), label="Correct class")
        ax.hist(incorrect, bins=20, alpha=0.4, color="gray", label="Other classes")
        ax.set_title(f"P({name})")
        ax.set_xlabel("Predicted probability")
        ax.legend(fontsize=8)

    plt.suptitle("Predicted probability distribution by true class", y=1.02, fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show()


# ── Saved model evaluation ────────────────────────────────────────────────────

def evaluate_saved_model(
    dataset_path: str = DATASET_PATH,
    models_dir: str = MODELS_DIR,
    cv_folds: int = 5,
    save_plots: bool = True,
) -> dict:
    """
    Loads the trained model from disk and runs full cross-validated evaluation.

    Returns the evaluation metrics dict.
    """
    model_path = os.path.join(models_dir, "best_model.joblib")
    meta_path  = os.path.join(models_dir, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run trainer.py first."
        )

    # model is a Pipeline (scaler + clf) — no separate scaler needed
    model = joblib.load(model_path)
    le    = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))

    with open(meta_path) as f:
        meta = json.load(f)

    feat_cols = meta["feature_cols"]
    label_names = meta["label_classes"]

    logger.info(f"Evaluating: {meta['model_version']}  ({meta['model_type']})")

    df = pd.read_parquet(dataset_path)
    feat_cols_avail = [c for c in feat_cols if c in df.columns]
    df_ml = df[feat_cols_avail + ["label"]].dropna(subset=["label"])
    df_ml[feat_cols_avail] = df_ml[feat_cols_avail].fillna(df_ml[feat_cols_avail].median())
    X_raw = df_ml[feat_cols_avail].values
    y_enc = le.transform(df_ml["label"])

    # Pass raw X — the Pipeline's scaler step normalizes inside each CV fold
    results, y_pred, y_proba = cross_validate_model(
        model, X_raw, y_enc, label_names, cv_folds
    )

    plots_dir = os.path.join(models_dir, "plots")
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    cm = np.array(results["confusion_matrix"])
    plot_confusion_matrix(
        cm, label_names,
        output_path=os.path.join(plots_dir, "confusion_matrix.png") if save_plots else None,
    )
    plot_feature_importance(
        model, feat_cols_avail,
        output_path=os.path.join(plots_dir, "feature_importance.png") if save_plots else None,
    )
    plot_probability_calibration(
        y_enc, y_proba, label_names,
        output_path=os.path.join(plots_dir, "probability_calibration.png") if save_plots else None,
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model with cross-validation")
    parser.add_argument("--dataset",    default=DATASET_PATH)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    parser.add_argument("--cv",         type=int, default=5)
    parser.add_argument("--no-plots",   dest="plots", action="store_false")
    args = parser.parse_args()

    results = evaluate_saved_model(
        dataset_path=args.dataset,
        models_dir=args.models_dir,
        cv_folds=args.cv,
        save_plots=args.plots,
    )
    print(f"\nF1 weighted (OOF): {results['f1_weighted']}")
    print(f"F1 macro    (OOF): {results['f1_macro']}")
    print(f"ROC-AUC     (OOF): {results['roc_auc_weighted']}")


if __name__ == "__main__":
    main()
