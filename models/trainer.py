"""
Sprint 3 — Model training: GradientBoosting vs XGBoost.

Steps:
  1. Load training dataset (data/training_dataset.parquet)
  2. Prepare features (select cols, fillna, scale)
  3. Train GradientBoosting with GridSearchCV + stratified 5-fold
  4. Train XGBoost with GridSearchCV + stratified 5-fold
  5. Compare by weighted F1 — save the best model
  6. Persist: best_model.joblib, scaler.joblib, label_encoder.joblib, metadata.json

Run:
    python models/trainer.py
    python models/trainer.py --dataset data/training_dataset.parquet --cv 5
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, f1_score

import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_PATH  = "data/training_dataset.parquet"
MODELS_DIR    = "models"
LABEL_ORDER   = ["BARATA", "NEUTRA", "CARA"]   # fixed class order for label encoder

# Features used for training — matches feature_engineer.py output
FEATURE_COLS: list[str] = [
    # Raw indicators (winsorized + null-filled)
    "dy", "p_l", "peg_ratio", "p_vp", "ev_ebitda", "ev_ebit",
    "p_ebitda", "p_ebit", "vpa", "p_ativo", "lpa", "p_sr",
    "p_cap_giro", "p_ativo_circ_liq", "div_liq_pl", "div_liq_ebitda",
    "div_liq_ebit", "pl_ativos", "passivos_ativos", "m_bruta",
    "m_ebitda", "m_ebit", "m_liquida", "roe", "roa", "roic",
    "giro_ativos", "liq_corrente", "cagr_receitas_5", "cagr_lucros_5",
    "graham_formula",
    # Sector-relative z-scores
    "p_l_z", "p_vp_z", "ev_ebit_z", "roe_z", "roic_z", "m_liquida_z", "dy_z",
    # Composite factor scores
    "value_score", "quality_score", "growth_score", "dividend_score",
]

# GridSearch parameter grids
PARAM_GRID_GB: dict = {
    "n_estimators":   [100, 200],
    "max_depth":      [2, 3],
    "learning_rate":  [0.05, 0.1],
    "subsample":      [0.8, 1.0],
    "min_samples_leaf": [5, 10],
}

PARAM_GRID_XGB: dict = {
    "n_estimators":    [100, 200],
    "max_depth":       [3, 4],
    "learning_rate":   [0.05, 0.1],
    "subsample":       [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha":       [0, 0.1],
}


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_prepare(dataset_path: str) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    """
    Loads the training parquet, selects available feature columns,
    encodes labels (BARATA=0, NEUTRA=1, CARA=2).

    Returns raw (unscaled) X so the scaler can be fit exclusively on training
    folds inside each CV split, preventing leakage from test folds.

    Returns:
        X_raw         — raw feature matrix (n_samples, n_features), NaN-filled
        y             — encoded label array
        label_encoder — fitted LabelEncoder
        feature_cols  — list of feature column names actually used
    """
    df = pd.read_parquet(dataset_path)
    logger.info(f"Dataset loaded: {df.shape}, labels: {df['label'].value_counts().to_dict()}")

    # Use only columns that exist in this dataset
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(feat_cols)
    if missing:
        logger.warning(f"Missing feature columns (will be skipped): {sorted(missing)}")

    df_ml = df[feat_cols + ["label"]].dropna(subset=["label"])

    # Fill any remaining NaN in features with column median
    df_ml[feat_cols] = df_ml[feat_cols].fillna(df_ml[feat_cols].median())

    X_raw = df_ml[feat_cols].values

    le = LabelEncoder()
    # Fit in fixed order so indices are stable across runs
    le.fit(LABEL_ORDER)
    y = le.transform(df_ml["label"])

    logger.info(
        f"Features: {len(feat_cols)} cols, {X_raw.shape[0]} samples  "
        f"| Classes: {dict(zip(le.classes_, np.bincount(y)))}"
    )
    return X_raw, y, le, feat_cols


# ── Model training ────────────────────────────────────────────────────────────

def _run_grid_search(
    estimator,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    name: str,
) -> tuple:
    """
    Runs GridSearchCV with stratified k-fold and balanced sample weights.

    Wraps estimator in a Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    so the scaler is fit only on training folds — no leakage from test folds.

    Returns (best_pipeline, best_cv_f1_weighted).
    """
    sample_weights = compute_sample_weight("balanced", y)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])

    # Prefix param_grid keys so GridSearchCV routes them to the "clf" step
    prefixed_grid = {f"clf__{k}": v for k, v in param_grid.items()}

    grid = GridSearchCV(
        pipeline,
        prefixed_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    logger.info(
        f"[{name}] GridSearch: {len(param_grid)} params × {cv_folds} folds "
        f"= {cv_folds * _count_combinations(param_grid)} fits"
    )
    grid.fit(X, y, clf__sample_weight=sample_weights)

    logger.info(
        f"[{name}] Best params:  {grid.best_params_}  "
        f"| CV F1-weighted: {grid.best_score_:.4f}"
    )
    return grid.best_estimator_, grid.best_score_


def _count_combinations(param_grid: dict) -> int:
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n


def train_gradient_boosting(X: np.ndarray, y: np.ndarray, cv_folds: int):
    gb = GradientBoostingClassifier(random_state=42)
    return _run_grid_search(gb, PARAM_GRID_GB, X, y, cv_folds, "GradientBoosting")


def train_xgboost(X: np.ndarray, y: np.ndarray, cv_folds: int):
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    return _run_grid_search(xgb_clf, PARAM_GRID_XGB, X, y, cv_folds, "XGBoost")


# ── Persistence ───────────────────────────────────────────────────────────────

def save_artifacts(
    model,
    le: LabelEncoder,
    feature_cols: list[str],
    metrics: dict,
    model_name: str,
    output_dir: str,
) -> str:
    """
    Saves the full Pipeline (scaler + classifier), label encoder and metadata.

    The Pipeline includes the StandardScaler as its first step, so no separate
    scaler.joblib is needed — prediction uses model.predict(X_raw) directly.

    Returns the model version string (used in stock_predictions.model_version).
    """
    os.makedirs(output_dir, exist_ok=True)
    version = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    joblib.dump(model, os.path.join(output_dir, "best_model.joblib"))
    joblib.dump(le,    os.path.join(output_dir, "label_encoder.joblib"))

    # Extract params from the "clf" step if model is a Pipeline
    clf_step = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    metadata = {
        "model_version":  version,
        "model_type":     model_name,
        "feature_cols":   feature_cols,
        "label_classes":  le.classes_.tolist(),
        "trained_at":     datetime.utcnow().isoformat(),
        "metrics":        metrics,
        "params":         clf_step.get_params(),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Artifacts saved to {output_dir}/  (version: {version})")
    return version


# ── Main ──────────────────────────────────────────────────────────────────────

def train(dataset_path: str = DATASET_PATH, cv_folds: int = 5, output_dir: str = MODELS_DIR):
    """
    Full training pipeline: load → train GB → train XGB → compare → save best.

    Returns the model version string of the winner.
    """
    logger.info("═══ Sprint 3: Model Training ═══")

    X, y, le, feat_cols = load_and_prepare(dataset_path)

    # Train both models (each returns a fitted Pipeline with scaler inside)
    gb_model,  gb_score  = train_gradient_boosting(X, y, cv_folds)
    xgb_model, xgb_score = train_xgboost(X, y, cv_folds)

    # Pick winner by CV F1-weighted
    if gb_score >= xgb_score:
        best_model, best_score, best_name = gb_model,  gb_score,  "GradientBoosting"
        runner_up = f"XGBoost ({xgb_score:.4f})"
    else:
        best_model, best_score, best_name = xgb_model, xgb_score, "XGBoost"
        runner_up = f"GradientBoosting ({gb_score:.4f})"

    logger.info(
        f"Winner: {best_name} (CV F1={best_score:.4f})  |  Runner-up: {runner_up}"
    )

    # Final classification report on full dataset (optimistic — use only for inspection)
    # The Pipeline's "clf" step receives sample_weight via the prefixed kwarg
    sample_weights = compute_sample_weight("balanced", y)
    best_model.fit(X, y, clf__sample_weight=sample_weights)
    y_pred = best_model.predict(X)

    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
    logger.info(
        f"Train-set report (reference only):\n"
        + classification_report(y, y_pred, target_names=le.classes_)
    )

    metrics = {
        "cv_f1_weighted_gb":  round(gb_score,  4),
        "cv_f1_weighted_xgb": round(xgb_score, 4),
        "cv_f1_weighted_best": round(best_score, 4),
        "train_f1_weighted":  round(f1_score(y, y_pred, average="weighted"), 4),
        "n_samples":          int(len(y)),
        "n_features":         int(len(feat_cols)),
        "class_distribution": {k: int(v) for k, v in zip(le.classes_, np.bincount(y))},
        "classification_report": report,
    }

    version = save_artifacts(best_model, le, feat_cols, metrics, best_name, output_dir)
    return version


def main():
    parser = argparse.ArgumentParser(description="Train Barata/Neutra/Cara classifier")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--cv",      type=int, default=5, help="CV folds (default: 5)")
    parser.add_argument("--output",  default=MODELS_DIR, help="Directory to save artifacts")
    args = parser.parse_args()

    train(dataset_path=args.dataset, cv_folds=args.cv, output_dir=args.output)


if __name__ == "__main__":
    main()
