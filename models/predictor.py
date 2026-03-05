"""
Sprint 3 — Prediction on current data.

Loads the trained model and generates Barata/Neutra/Cara predictions
for the most recent year of indicators available in the DB.

Design decision — consistent normalization:
    Feature engineering (z-scores, composite scores) is recomputed on ALL
    historical data + the current year together. This ensures the current-year
    z-scores are computed relative to the same historical distribution the model
    was trained on, avoiding distribution shift.

Run:
    python models/predictor.py
    python models/predictor.py --dry-run   # prints predictions, does not save to DB
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

from database.connector import get_session
from database.queries import save_predictions
from data_processing.processor import build_dataset
from data_processing.feature_engineer import engineer_features
from models.trainer import MODELS_DIR

logger = logging.getLogger(__name__)


# ── Load artifacts ────────────────────────────────────────────────────────────

def load_artifacts(models_dir: str = MODELS_DIR) -> tuple:
    """
    Loads model, scaler, label_encoder and metadata from disk.

    Returns:
        (model, scaler, label_encoder, metadata_dict)
    """
    model_path = os.path.join(models_dir, "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run 'python models/trainer.py' first."
        )

    model  = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    le     = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))

    with open(os.path.join(models_dir, "metadata.json")) as f:
        meta = json.load(f)

    logger.info(
        f"Loaded model: {meta['model_version']}  "
        f"(CV F1={meta['metrics'].get('cv_f1_weighted_best', '?')})"
    )
    return model, scaler, le, meta


# ── Feature preparation for current data ─────────────────────────────────────

def prepare_current_features(
    feat_cols: list[str],
) -> pd.DataFrame:
    """
    Loads the full historical dataset from DB, applies feature engineering,
    and returns only the most recent year per ticker.

    Using the full history for z-score computation keeps the normalization
    consistent with what the model saw during training.
    """
    logger.info("Loading full dataset from DB for feature normalization context…")
    df = build_dataset()
    df = engineer_features(df)

    # Most recent year present for each ticker
    latest_by_ticker = df.groupby("ticker")["year"].max().reset_index()
    df_current = df.merge(latest_by_ticker, on=["ticker", "year"])

    logger.info(
        f"Current snapshot: {len(df_current)} tickers, "
        f"year range {df_current['year'].min()}–{df_current['year'].max()}"
    )

    # Keep only feature columns that exist
    avail_cols = [c for c in feat_cols if c in df_current.columns]
    missing = set(feat_cols) - set(avail_cols)
    if missing:
        logger.warning(f"Missing feature columns: {sorted(missing)}")

    return df_current, avail_cols


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(
    models_dir: str = MODELS_DIR,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Generates BARATA/NEUTRA/CARA predictions for the current snapshot and
    optionally saves them to stock_predictions.

    Args:
        models_dir: Directory with trained model artifacts.
        dry_run:    If True, returns predictions without saving to DB.

    Returns:
        DataFrame with columns:
            ticker, year, label, prob_barata, prob_neutra, prob_cara,
            composite_score, model_version
    """
    model, scaler, le, meta = load_artifacts(models_dir)
    feat_cols = meta["feature_cols"]
    version   = meta["model_version"]
    label_classes = meta["label_classes"]   # e.g. ['BARATA', 'NEUTRA', 'CARA']

    df_current, avail_cols = prepare_current_features(feat_cols)

    # Fill any remaining NaN with column median (same strategy as training)
    X_raw = df_current[avail_cols].copy()
    X_raw = X_raw.fillna(X_raw.median())
    X_scaled = scaler.transform(X_raw.values)

    y_pred  = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)

    labels = le.inverse_transform(y_pred)

    # Map class name → probability column index
    class_idx = {cls: i for i, cls in enumerate(label_classes)}

    results = []
    for i, (_, row) in enumerate(df_current.iterrows()):
        results.append(
            {
                "ticker":          row["ticker"],
                "year":            int(row["year"]),
                "label":           labels[i],
                "prob_barata":     float(y_proba[i][class_idx.get("BARATA", 0)]),
                "prob_neutra":     float(y_proba[i][class_idx.get("NEUTRA", 1)]),
                "prob_cara":       float(y_proba[i][class_idx.get("CARA",   2)]),
                "composite_score": float(row["composite_score"]) if "composite_score" in row else None,
                "model_version":   version,
            }
        )

    df_predictions = pd.DataFrame(results)

    logger.info(
        f"Predictions generated: {len(df_predictions)} tickers  "
        f"| {df_predictions['label'].value_counts().to_dict()}"
    )

    if dry_run:
        logger.info("Dry run — predictions NOT saved to DB")
        return df_predictions

    # Save to stock_predictions (with run_date = NOW() from DB server default)
    db_records = [
        {
            "ticker":          r["ticker"],
            "label":           r["label"],
            "prob_barata":     r["prob_barata"],
            "prob_neutra":     r["prob_neutra"],
            "prob_cara":       r["prob_cara"],
            "composite_score": r["composite_score"],
            "model_version":   r["model_version"],
        }
        for r in results
    ]

    session = get_session()
    n_saved = save_predictions(session, db_records)
    session.close()

    logger.info(f"Saved {n_saved} prediction records to stock_predictions")
    return df_predictions


def print_top_predictions(df: pd.DataFrame, top_n: int = 15) -> None:
    """Prints top BARATA and top CARA candidates sorted by probability."""
    print(f"\n{'═'*60}")
    print(f"  TOP {top_n} BARATAS (maior prob_barata + composite_score)")
    print(f"{'═'*60}")
    top_barata = (
        df[df["label"] == "BARATA"]
        .sort_values(["prob_barata", "composite_score"], ascending=False)
        .head(top_n)[["ticker", "year", "prob_barata", "prob_neutra", "composite_score"]]
    )
    print(top_barata.round(3).to_string(index=False))

    print(f"\n{'═'*60}")
    print(f"  TOP {top_n} CARAS (maior prob_cara)")
    print(f"{'═'*60}")
    top_cara = (
        df[df["label"] == "CARA"]
        .sort_values("prob_cara", ascending=False)
        .head(top_n)[["ticker", "year", "prob_cara", "prob_barata", "composite_score"]]
    )
    print(top_cara.round(3).to_string(index=False))

    print(f"\n{'─'*60}")
    print(f"  Summary: {df['label'].value_counts().to_dict()}")
    print(f"  Model:   {df['model_version'].iloc[0]}")


def main():
    parser = argparse.ArgumentParser(description="Generate stock predictions and save to DB")
    parser.add_argument("--models-dir", default=MODELS_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print predictions but do not save to DB",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = predict(models_dir=args.models_dir, dry_run=args.dry_run)
    print_top_predictions(df)


if __name__ == "__main__":
    main()
