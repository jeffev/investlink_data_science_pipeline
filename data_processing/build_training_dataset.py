"""
Sprint 2 entry point: builds the labeled training dataset from DB data.

Pipeline:
  1. Load indicators + prices from PostgreSQL  (processor.py)
  2. Feature engineering: winsorize, fill nulls, z-scores, composite scores  (feature_engineer.py)
  3. Generate Barata/Neutra/Cara labels from annual price returns  (labeler.py)
  4. Save to parquet (fast loading for Sprint 3 model training)

Usage:
    # Full run — relative to Ibovespa, saves to data/training_dataset.parquet
    python data_processing/build_training_dataset.py

    # Use absolute returns (no Ibovespa comparison)
    python data_processing/build_training_dataset.py --no-relative

    # Custom output path
    python data_processing/build_training_dataset.py --output data/my_dataset.parquet

Run from the data_science_pipeline root directory.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging

import pandas as pd

from data_processing.processor import build_dataset, INDICATOR_COLS
from data_processing.feature_engineer import engineer_features
from data_processing.labeler import add_labels, drop_unlabeled, label_distribution_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "data/training_dataset.parquet"

# Feature columns included in the final ML-ready dataset
# (excludes metadata cols like price_current, price_next, stock_return, ibov_return)
ML_FEATURE_COLS: list[str] = (
    INDICATOR_COLS
    + [f"{c}_z" for c in ["p_l", "p_vp", "ev_ebit", "roe", "roic", "m_liquida", "dy"]]
    + ["value_score", "quality_score", "growth_score", "dividend_score", "composite_score"]
)

# Columns kept in the saved dataset (features + metadata needed for analysis)
FINAL_COLS: list[str] = (
    ["ticker", "year", "sectorname", "subsectorname", "segmentname"]
    + ML_FEATURE_COLS
    + ["price_current", "price_next", "stock_return", "ibov_return", "alpha", "label"]
)


def build_training_dataset(
    use_relative: bool = True,
    output_path: str = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """
    Full Sprint 2 pipeline.

    Args:
        use_relative:  Adjust returns by Ibovespa (recommended).
        output_path:   Path for the output parquet file.

    Returns:
        Labeled DataFrame saved to output_path.
    """
    logger.info("═══ Sprint 2: Building Training Dataset ═══")

    logger.info("─── Step 1: Load data from DB ───────────")
    df = build_dataset()

    logger.info("─── Step 2: Feature engineering ─────────")
    df = engineer_features(df)

    logger.info("─── Step 3: Generate labels ──────────────")
    df = add_labels(df, use_relative=use_relative)
    label_distribution_report(df)
    df = drop_unlabeled(df)

    logger.info("─── Step 4: Select & save columns ───────")
    # Keep only columns that exist in the DataFrame (future-proof for missing indicators)
    cols_to_save = [c for c in FINAL_COLS if c in df.columns]
    df_final = df[cols_to_save].copy()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_final.to_parquet(output_path, index=False)

    logger.info(f"═══ Dataset saved: {output_path} ═══")
    logger.info(f"    Rows:    {len(df_final)}")
    logger.info(f"    Columns: {df_final.shape[1]}")
    logger.info(f"    Tickers: {df_final['ticker'].nunique()}")
    logger.info(f"    Years:   {df_final['year'].min()} – {df_final['year'].max()}")
    logger.info(f"    Labels:  {df_final['label'].value_counts().to_dict()}")

    return df_final


def _print_summary(df: pd.DataFrame) -> None:
    """Prints a concise summary to stdout for quick inspection."""
    print("\n══ Dataset Summary ══════════════════════════════")
    print(f"  Shape:       {df.shape}")
    print(f"  Tickers:     {df['ticker'].nunique()}")
    print(f"  Years:       {df['year'].min()} – {df['year'].max()}")
    print(f"  Labels:      {df['label'].value_counts().to_dict()}")
    print(f"  Null values: {df.isna().sum().sum()} total")
    print("\n── Composite score by label ─────────────────────")
    if "composite_score" in df.columns and "label" in df.columns:
        print(df.groupby("label")["composite_score"].describe().round(1).to_string())
    print("\n── Sample rows ──────────────────────────────────")
    print(df[["ticker", "year", "sectorname", "composite_score", "label"]].head(10).to_string())
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build labeled training dataset for InvestLink ML pipeline"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-relative",
        dest="relative",
        action="store_false",
        help="Use absolute price returns instead of Ibovespa-adjusted alpha",
    )
    args = parser.parse_args()

    df = build_training_dataset(use_relative=args.relative, output_path=args.output)
    _print_summary(df)


if __name__ == "__main__":
    main()
