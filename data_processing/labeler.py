"""
Generates Barata / Neutra / Cara labels from actual annual price returns.

Why Ibovespa-relative?
    In a strong bull year the index might rise 30%. A stock that rose 20%
    actually underperformed the market — calling it BARATA would be misleading.
    By subtracting the Ibovespa return, we measure alpha: did this stock beat
    the market? That's what matters for investment decisions.

Label logic:
    stock_return   = (price_next − price_current) / price_current
    ibov_return    = Ibovespa mean return for (year + 1)
    alpha          = stock_return − ibov_return   [if use_relative=True]

    alpha > +GAIN_THRESHOLD  → BARATA   (beat market by >15%)
    alpha < -LOSS_THRESHOLD  → CARA     (lagged market by >15%)
    otherwise                → NEUTRA

    When use_relative=False (or Ibovespa download fails), alpha = stock_return.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

GAIN_THRESHOLD: float = 0.15   # >+15% relative → BARATA
LOSS_THRESHOLD: float = 0.15   # >-15% relative → CARA   (absolute value)
IBOV_TICKER: str = "^BVSP"


# ── Ibovespa annual returns ───────────────────────────────────────────────────

def _fetch_ibov_annual_returns(start_year: int, end_year: int) -> dict[int, float]:
    """
    Downloads Ibovespa monthly closes and returns mean-price annual returns.

    The return for year Y is: (mean_close_Y − mean_close_{Y-1}) / mean_close_{Y-1}
    Key: returns[Y] is the market return realized *during* year Y, which is what
         we compare against a stock that was priced in year Y-1 and Y.

    Returns:
        {year: return_fraction}   e.g. {2023: 0.22, 2022: -0.10}
        Empty dict on failure — caller falls back to absolute returns.
    """
    try:
        df = yf.Ticker(IBOV_TICKER).history(
            start=f"{start_year - 1}-12-01",
            end=f"{end_year + 1}-02-01",
            interval="1mo",
        )
        if df.empty:
            logger.warning("Ibovespa history empty, falling back to absolute returns")
            return {}

        df = df.reset_index()
        df["year"] = df["Date"].dt.year
        annual_mean = df.groupby("year")["Close"].mean()

        years = sorted(annual_mean.index)
        returns: dict[int, float] = {}
        for i in range(1, len(years)):
            yr, prev = years[i], years[i - 1]
            if annual_mean[prev] > 0:
                returns[yr] = float((annual_mean[yr] - annual_mean[prev]) / annual_mean[prev])

        logger.info(f"Ibovespa annual returns fetched for {len(returns)} years")
        return returns

    except Exception as exc:
        logger.warning(f"Ibovespa download failed ({exc}), falling back to absolute returns")
        return {}


# ── Labeling ─────────────────────────────────────────────────────────────────

def add_labels(
    df: pd.DataFrame,
    use_relative: bool = True,
    gain_threshold: float = GAIN_THRESHOLD,
    loss_threshold: float = LOSS_THRESHOLD,
) -> pd.DataFrame:
    """
    Adds a 'label' column (BARATA / NEUTRA / CARA) and 'stock_return' column.

    Args:
        df:              Dataset from feature_engineer.engineer_features().
                         Must contain price_current and price_next columns.
        use_relative:    If True, adjusts stock return by Ibovespa return (alpha).
        gain_threshold:  Alpha above this → BARATA (default 15%).
        loss_threshold:  Alpha below negative of this → CARA (default 15%).

    Returns:
        DataFrame with new columns:
            stock_return   — raw year-over-year return (fraction)
            ibov_return    — Ibovespa return for that transition year (0 if unavailable)
            alpha          — stock_return minus ibov_return
            label          — BARATA / NEUTRA / CARA (None when price data missing)
    """
    df = df.copy()
    df["stock_return"] = np.nan
    df["ibov_return"]  = 0.0
    df["alpha"]        = np.nan
    df["label"]        = None

    valid = df["price_current"].notna() & df["price_next"].notna() & (df["price_current"] > 0)

    if valid.sum() == 0:
        logger.warning("No rows with both current and next-year price — all labels will be None")
        return df

    # Raw stock return
    df.loc[valid, "stock_return"] = (
        (df.loc[valid, "price_next"] - df.loc[valid, "price_current"])
        / df.loc[valid, "price_current"]
    )

    # Ibovespa adjustment
    if use_relative:
        years = df.loc[valid, "year"].unique()
        # Returns for (year + 1): the year the return is realized
        ibov = _fetch_ibov_annual_returns(int(years.min()), int(years.max()) + 1)
        if ibov:
            # Map indicator year → next year's Ibovespa return
            df.loc[valid, "ibov_return"] = (df.loc[valid, "year"] + 1).map(ibov).fillna(0.0)
            logger.info(
                f"Ibovespa adjustment applied. "
                f"Mean ibov_return={df.loc[valid, 'ibov_return'].mean():.2%}"
            )

    df.loc[valid, "alpha"] = df.loc[valid, "stock_return"] - df.loc[valid, "ibov_return"]

    # Assign labels
    alpha = df.loc[valid, "alpha"]
    labels = pd.Series("NEUTRA", index=alpha.index, dtype=str)
    labels[alpha >  gain_threshold] = "BARATA"
    labels[alpha < -loss_threshold] = "CARA"
    df.loc[valid, "label"] = labels

    # Summary
    logger.info(
        f"Labels generated for {valid.sum()} rows:\n"
        f"  BARATA: {(df['label'] == 'BARATA').sum()}  "
        f"  NEUTRA: {(df['label'] == 'NEUTRA').sum()}  "
        f"  CARA:   {(df['label'] == 'CARA').sum()}  "
        f"  None:   {df['label'].isna().sum()}"
    )
    return df


def drop_unlabeled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows without a label (no next-year price available — typically the
    most recent year scraped, since we can't know next year's price yet).
    """
    before = len(df)
    df = df[df["label"].notna()].reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} unlabeled rows, {len(df)} usable for training")
    return df


def label_distribution_report(df: pd.DataFrame) -> None:
    """Logs class balance, mean return per label, and per-sector counts."""
    if "label" not in df.columns:
        logger.warning("No 'label' column found")
        return

    total = len(df)
    logger.info("── Label distribution ──────────────────")
    for lbl in ["BARATA", "NEUTRA", "CARA"]:
        subset = df[df["label"] == lbl]
        pct = len(subset) / total * 100
        mean_alpha = subset["alpha"].mean() if "alpha" in df.columns else float("nan")
        logger.info(f"  {lbl:8s} {len(subset):4d} rows ({pct:5.1f}%)  mean alpha={mean_alpha:+.1%}")

    if "sectorname" in df.columns:
        logger.info("── Counts per sector ───────────────────")
        for sector, grp in df.groupby("sectorname"):
            counts = grp["label"].value_counts().to_dict()
            logger.info(f"  {sector:<30s} {counts}")
