"""
Feature engineering for the ML training dataset.

Steps:
  1. Winsorize — cap extreme values at 1st/99th percentile globally
  2. Fill nulls — replace NaN with sector+year median (fallback: global median)
  3. Sector z-scores — robust z-score relative to sector in that year
  4. Composite scores — value / quality / growth / dividend (0–100 percentile rank within sector+year)

Why sector normalization?
    A P/L of 12 is cheap for a tech company but expensive for a utility.
    Comparing within sector corrects this structural bias.

Why robust z-score (IQR instead of std)?
    Financial indicators have fat tails. IQR-based z-scores are less
    distorted by the few extreme outliers that survive winsorization.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# All 31 raw indicator columns (from processor.py)
INDICATOR_COLS: list[str] = [
    "dy", "p_l", "peg_ratio", "p_vp", "ev_ebitda", "ev_ebit",
    "p_ebitda", "p_ebit", "vpa", "p_ativo", "lpa", "p_sr",
    "p_cap_giro", "p_ativo_circ_liq", "div_liq_pl", "div_liq_ebitda",
    "div_liq_ebit", "pl_ativos", "passivos_ativos", "m_bruta",
    "m_ebitda", "m_ebit", "m_liquida", "roe", "roa", "roic",
    "giro_ativos", "liq_corrente", "cagr_receitas_5", "cagr_lucros_5",
    "graham_formula",
]

# Columns used for sector z-score normalization (the most sector-sensitive multiples)
ZSCORE_COLS: list[str] = [
    "p_l", "p_vp", "ev_ebit", "roe", "roic", "m_liquida", "dy",
]

# Composite score definitions
# "lower is better" cols → negated before ranking
VALUE_COLS: list[str]    = ["p_l", "p_vp", "ev_ebit", "p_ebit"]
QUALITY_COLS: list[str]  = ["roe", "roic", "m_liquida", "m_ebit"]
GROWTH_COLS: list[str]   = ["cagr_receitas_5", "cagr_lucros_5"]
DIVIDEND_COLS: list[str] = ["dy"]

# Composite weights (must sum to 1.0)
WEIGHTS = {
    "value_score":    0.30,
    "quality_score":  0.35,
    "growth_score":   0.20,
    "dividend_score": 0.15,
}

MIN_SECTOR_SIZE = 3  # minimum valid rows in (sector, year) to use sector stats


# ── Step 1: Winsorization ────────────────────────────────────────────────────

def winsorize(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    Clips each column to its 1st and 99th global percentile.
    Reduces impact of data entry errors and extreme outliers without dropping rows.
    """
    cols = cols or INDICATOR_COLS
    df = df.copy()
    for col in [c for c in cols if c in df.columns]:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lo, upper=hi)
    logger.debug(f"Winsorized {len(cols)} columns")
    return df


# ── Step 2: Fill nulls ───────────────────────────────────────────────────────

def fill_nulls_with_sector_median(
    df: pd.DataFrame, cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Fills NaN in each column with the median of its (sectorname, year) group.
    Falls back to the global column median when the group has fewer than
    MIN_SECTOR_SIZE valid values.

    Uses pandas transform() for vectorized, index-preserving computation.
    """
    cols = cols or INDICATOR_COLS
    df = df.copy()
    global_medians = df[[c for c in cols if c in df.columns]].median()

    for col in [c for c in cols if c in df.columns]:
        # Sector+year median — only populated when group is large enough
        group_median = df.groupby(["sectorname", "year"])[col].transform(
            lambda x: x.median() if x.notna().sum() >= MIN_SECTOR_SIZE else np.nan
        )
        df[col] = df[col].fillna(group_median).fillna(global_medians[col])

    null_remaining = df[[c for c in cols if c in df.columns]].isna().sum().sum()
    logger.debug(f"fill_nulls: {null_remaining} NaN values remaining after fill")
    return df


# ── Step 3: Sector z-scores ──────────────────────────────────────────────────

def add_sector_zscores(
    df: pd.DataFrame, cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Adds robust z-score columns: z = (x − sector_median) / sector_IQR.

    New columns are named {col}_z (e.g., p_l_z, roe_z).
    When IQR = 0 (all values identical in the group), z-score is set to 0.
    """
    cols = cols or ZSCORE_COLS
    df = df.copy()

    for col in [c for c in cols if c in df.columns]:
        z_col = f"{col}_z"
        grp = df.groupby(["sectorname", "year"])[col]

        median = grp.transform("median")
        q75    = grp.transform(lambda x: x.quantile(0.75))
        q25    = grp.transform(lambda x: x.quantile(0.25))
        iqr    = (q75 - q25).replace(0, np.nan)   # NaN where IQR=0 → fill with 0 below

        df[z_col] = ((df[col] - median) / iqr).fillna(0.0)

    logger.debug(f"Sector z-scores added for: {cols}")
    return df


# ── Step 4: Composite scores ─────────────────────────────────────────────────

def _sector_percentile_rank(df: pd.DataFrame, raw: pd.Series) -> pd.Series:
    """
    Ranks `raw` within (sectorname, year) groups, returning percentile 0–100.

    Falls back to global ranking when group columns are absent.
    This ensures an electric utility's P/L is compared only to other utilities,
    not to fintechs — consistent with how add_sector_zscores already works.
    """
    if "sectorname" not in df.columns or "year" not in df.columns:
        return raw.rank(method="average", pct=True, na_option="keep") * 100

    tmp = df[["sectorname", "year"]].copy()
    tmp["_raw"] = raw.values
    return tmp.groupby(["sectorname", "year"])["_raw"].rank(
        method="average", pct=True, na_option="keep"
    ) * 100


def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates four factor scores (0–100) and a weighted composite_score.
    Ranking is done within (sectorname, year) groups — a P/L of 12 is cheap
    for a utility but expensive for a tech company; intra-sector ranking
    captures this structural difference (consistent with add_sector_zscores).

    value_score:    rank on valuation multiples    — lower P/L, P/VP, EV/EBIT = better
    quality_score:  rank on profitability          — higher ROE, ROIC, margins = better
    growth_score:   rank on growth metrics         — higher CAGR = better
    dividend_score: rank on dividend yield         — higher DY = better

    composite_score = weighted average (weights in WEIGHTS constant above).
    """
    df = df.copy()

    # Value: lower multiple → negate so that lower raw value → higher score
    avail_value = [c for c in VALUE_COLS if c in df.columns]
    if avail_value:
        df["value_score"] = _sector_percentile_rank(df, -df[avail_value].mean(axis=1))
    else:
        df["value_score"] = 50.0

    # Quality: higher is better
    avail_quality = [c for c in QUALITY_COLS if c in df.columns]
    if avail_quality:
        df["quality_score"] = _sector_percentile_rank(df, df[avail_quality].mean(axis=1))
    else:
        df["quality_score"] = 50.0

    # Growth: higher is better
    avail_growth = [c for c in GROWTH_COLS if c in df.columns]
    if avail_growth:
        df["growth_score"] = _sector_percentile_rank(df, df[avail_growth].mean(axis=1))
    else:
        df["growth_score"] = 50.0

    # Dividend: higher is better
    avail_div = [c for c in DIVIDEND_COLS if c in df.columns]
    if avail_div:
        df["dividend_score"] = _sector_percentile_rank(df, df[avail_div[0]])
    else:
        df["dividend_score"] = 50.0

    df["composite_score"] = sum(
        WEIGHTS[score] * df[score]
        for score in WEIGHTS
    ).round(2)

    logger.info(
        f"Composite scores — composite_score stats: "
        f"mean={df['composite_score'].mean():.1f}, "
        f"std={df['composite_score'].std():.1f}"
    )
    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all feature engineering steps in order.

    Args:
        df: Raw dataset from processor.build_dataset()

    Returns:
        Enriched DataFrame with cleaned indicators, z-scores, and composite scores.
    """
    logger.info(f"Feature engineering: {len(df)} rows, {df.shape[1]} columns (input)")

    df = winsorize(df)
    df = fill_nulls_with_sector_median(df)
    df = add_sector_zscores(df)
    df = add_composite_scores(df)

    logger.info(f"Feature engineering complete: {df.shape[1]} columns (output)")
    return df
