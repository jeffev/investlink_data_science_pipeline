"""
Loads historical indicators and price data from PostgreSQL into pandas DataFrames.

Responsibilities:
  - load_indicators()         → raw indicators joined with sector info
  - load_annual_prices()      → weekly prices aggregated into yearly means
  - build_dataset()           → join indicators + current price + next-year price

Run standalone to preview what's in the DB:
    python data_processing/processor.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import pandas as pd
from sqlalchemy import text

from database.connector import get_engine

logger = logging.getLogger(__name__)

# All 30 raw indicator columns (matches StockIndicatorHistory model)
INDICATOR_COLS: list[str] = [
    "dy", "p_l", "peg_ratio", "p_vp", "ev_ebitda", "ev_ebit",
    "p_ebitda", "p_ebit", "vpa", "p_ativo", "lpa", "p_sr",
    "p_cap_giro", "p_ativo_circ_liq", "div_liq_pl", "div_liq_ebitda",
    "div_liq_ebit", "pl_ativos", "passivos_ativos", "m_bruta",
    "m_ebitda", "m_ebit", "m_liquida", "roe", "roa", "roic",
    "giro_ativos", "liq_corrente", "cagr_receitas_5", "cagr_lucros_5",
    "graham_formula",
]


def load_indicators() -> pd.DataFrame:
    """
    Loads stock_indicators_history joined with sector metadata from stocks.

    Returns:
        DataFrame with columns:
            ticker, year, sectorname, subsectorname, segmentname,
            + 31 indicator columns (INDICATOR_COLS)
    """
    query = text("""
        SELECT
            h.ticker,
            h.year,
            COALESCE(s.sectorname,    'Sem Setor')    AS sectorname,
            COALESCE(s.subsectorname, 'Sem Subsetor') AS subsectorname,
            COALESCE(s.segmentname,   'Sem Segmento') AS segmentname,
            h.dy, h.p_l, h.peg_ratio, h.p_vp, h.ev_ebitda, h.ev_ebit,
            h.p_ebitda, h.p_ebit, h.vpa, h.p_ativo, h.lpa, h.p_sr,
            h.p_cap_giro, h.p_ativo_circ_liq, h.div_liq_pl, h.div_liq_ebitda,
            h.div_liq_ebit, h.pl_ativos, h.passivos_ativos, h.m_bruta,
            h.m_ebitda, h.m_ebit, h.m_liquida, h.roe, h.roa, h.roic,
            h.giro_ativos, h.liq_corrente, h.cagr_receitas_5, h.cagr_lucros_5,
            h.graham_formula
        FROM stock_indicators_history h
        JOIN stocks s ON h.ticker = s.ticker
        ORDER BY h.ticker, h.year
    """)

    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn)

    logger.info(
        f"Indicators loaded: {len(df)} rows, "
        f"{df['ticker'].nunique()} tickers, "
        f"years {df['year'].min()}–{df['year'].max()}"
    )
    return df


def load_annual_prices() -> pd.DataFrame:
    """
    Aggregates weekly prices in stock_prices_history into annual mean closing prices.

    Returns:
        DataFrame with columns: ticker, year (int), price_mean (float)
    """
    query = text("""
        SELECT
            ticker,
            EXTRACT(YEAR FROM date)::INTEGER AS year,
            AVG(close_price)                 AS price_mean
        FROM stock_prices_history
        WHERE close_price IS NOT NULL
        GROUP BY ticker, EXTRACT(YEAR FROM date)
        ORDER BY ticker, year
    """)

    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn)

    df["year"] = df["year"].astype(int)
    logger.info(
        f"Annual prices loaded: {df['ticker'].nunique()} tickers, "
        f"years {df['year'].min()}–{df['year'].max()}"
    )
    return df


def build_dataset() -> pd.DataFrame:
    """
    Joins indicators with annual prices (current year) and next-year price.

    The next-year price is what enables labeling: if a stock's price rises
    >15% vs the benchmark in the following year, it's labeled BARATA.

    Returns:
        DataFrame with all indicator columns plus:
            price_current  — mean annual price for that indicator year
            price_next     — mean annual price for year + 1 (may be NaN for last year)
    """
    df_indicators = load_indicators()
    df_prices = load_annual_prices()

    # Join current-year price
    df = df_indicators.merge(
        df_prices.rename(columns={"price_mean": "price_current"}),
        on=["ticker", "year"],
        how="left",
    )

    # Join next-year price: shift the year back by 1 so it aligns on current year
    df_next = df_prices.copy()
    df_next["year"] = df_next["year"] - 1
    df = df.merge(
        df_next.rename(columns={"price_mean": "price_next"}),
        on=["ticker", "year"],
        how="left",
    )

    n_labeled = df["price_next"].notna().sum()
    logger.info(
        f"Dataset built: {len(df)} rows total, "
        f"{n_labeled} rows have next-year price (usable for labeling)"
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = build_dataset()
    print("\n── Sample (5 rows) ──")
    print(df[["ticker", "year", "sectorname", "price_current", "price_next"]].head())
    print(f"\n── Null counts (top 10) ──")
    print(df[INDICATOR_COLS].isna().sum().sort_values(ascending=False).head(10))
