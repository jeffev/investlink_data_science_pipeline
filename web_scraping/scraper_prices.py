"""
Downloads weekly historical stock prices from Yahoo Finance (Brazilian tickers use .SA suffix).

Adapted from analise-mercado-financeiro-brasil/src/data_processing/processor.py.
Key differences:
  - Returns list of dicts ready for DB insertion instead of a DataFrame.
  - Uses yf.Ticker().history() to avoid MultiIndex column issues.
  - Date range defaults to 2007-01-01 (covers all historical indicators back to 2008).
"""
import logging
from datetime import date

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_START = "2007-01-01"


def scrape_prices(
    ticker: str,
    start_date: str = DEFAULT_START,
    end_date: str | None = None,
) -> list[dict]:
    """
    Downloads weekly closing prices for a Brazilian stock from Yahoo Finance.

    Args:
        ticker:     Stock ticker without suffix, e.g. "VALE3".
        start_date: ISO date string for the start of the range.
        end_date:   ISO date string for the end (defaults to today).

    Returns:
        List of dicts: each dict has keys
            ticker, date, close_price, open_price, high, low, volume
        Returns [] on failure or if Yahoo Finance has no data.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    yahoo_ticker = f"{ticker}.SA"

    try:
        obj = yf.Ticker(yahoo_ticker)
        df = obj.history(start=start_date, end=end_date, interval="1wk")

        if df.empty:
            logger.warning(f"[{ticker}] No price data from Yahoo Finance")
            return []

        df = df.reset_index()

        records = []
        for _, row in df.iterrows():
            row_date = row["Date"]
            if hasattr(row_date, "date"):
                row_date = row_date.date()

            records.append(
                {
                    "ticker": ticker,
                    "date": row_date,
                    "close_price": _safe_float(row.get("Close")),
                    "open_price": _safe_float(row.get("Open")),
                    "high": _safe_float(row.get("High")),
                    "low": _safe_float(row.get("Low")),
                    "volume": _safe_float(row.get("Volume")),
                }
            )

        logger.info(f"[{ticker}] Downloaded {len(records)} weekly price records")
        return records

    except Exception as exc:
        logger.error(f"[{ticker}] Price download failed: {exc}")
        return []


def _safe_float(value) -> float | None:
    try:
        f = float(value)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None
