"""
Database read/write helpers for the data science pipeline.
All functions receive a SQLAlchemy session as first argument.
"""
import logging
from sqlalchemy import text, func
from sqlalchemy.dialects.postgresql import insert

from database.models import StockIndicatorHistory, StockPriceHistory, StockPrediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# READ helpers
# ---------------------------------------------------------------------------

def get_all_tickers(session) -> list[str]:
    """Returns all ticker symbols present in the stocks table."""
    result = session.execute(text("SELECT ticker FROM stocks ORDER BY ticker"))
    return [row[0] for row in result]


def get_scraped_years(session, ticker: str) -> set[int]:
    """Returns the set of years already stored for a given ticker."""
    rows = (
        session.query(StockIndicatorHistory.year)
        .filter(StockIndicatorHistory.ticker == ticker)
        .all()
    )
    return {row[0] for row in rows}


def get_price_date_range(session, ticker: str) -> tuple:
    """
    Returns (min_date, max_date) for prices already stored for a ticker.
    Both values are None if no data exists.
    """
    result = (
        session.query(
            func.min(StockPriceHistory.date),
            func.max(StockPriceHistory.date),
        )
        .filter(StockPriceHistory.ticker == ticker)
        .first()
    )
    return result  # (None, None) when no rows exist


def has_prices(session, ticker: str) -> bool:
    """Returns True if there is any price record for the ticker."""
    min_date, _ = get_price_date_range(session, ticker)
    return min_date is not None


# ---------------------------------------------------------------------------
# WRITE helpers
# ---------------------------------------------------------------------------

def save_indicators(session, records: list[dict]) -> int:
    """
    Bulk-inserts indicator records into stock_indicators_history.
    Skips records where (ticker, year) already exists (do nothing on conflict).

    Args:
        records: list of dicts with keys matching StockIndicatorHistory columns.

    Returns:
        Number of rows actually inserted.
    """
    if not records:
        return 0

    stmt = insert(StockIndicatorHistory).values(records)
    stmt = stmt.on_conflict_do_nothing(index_elements=["ticker", "year"])
    result = session.execute(stmt)
    session.commit()
    inserted = result.rowcount if result.rowcount != -1 else len(records)
    logger.debug(f"save_indicators: {inserted} rows inserted")
    return inserted


def save_prices(session, records: list[dict]) -> int:
    """
    Bulk-inserts weekly price records into stock_prices_history.
    Skips records where (ticker, date) already exists.

    Args:
        records: list of dicts with keys matching StockPriceHistory columns.

    Returns:
        Number of rows actually inserted.
    """
    if not records:
        return 0

    stmt = insert(StockPriceHistory).values(records)
    stmt = stmt.on_conflict_do_nothing(index_elements=["ticker", "date"])
    result = session.execute(stmt)
    session.commit()
    inserted = result.rowcount if result.rowcount != -1 else len(records)
    logger.debug(f"save_prices: {inserted} rows inserted")
    return inserted


def save_predictions(session, records: list[dict]) -> int:
    """
    Upserts ML prediction records into stock_predictions.
    On conflict (ticker, run_date): updates label and scores.

    Args:
        records: list of dicts with keys matching StockPrediction columns.

    Returns:
        Number of rows affected.
    """
    if not records:
        return 0

    stmt = insert(StockPrediction).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "run_date"],
        set_={
            "label": stmt.excluded.label,
            "prob_barata": stmt.excluded.prob_barata,
            "prob_neutra": stmt.excluded.prob_neutra,
            "prob_cara": stmt.excluded.prob_cara,
            "composite_score": stmt.excluded.composite_score,
            "model_version": stmt.excluded.model_version,
        },
    )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount if result.rowcount != -1 else len(records)
