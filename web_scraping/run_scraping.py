"""
Orchestrates the full scraping pipeline: indicators + prices.

Incremental by default:
  - Indicators: skips (ticker, year) pairs already in the DB.
    The current year is always re-scraped to catch mid-year updates.
  - Prices: skips tickers that already have price records.
    Use --force to re-download prices even if they exist.

Usage examples:
    # Scrape everything for all tickers in the stocks table
    python web_scraping/run_scraping.py

    # Only indicators, for specific tickers, no headless browser
    python web_scraping/run_scraping.py --mode indicators --tickers VALE3 PETR4 --no-headless

    # Force re-download prices for all tickers
    python web_scraping/run_scraping.py --mode prices --force

Run from the data_science_pipeline root directory.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime

from database.connector import get_session
from database.queries import (
    get_all_tickers,
    get_scraped_years,
    has_prices,
    save_indicators,
    save_prices,
)
from web_scraping.scraper_indicators import scrape_indicators
from web_scraping.scraper_prices import scrape_prices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraping.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_indicators(tickers: list[str], headless: bool = True) -> int:
    """
    Scrapes historical indicators for each ticker and saves new years to DB.
    Always re-scrapes the current year to capture mid-year updates.

    Returns:
        Total number of new records inserted across all tickers.
    """
    session = get_session()
    current_year = datetime.now().year
    total_inserted = 0

    for ticker in tickers:
        logger.info(f"── Indicators: {ticker}")

        existing_years = get_scraped_years(session, ticker)
        year_data = scrape_indicators(ticker, headless=headless)

        if not year_data:
            logger.warning(f"[{ticker}] No data returned, skipping")
            continue

        records = []
        for year, indicators in year_data.items():
            # Always re-scrape current year; skip older years already stored
            if year in existing_years and year < current_year:
                continue
            records.append({"ticker": ticker, "year": year, **indicators})

        if not records:
            logger.info(f"[{ticker}] All years already stored")
            continue

        inserted = save_indicators(session, records)
        total_inserted += inserted
        logger.info(f"[{ticker}] Saved {inserted} new year-records")

    session.close()
    logger.info(f"Indicators done — {total_inserted} records inserted total")
    return total_inserted


def run_prices(tickers: list[str], force: bool = False) -> int:
    """
    Downloads weekly prices for each ticker and saves to DB.
    Skips tickers that already have price data unless force=True.

    Returns:
        Total number of new records inserted across all tickers.
    """
    session = get_session()
    total_inserted = 0

    for ticker in tickers:
        logger.info(f"── Prices: {ticker}")

        if not force and has_prices(session, ticker):
            logger.info(f"[{ticker}] Prices already stored, skipping (use --force to refresh)")
            continue

        records = scrape_prices(ticker)
        if not records:
            logger.warning(f"[{ticker}] No price data returned, skipping")
            continue

        inserted = save_prices(session, records)
        total_inserted += inserted
        logger.info(f"[{ticker}] Saved {inserted} price records")

    session.close()
    logger.info(f"Prices done — {total_inserted} records inserted total")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(
        description="InvestLink scraping pipeline — indicators and prices"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "indicators", "prices"],
        default="all",
        help="What to scrape (default: all)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Specific tickers to process (default: all from DB stocks table)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run Selenium in headless mode (default: True)",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Show browser window (useful for debugging)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape prices even if data already exists",
    )

    args = parser.parse_args()

    # Resolve ticker list
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        session = get_session()
        tickers = get_all_tickers(session)
        session.close()

    if not tickers:
        logger.error("No tickers found. Make sure the stocks table has data and DATABASE_URL is set.")
        sys.exit(1)

    logger.info(
        f"Starting scraping — mode={args.mode}, tickers={len(tickers)}, "
        f"headless={args.headless}, force={args.force}"
    )
    logger.info(f"First 10 tickers: {tickers[:10]}")

    if args.mode in ("all", "indicators"):
        run_indicators(tickers, headless=args.headless)

    if args.mode in ("all", "prices"):
        run_prices(tickers, force=args.force)

    logger.info("Scraping pipeline complete.")


if __name__ == "__main__":
    main()
