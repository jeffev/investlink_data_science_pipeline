"""
Creates the data science pipeline tables in the existing InvestLink PostgreSQL database.

Run from the data_science_pipeline root:
    python database/migrations.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from database.connector import get_engine
from database.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TABLES = [
    "stock_indicators_history",
    "stock_prices_history",
    "stock_predictions",
]


def create_tables():
    engine = get_engine()
    Base.metadata.create_all(engine, checkfirst=True)
    logger.info(f"Tables created (if not existed): {', '.join(TABLES)}")


def drop_tables():
    """Drops all pipeline tables. Use only in development."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.warning(f"Tables dropped: {', '.join(TABLES)}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "create"
    if action == "drop":
        confirm = input("Drop all pipeline tables? This cannot be undone. Type YES to confirm: ")
        if confirm == "YES":
            drop_tables()
    else:
        create_tables()
