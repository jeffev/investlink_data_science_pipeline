import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:123@localhost:5433/investlink",
        )
        _engine = create_engine(database_url, pool_pre_ping=True)
        logger.info("Database engine created")
    return _engine


def get_session():
    Session = sessionmaker(bind=get_engine())
    return Session()
