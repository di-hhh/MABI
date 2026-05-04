"""Database engine singleton — SQLAlchemy + PyMySQL."""
import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env")

_engine = None


def get_engine():
    """Return the SQLAlchemy engine singleton."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 30,
                "read_timeout": 600,
                "write_timeout": 600,
            },
        )
        logger.info("Database engine created: %s", DATABASE_URL.split("@")[-1])
    return _engine


def execute_query(sql: str, params=None):
    """Execute a read-only SQL query, return list of dict rows."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            rows = result.fetchall()
            if rows:
                return [dict(row._mapping) for row in rows]
            return []
    except SQLAlchemyError as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise


def execute_sql(sql: str, params=None):
    """Execute a SQL statement (DDL / DML)."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text(sql), params or {})
        logger.info("SQL executed successfully.")
    except SQLAlchemyError as e:
        logger.error("SQL execution failed: %s", e, exc_info=True)
        raise


def get_table_names():
    """Return a set of table and view names in the database."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SHOW FULL TABLES"))
        tables = set()
        for row in rows:
            tables.add(row[0])
        return tables


def table_exists(name: str) -> bool:
    """Check if a table or view exists."""
    return name in get_table_names()
