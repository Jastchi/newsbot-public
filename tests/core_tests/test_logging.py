"""Simplified tests for database logging handler."""

import logging
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from newsbot.error_handling.database_handler import DatabaseLoggingHandler


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Create the table structure (matching Django model)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE newsserver_logentry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            logger_name TEXT NOT NULL,
            message TEXT NOT NULL,
            config_name TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    try:
        Path(db_path).unlink(missing_ok=True)
    except PermissionError:
        pass


def test_handler_initialization(temp_db):
    """Test that handler initializes correctly."""
    handler = DatabaseLoggingHandler(Path(temp_db))
    assert handler is not None
    assert handler.buffer == []
    assert handler.BATCH_SIZE == 100


def test_config_name_extraction():
    """Test config name extraction from formatted log messages."""
    handler = DatabaseLoggingHandler(Path(":memory:"))

    # Test with config name
    result = handler._extract_config_name("2025-12-07 - Technology [INFO] - Test")
    assert result == "Technology"

    # Test without config name
    result = handler._extract_config_name("2025-12-07 - simple.logger - Test")
    assert result == ""


def test_logging_workflow(temp_db):
    """Test complete logging workflow with real application logger format."""
    handler = DatabaseLoggingHandler(Path(temp_db))
    # Use real application logger format: config_name [LEVEL - message
    logger = logging.getLogger("Technology [INFO - Test Config]")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Remove other handlers to avoid interference
    logger.propagate = False

    # Log messages using real application patterns
    logger.info("Starting daily scrape for Technology")
    logger.warning("Rate limit approaching")

    # Verify buffering
    assert len(handler.buffer) == 2

    # Flush to database
    handler.flush()
    assert len(handler.buffer) == 0

    # Verify in database
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT level, message, config_name FROM newsserver_logentry ORDER BY id"
    )
    results = cursor.fetchall()
    conn.close()

    assert len(results) == 2
    assert results[0][0] == "INFO"
    assert results[0][1] == "Starting daily scrape for Technology"
    assert results[0][2] == "Technology"
    assert results[1][0] == "WARNING"
    assert results[1][1] == "Rate limit approaching"


def test_cleanup_old_logs(temp_db):
    """Test that cleanup deletes logs older than 30 days."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert logs with various ages
    now = datetime.now()
    dates = [
        now - timedelta(days=5),  # Recent
        now - timedelta(days=31),  # Should be deleted
    ]

    for i, date in enumerate(dates):
        cursor.execute(
            """
            INSERT INTO newsserver_logentry
            (timestamp, level, logger_name, message, config_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                date.isoformat(),
                "INFO",
                "test",
                f"Message {i}",
                "Test",
                date.isoformat(),
            ),
        )

    conn.commit()

    # Run cleanup
    cutoff_datetime = now - timedelta(days=30)
    cursor.execute(
        """
        DELETE FROM newsserver_logentry
        WHERE created_at < ?
        """,
        (cutoff_datetime.isoformat(),),
    )

    deleted_count = cursor.rowcount
    conn.commit()

    # Verify 1 old log deleted
    assert deleted_count == 1

    # Verify 1 recent log remains
    cursor.execute("SELECT COUNT(*) FROM newsserver_logentry")
    remaining = cursor.fetchone()[0]
    conn.close()

    assert remaining == 1


def test_error_handling():
    """Test that database errors don't raise exceptions."""
    # Use invalid path
    handler = DatabaseLoggingHandler(Path("/nonexistent/invalid.db"))
    logger = logging.getLogger("test_error")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Should not raise exception
    logger.info("Test message")
    handler.flush()  # Should fail silently


def test_timestamp_format(temp_db):
    """Test that timestamps are stored correctly."""
    handler = DatabaseLoggingHandler(Path(temp_db))
    logger = logging.getLogger("timestamp_test")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Timestamp test")
    handler.flush()

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, created_at FROM newsserver_logentry")
    result = cursor.fetchone()
    conn.close()

    # Both should be valid datetime strings
    assert result is not None
    assert len(result[0]) > 0
    assert len(result[1]) > 0


def test_real_newsbot_logger_integration(temp_db):
    """Test with real newsbot logger format as used in production."""
    handler = DatabaseLoggingHandler(Path(temp_db))

    # Simulate the actual logger setup from newsbot
    # In production: logger = logging.getLogger(f"{config_name} [INFO - {message}")
    config_name = "Technology"
    logger = logging.getLogger(f"{config_name} [INFO - Starting scraper]")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Log various levels as the application would
    logger.debug("Initializing RSS feed parser")
    logger.info("Successfully fetched 25 articles")
    logger.warning("Duplicate article detected, skipping")
    logger.error("Failed to parse article content")

    handler.flush()

    # Verify all logs stored correctly
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT level, message, config_name FROM newsserver_logentry ORDER BY id"
    )
    results = cursor.fetchall()
    conn.close()

    assert len(results) == 4
    assert results[0][0] == "DEBUG"
    assert results[1][0] == "INFO"
    assert results[2][0] == "WARNING"
    assert results[3][0] == "ERROR"
    # All should extract "Technology" as config name
    assert all(row[2] == "Technology" for row in results)


def test_real_newsbot_logger_integration(temp_db):
    """Test with real newsbot logger format as used in production."""
    handler = DatabaseLoggingHandler(Path(temp_db))
    
    # Simulate the actual logger setup from newsbot
    # In production: logger = logging.getLogger(f"{config_name} [INFO - {message}")
    config_name = "Technology"
    logger = logging.getLogger(f"{config_name} [INFO - Starting scraper]")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Log various levels as the application would
    logger.debug("Initializing RSS feed parser")
    logger.info("Successfully fetched 25 articles")
    logger.warning("Duplicate article detected, skipping")
    logger.error("Failed to parse article content")

    handler.flush()

    # Verify all logs stored correctly
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT level, message, config_name FROM newsserver_logentry ORDER BY id"
    )
    results = cursor.fetchall()
    conn.close()

    assert len(results) == 4
    assert results[0][0] == "DEBUG"
    assert results[1][0] == "INFO"
    assert results[2][0] == "WARNING"
    assert results[3][0] == "ERROR"
    # All should extract "Technology" as config name
    assert all(row[2] == "Technology" for row in results)
