"""Database logging handler for storing logs in Django database."""

import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from newsbot.constants import TZ

# Create a separate logger for database handler errors
# (won't use DatabaseLoggingHandler)
_db_handler_logger = logging.getLogger(
    "newsbot.error_handling.database_handler",
)


class DatabaseLoggingHandler(logging.Handler):
    """Handler that batches and writes logs to SQLite database."""

    BATCH_SIZE = 100
    CONFIG_NAME_PATTERN = re.compile(r"(\w+)\s*\[")  # Parse from log line

    def __init__(self, db_path: Path) -> None:
        """
        Initialize the database logging handler.

        Args:
            db_path: Path to the SQLite database (webserver.sqlite3)

        """
        super().__init__()
        self.db_path = db_path
        self.buffer: list[tuple] = []
        # Use a formatter with milliseconds in the timestamp
        self.setFormatter(
            logging.Formatter(
                (
                    "%(asctime)s,%(msecs)03d - %(name)s "
                    "- %(levelname)s - %(message)s"
                ),
                datefmt="%Y-%m-%d %H:%M:%S",
            ),
        )

    def emit(self, record: logging.LogRecord) -> None:
        """
        Add log record to buffer.

        When buffer reaches BATCH_SIZE, it's automatically flushed.

        Args:
            record: The log record to buffer

        """
        try:
            # Extract config name from the formatted message
            formatted_message = self.format(record)
            config_name = self._extract_config_name(formatted_message) or ""

            # Create tuple for database insertion
            # Convert datetime to ISO format string to avoid deprecation
            # warning
            timestamp = datetime.fromtimestamp(
                record.created,
                tz=TZ,
            ).isoformat()
            log_entry = (
                timestamp,
                record.levelname,
                record.name,
                record.getMessage(),
                config_name,
            )

            self.buffer.append(log_entry)

            # Flush if buffer is full
            if len(self.buffer) >= self.BATCH_SIZE:
                self.flush()

        except Exception:
            # Log to separate logger that doesn't use
            # DatabaseLoggingHandler
            _db_handler_logger.exception(
                "Error in DatabaseLoggingHandler.emit(): %s",
            )

    def flush(self) -> None:
        """
        Write all buffered log records to the database.

        Catches and silently ignores SQL errors.
        """
        if not self.buffer:
            return

        if not self.db_path.exists():
            _db_handler_logger.error(
                "Database does not exist at %s, skipping log flush",
                self.db_path,
            )
            self.buffer.clear()
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert all buffered records into Django-created table
            # No need to create table - Django migration handles it
            cursor.executemany(
                """
                INSERT INTO newsserver_logentry
                (timestamp, level, logger_name, message, config_name,
                created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                self.buffer,
            )

            conn.commit()
            conn.close()

            # Clear buffer after successful write
            self.buffer.clear()

        except Exception:
            # Log error using the separate logger
            _db_handler_logger.exception(
                "Error writing logs to database at %s",
                self.db_path,
            )
            # Still clear buffer to avoid memory bloat
            self.buffer.clear()

    def _extract_config_name(self, formatted_message: str) -> str:
        """
        Extract config name from formatted log message.

        Expected format:
        "2025-12-04 22:06:35,617 Technology [INFO - ..."
        Extracts "Technology" from the pattern.

        Args:
            formatted_message: The formatted log message

        Returns:
            The config name if found, otherwise empty string

        """
        match = self.CONFIG_NAME_PATTERN.search(formatted_message)
        if match:
            return match.group(1)
        return ""

    def close(self) -> None:
        """Flush remaining logs and close the handler."""
        self.flush()
        super().close()
