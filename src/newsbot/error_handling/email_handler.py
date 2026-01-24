"""Settings for email error handling."""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class EmailErrorHandler(logging.Handler):
    """
    Custom logging handler that collects ERROR and CRITICAL logs.

    Collects errors during a run and sends them in a single email when
    flush() is called.

    - Errors are collected in memory during a run and sent as one email
      when flush() is called.
    - Each distinct error is collected only once, with all timestamps
      tracked.
    - If send_emails is False (when smtp_host == "no-server"), errors
      are not collected.
    - Logs from specific loggers can be ignored up to a configured level
      (e.g., {LOGGER: LEVEL} ignores logs from the logger {LOGGER} at or
      below level {LEVEL}).
    """

    # Map logger name to max log level to ignore
    IGNORED_LOG_LEVELS: ClassVar[dict[str, int]] = {
        "trafilatura": logging.ERROR,
    }

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        from_email: str,
        to_email: str,
        password: str,
        subject_prefix: str = "[NewsBot Error]",
    ) -> None:
        """Initialize the email handler."""
        super().__init__(level=logging.ERROR)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_email = from_email
        self.to_email = to_email
        self.password = password
        self.subject_prefix = subject_prefix
        self.send_emails = True
        self.config_name: str = ""
        # Map distinct errors to their timestamps and a representative
        # record
        self.error_records: dict[str, tuple[logging.LogRecord, list[str]]] = {}

        if not self.smtp_host:
            self.send_emails = False
            logger.debug("EmailErrorHandler is not configured.")

    def _get_error_key(self, record: logging.LogRecord) -> str:
        """
        Generate a unique key for an error based on its characteristics.

        Args:
            record: Log record to generate key for

        Returns:
            Unique string identifier for this error type

        """
        return (
            f"{record.name}:{record.funcName}:{record.lineno}:"
            f"{record.getMessage()}"
        )

    def emit(self, record: logging.LogRecord) -> None:
        """
        Collect error log records.

        Each distinct error is collected only once, with all timestamps
        tracked. Logs from ignored loggers are skipped.

        Args:
            record: Log record to collect

        """
        if not self.send_emails:
            return

        # Skip logs from specific loggers up to their configured max
        # level
        for logger_prefix, max_level in self.IGNORED_LOG_LEVELS.items():
            if (
                record.name.startswith(logger_prefix)
                and record.levelno <= max_level
            ):
                return

        error_key = self._get_error_key(record)
        timestamp = record.asctime

        if error_key in self.error_records:
            # Add timestamp to existing error
            self.error_records[error_key][1].append(timestamp)
        else:
            # Store new distinct error with its first timestamp
            self.error_records[error_key] = (record, [timestamp])

    def flush(self) -> None:
        """
        Send all collected errors in a single email.

        Each distinct error is shown once with all its timestamps.
        If no errors were collected, no email is sent.
        """
        if not self.send_emails or not self.error_records:
            self.error_records.clear()
            return

        try:
            distinct_error_count = len(self.error_records)
            total_error_count = sum(
                len(timestamps)
                for _, timestamps in self.error_records.values()
            )
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = self.to_email
            config_info = f" [{self.config_name}]" if self.config_name else ""
            msg["Subject"] = (
                f"{self.subject_prefix}{config_info} "
                f"{distinct_error_count} Distinct Error(s) "
                f"({total_error_count} Total Occurrences) "
                f"in NewsBot Run"
            )

            # Create email body with all error details
            template_path = Path(__file__).parent / "email_template.txt"
            template = template_path.read_text(encoding="utf-8")

            # Build body with all errors
            config_info = f" [{self.config_name}]" if self.config_name else ""
            body_parts = [
                (
                    f"NewsBot{config_info} encountered "
                    f"{distinct_error_count} distinct error(s) "
                    f"({total_error_count} total occurrence(s)) "
                    f"during this run.\n"
                ),
                "=" * 70 + "\n",
            ]

            for i, (_, (record, timestamps)) in enumerate(
                self.error_records.items(),
                1,
            ):
                occurrence_count = len(timestamps)
                body_parts.append(
                    f"\n--- Error {i} of {distinct_error_count} "
                    f"({occurrence_count} occurrence(s)) ---\n",
                )
                body_parts.append(
                    template.format(
                        levelname=record.levelname,
                        module=record.name,
                        function=record.funcName,
                        line=record.lineno,
                        time=", ".join(timestamps),
                        message=record.getMessage(),
                        trace=self.format(record),
                    ),
                )
                body_parts.append("\n")

            body = "".join(body_parts)
            msg.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.from_email, self.password)
                server.send_message(msg)

            logger.info(
                f"Sent error email to {self.to_email} "
                f"with {distinct_error_count} distinct error(s) "
                f"({total_error_count} total occurrence(s))",
            )

        except Exception:
            # Don't let email failures crash the application
            logger.warning(
                f"Failed to send error email to {self.to_email} "
                f"for {len(self.error_records)} distinct error(s)",
            )
        finally:
            # Clear collected errors after attempting to send
            self.error_records.clear()


def get_email_error_handler() -> EmailErrorHandler:
    """Create and return an EmailErrorHandler instance."""
    # Check if email is enabled
    if os.getenv("EMAIL_ENABLED", "false").lower() not in ("true", "1", "yes"):
        # Return handler with empty smtp_host to disable email sending
        return EmailErrorHandler(
            smtp_host="",
            smtp_port=0,
            from_email="",
            to_email="",
            password="",
        )

    return EmailErrorHandler(
        smtp_host=os.getenv("EMAIL_SMTP_SERVER", ""),
        smtp_port=int(os.getenv("EMAIL_SMTP_PORT", "0")),
        from_email=os.getenv("EMAIL_SENDER", ""),
        to_email=os.getenv("EMAIL_RECIPIENT", ""),
        password=os.getenv("EMAIL_PASSWORD", ""),
        subject_prefix=os.getenv("EMAIL_SUBJECT_PREFIX", "[NewsBot Error]"),
    )
