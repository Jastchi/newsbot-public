import logging
from unittest.mock import Mock, patch

from newsbot.error_handling.email_handler import EmailErrorHandler


def _log_record(
    message: str = "boom",
    logger_name: str = "newsbot.test",
    level: int = logging.ERROR,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=logger_name,
        level=level,
        pathname=__file__,
        lineno=10,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.asctime = "now"
    return record


def test_emit_and_flush_sends_email(monkeypatch):
    handler = EmailErrorHandler(
        smtp_host="smtp.test",
        smtp_port=587,
        from_email="from@test.com",
        to_email="to@test.com",
        password="pwd",
    )

    handler.emit(_log_record("first error"))
    handler.emit(_log_record("first error"))

    smtp_mock = Mock()
    smtp_mock.__enter__ = lambda self=smtp_mock: self
    smtp_mock.__exit__ = Mock(return_value=False)
    monkeypatch.setattr("smtplib.SMTP", Mock(return_value=smtp_mock))

    handler.flush()

    smtp_mock.starttls.assert_called_once()
    smtp_mock.login.assert_called_once_with("from@test.com", "pwd")
    smtp_mock.send_message.assert_called_once()
    assert handler.error_records == {}


def test_flush_empty_smtp_host_disables_collection(monkeypatch):
    handler = EmailErrorHandler(
        smtp_host="",
        smtp_port=0,
        from_email="",
        to_email="",
        password="",
    )

    handler.emit(_log_record("ignored"))

    smtp = Mock()
    monkeypatch.setattr("smtplib.SMTP", smtp)

    handler.flush()

    smtp.assert_not_called()
    assert handler.error_records == {}


def test_flush_handles_send_failure(monkeypatch):
    handler = EmailErrorHandler(
        smtp_host="smtp.test",
        smtp_port=587,
        from_email="from@test.com",
        to_email="to@test.com",
        password="pwd",
    )

    handler.emit(_log_record("first error"))

    failing_smtp = Mock()
    failing_smtp.__enter__ = lambda self=failing_smtp: self
    failing_smtp.__exit__ = Mock(return_value=False)
    failing_smtp.send_message.side_effect = RuntimeError("fail")
    failing_smtp.starttls = Mock()
    failing_smtp.login = Mock()
    monkeypatch.setattr("smtplib.SMTP", Mock(return_value=failing_smtp))

    handler.flush()

    # Errors should be cleared even on send failure
    assert handler.error_records == {}


def test_get_email_error_handler_enabled(monkeypatch):
    monkeypatch.setenv("EMAIL_ENABLED", "true")
    monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.test.com")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "587")
    monkeypatch.setenv("EMAIL_SENDER", "from@test.com")
    monkeypatch.setenv("EMAIL_RECIPIENT", "to@test.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "pwd")
    monkeypatch.setenv("EMAIL_SUBJECT_PREFIX", "[Test Error]")

    from newsbot.error_handling.email_handler import get_email_error_handler

    handler = get_email_error_handler()

    assert isinstance(handler, EmailErrorHandler)
    assert handler.smtp_host == "smtp.test.com"
    assert handler.smtp_port == 587
    assert handler.from_email == "from@test.com"
    assert handler.to_email == "to@test.com"
    assert handler.password == "pwd"
    assert handler.subject_prefix == "[Test Error]"
    assert handler.send_emails is True


def test_get_email_error_handler_disabled_false(monkeypatch):
    monkeypatch.setenv("EMAIL_ENABLED", "false")

    from newsbot.error_handling.email_handler import get_email_error_handler

    handler = get_email_error_handler()

    assert isinstance(handler, EmailErrorHandler)
    assert handler.smtp_host == ""
    assert handler.send_emails is False


def test_get_email_error_handler_disabled_not_set(monkeypatch):
    monkeypatch.delenv("EMAIL_ENABLED", raising=False)

    from newsbot.error_handling.email_handler import get_email_error_handler

    handler = get_email_error_handler()

    assert isinstance(handler, EmailErrorHandler)
    assert handler.smtp_host == ""
    assert handler.send_emails is False


def test_get_email_error_handler_enabled_variations(monkeypatch):
    """Test that EMAIL_ENABLED accepts multiple true values."""
    from newsbot.error_handling.email_handler import get_email_error_handler

    monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.test.com")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "587")
    monkeypatch.setenv("EMAIL_SENDER", "from@test.com")
    monkeypatch.setenv("EMAIL_RECIPIENT", "to@test.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "pwd")

    for enabled_value in ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]:
        monkeypatch.setenv("EMAIL_ENABLED", enabled_value)
        handler = get_email_error_handler()
        assert handler.send_emails is True, (
            f"Failed for EMAIL_ENABLED={enabled_value}"
        )


def test_ignored_log_levels_skip_messages(monkeypatch):
    """Test that messages from loggers in IGNORED_LOG_LEVELS are not counted as errors."""
    handler = EmailErrorHandler(
        smtp_host="smtp.test",
        smtp_port=587,
        from_email="from@test.com",
        to_email="to@test.com",
        password="pwd",
    )

    # Emit an error from an ignored logger (trafilatura)
    handler.emit(_log_record("trafilatura error", logger_name="trafilatura.core"))
    # Verify it was not collected
    assert len(handler.error_records) == 0

    # Emit a regular error that should be collected
    handler.emit(_log_record("regular error", logger_name="newsbot.test"))
    # Verify it was collected
    assert len(handler.error_records) == 1
    assert "regular error" in str(handler.error_records)

    # Emit another error from ignored logger with different message
    handler.emit(_log_record("another trafilatura error", logger_name="trafilatura.utils"))
    # Verify it still wasn't collected
    assert len(handler.error_records) == 1


def test_config_name_in_email_subject_and_body(monkeypatch):
    """Test that config_name is included in email subject and body."""
    handler = EmailErrorHandler(
        smtp_host="smtp.test",
        smtp_port=587,
        from_email="from@test.com",
        to_email="to@test.com",
        password="pwd",
    )
    handler.config_name = "Technology"

    handler.emit(_log_record("test error"))

    sent_message = None

    def capture_send_message(msg):
        nonlocal sent_message
        sent_message = msg

    smtp_mock = Mock()
    smtp_mock.__enter__ = lambda self=smtp_mock: self
    smtp_mock.__exit__ = Mock(return_value=False)
    smtp_mock.send_message = capture_send_message
    monkeypatch.setattr("smtplib.SMTP", Mock(return_value=smtp_mock))

    handler.flush()

    assert sent_message is not None
    assert "[Technology]" in sent_message["Subject"]
    # Check the body contains the config name
    body = sent_message.get_payload()[0].get_payload()
    assert "[Technology]" in body


def test_config_name_empty_not_in_email(monkeypatch):
    """Test that empty config_name is not included in email."""
    handler = EmailErrorHandler(
        smtp_host="smtp.test",
        smtp_port=587,
        from_email="from@test.com",
        to_email="to@test.com",
        password="pwd",
    )
    # config_name defaults to empty string

    handler.emit(_log_record("test error"))

    sent_message = None

    def capture_send_message(msg):
        nonlocal sent_message
        sent_message = msg

    smtp_mock = Mock()
    smtp_mock.__enter__ = lambda self=smtp_mock: self
    smtp_mock.__exit__ = Mock(return_value=False)
    smtp_mock.send_message = capture_send_message
    monkeypatch.setattr("smtplib.SMTP", Mock(return_value=smtp_mock))

    handler.flush()

    assert sent_message is not None
    # Should not have brackets when config_name is empty
    assert "[]" not in sent_message["Subject"]


def test_send_error_email_once_sends_when_enabled(monkeypatch):
    """send_error_email_once sends a single email when EMAIL_ENABLED and SMTP are set."""
    monkeypatch.setenv("EMAIL_ENABLED", "true")
    monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.test.com")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "587")
    monkeypatch.setenv("EMAIL_SENDER", "from@test.com")
    monkeypatch.setenv("EMAIL_RECIPIENT", "to@test.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "pwd")

    smtp_mock = Mock()
    smtp_mock.__enter__ = lambda self=smtp_mock: self
    smtp_mock.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(
        "newsbot.error_handling.email_handler.smtplib.SMTP",
        Mock(return_value=smtp_mock),
    )

    from newsbot.error_handling.email_handler import send_error_email_once

    send_error_email_once(
        "Error loading config 'world'",
        "Traceback (most recent call last):\n  ...",
        config_key="world",
    )

    smtp_mock.starttls.assert_called_once()
    smtp_mock.login.assert_called_once_with("from@test.com", "pwd")
    smtp_mock.send_message.assert_called_once()
    msg = smtp_mock.send_message.call_args[0][0]
    assert "[world]" in msg["Subject"]
    assert "no handler yet" in msg["Subject"]
    body = msg.get_payload()[0].get_payload()
    assert "Error loading config 'world'" in body
    assert "EmailErrorHandler was created" in body


def test_send_error_email_once_skips_when_disabled(monkeypatch):
    """send_error_email_once does nothing when EMAIL_ENABLED is false."""
    monkeypatch.setenv("EMAIL_ENABLED", "false")
    monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.test.com")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "587")
    smtp = Mock()
    monkeypatch.setattr(
        "newsbot.error_handling.email_handler.smtplib.SMTP",
        smtp,
    )

    from newsbot.error_handling.email_handler import send_error_email_once

    send_error_email_once("msg", "tb", config_key="x")

    smtp.assert_not_called()
