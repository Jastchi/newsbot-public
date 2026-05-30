"""Email provider configuration dataclasses and loaders."""

import logging
import os
from dataclasses import dataclass

from utilities import is_truthy_env

logger = logging.getLogger(__name__)


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass
class SMTPConfig:
    smtp_server: str
    smtp_port: int
    use_ssl: bool
    sender_email: str
    login_email: str
    sender_password: str
    cancellation_email: str


@dataclass
class ResendConfig:
    api_key: str
    sender_email: str
    sender_name: str
    cancellation_email: str


@dataclass
class EmailJSConfig:
    service_id: str
    template_id: str
    user_id: str
    private_key: str | None
    sender_email: str


def _get_smtp_config() -> SMTPConfig | None:
    sender_email = _env_str("EMAIL_SENDER")
    sender_password = _env_str("EMAIL_PASSWORD")
    if not sender_email or not sender_password:
        logger.warning(
            "Email not configured: EMAIL_SENDER and EMAIL_PASSWORD required",
        )
        return None
    login_email = _env_str("EMAIL_LOGIN", sender_email) or sender_email
    return SMTPConfig(
        smtp_server=_env_str("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
        smtp_port=int(_env_str("EMAIL_SMTP_PORT", "587")),
        use_ssl=is_truthy_env("EMAIL_USE_SSL"),
        sender_email=sender_email,
        login_email=login_email,
        sender_password=sender_password,
        cancellation_email=_env_str("EMAIL_FOR_CANCELLATION", sender_email),
    )


def _get_resend_config() -> ResendConfig | None:
    api_key = _env_str("RESEND_API_KEY")
    sender_email = _env_str("EMAIL_SENDER")
    if not api_key or not sender_email:
        logger.warning(
            "Resend not configured: RESEND_API_KEY and EMAIL_SENDER required",
        )
        return None
    return ResendConfig(
        api_key=api_key,
        sender_email=sender_email,
        sender_name=_env_str("EMAIL_SENDER_NAME", "NewsBot"),
        cancellation_email=_env_str("EMAIL_FOR_CANCELLATION", sender_email),
    )


def _get_emailjs_config() -> EmailJSConfig | None:
    service_id = _env_str("EMAILJS_SERVICE_ID")
    template_id = _env_str("EMAILJS_TEMPLATE_ID")
    user_id = _env_str("EMAILJS_USER_ID")
    sender_email = _env_str("EMAIL_SENDER")
    if not all((service_id, template_id, user_id, sender_email)):
        logger.warning(
            "EmailJS not configured: EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, "
            "EMAILJS_USER_ID, EMAIL_SENDER required",
        )
        return None
    return EmailJSConfig(
        service_id=service_id,
        template_id=template_id,
        user_id=user_id,
        private_key=_env_str("EMAILJS_PRIVATE_KEY") or None,
        sender_email=sender_email,
    )
