"""
Email Hook - Send analysis reports via email.

Configuration (via environment variables in email/*.env files):
---------------------------------------------------------------
EMAIL_ENABLED=true
EMAIL_PROVIDER=smtp | resend | emailjs
EMAIL_SENDER=your@email.com
EMAIL_SENDER_NAME=NewsBot
NEWSSERVER_BASE_URL=https://your-domain.com
UNSUBSCRIBE_TOKEN_SECRET=<long random string>

SMTP: EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, EMAIL_PASSWORD, EMAIL_USE_SSL
Resend: RESEND_API_KEY
EmailJS: EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, EMAILJS_USER_ID,
         EMAILJS_PRIVATE_KEY
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

from newsbot.models import AnalysisData
from utilities import is_truthy_env
from utilities.django_models import Subscriber

from .email._config import (
    _env_str,
    _get_emailjs_config,
    _get_resend_config,
    _get_smtp_config,
)
from .email._providers import (
    _send_via_emailjs,
    _send_via_resend,
    _send_via_smtp,
)
from .email._report import (
    _prepare_base_html,
    get_available_newsletters,
    replace_placeholders_in_report,
)

logger = logging.getLogger(__name__)

__all__ = [
    "execute",
    "get_available_newsletters",
    "replace_placeholders_in_report",
]


# ----------------------------------------------------------------------
# Recipients & subject
# ----------------------------------------------------------------------


def get_recipients_for_config(config_key: str) -> list[str]:
    """Return email addresses of active subscribers for a config key."""
    try:
        emails = list(
            Subscriber.objects.filter(
                is_active=True,
                configs__key=config_key,
            ).values_list("email", flat=True),
        )
        logger.info(
            "Found %d recipients for config key '%s'%s",
            len(emails),
            config_key,
            ": " + ", ".join(emails) if emails else "",
        )
    except Exception:
        logger.exception(
            "Error querying database for config key '%s'.", config_key,
        )
        return []
    else:
        return emails


def _build_subject(topic: str, analysis_data: AnalysisData) -> str:
    stories_count = analysis_data.get("stories_count", "")
    from_date = analysis_data.get("from_date")
    to_date = analysis_data.get("to_date")
    date_range = (
        f"{from_date.strftime('%d %b')} - {to_date.strftime('%d %b')}"
        if from_date and to_date
        else ""
    )
    return f"{stories_count} Top Stories in {topic} - {date_range}"


def _get_recipient_emails(
    analysis_data: AnalysisData,
    config_key: str,
) -> list[str]:
    override = analysis_data.get("email_receivers_override")
    if override is not None:
        if override:
            logger.info(
                "Using overridden email recipients: %s", ", ".join(override),
            )
        else:
            logger.info(
                "Sending to sender only — "
                "--email-receivers passed with no arguments",
            )
        return override
    recipients = get_recipients_for_config(config_key)
    if not recipients:
        logger.info(
            "No active subscribers for config key '%s'. "
            "Sending to sender only.",
            config_key,
        )
    return recipients


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def execute(report_path: Path, analysis_data: AnalysisData) -> None:
    """Send the analysis report via email."""
    report_path = report_path.parent / "email_reports" / report_path.name

    config_name = analysis_data.get("config_name", "News")
    config_key = analysis_data.get("config_key", "default")

    load_dotenv()

    if not is_truthy_env("EMAIL_ENABLED"):
        logger.debug("Email hook disabled (EMAIL_ENABLED not set to true)")
        return

    sender_email = _env_str("EMAIL_SENDER")
    if not sender_email:
        logger.warning("EMAIL_SENDER required for email report placeholders")
        return

    recipient_emails = (
        _get_recipient_emails(analysis_data, config_key) or [sender_email]
    )
    base_html = _prepare_base_html(report_path, sender_email, config_key)
    if not base_html:
        return

    topic = config_name
    sender_name = f"The {topic} NewsBot"
    provider = _env_str("EMAIL_PROVIDER", "smtp").lower()
    subject = _build_subject(topic, analysis_data)

    _dispatch_send(
        provider, subject, base_html, recipient_emails, sender_name, topic,
    )


def _dispatch_send(
    provider: str,
    subject: str,
    base_html: str,
    recipient_emails: list[str],
    sender_name: str,
    topic: str,
) -> None:
    if provider == "resend":
        resend_config = _get_resend_config()
        if resend_config:
            _send_via_resend(
                resend_config,
                subject=subject,
                base_html=base_html,
                recipient_emails=recipient_emails,
                sender_name=sender_name,
            )
        return

    if provider == "emailjs":
        emailjs_config = _get_emailjs_config()
        if emailjs_config:
            _send_via_emailjs(
                emailjs_config,
                subject=subject,
                html_body=base_html,
                recipient_emails=recipient_emails,
                sender_name=sender_name,
                topic=topic,
            )
        return

    if provider != "smtp":
        logger.warning(
            "Unknown EMAIL_PROVIDER '%s', falling back to smtp", provider,
        )
    smtp_config = _get_smtp_config()
    if smtp_config:
        _send_via_smtp(
            smtp_config,
            subject=subject,
            base_html=base_html,
            recipient_emails=recipient_emails,
            sender_name=sender_name,
        )
