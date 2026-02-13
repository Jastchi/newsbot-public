"""
Email Hook - Send analysis reports via email.

Recipients are now managed in the Django database using the Subscriber
model. You can send via SMTP (Gmail, etc.) or via EmailJS.

Configuration (via environment variables in email/*.env files):
---------------------------------------------------------------
EMAIL_ENABLED=true                    # Whether to enable email sending
EMAIL_PROVIDER=smtp                   # "smtp" or "emailjs"

SMTP (when EMAIL_PROVIDER=smtp):
  EMAIL_SMTP_SERVER=smtp.gmail.com    # SMTP server address
  EMAIL_SMTP_PORT=587                 # 587 for TLS, 465 for SSL
  EMAIL_USE_SSL=false                 # Use SSL instead of TLS
  EMAIL_SENDER=your@email.com         # Sender email address
  EMAIL_SENDER_NAME=NewsBot           # Display name for sender
  EMAIL_PASSWORD=your_app_password    # Email password

EmailJS (when EMAIL_PROVIDER=emailjs):
  EMAILJS_SERVICE_ID=your_service_id  # From EmailJS dashboard
  EMAILJS_TEMPLATE_ID=your_template   # template id
  EMAILJS_USER_ID=your_public_key     # Public key
  EMAILJS_PRIVATE_KEY=...             # Optional; for server-side auth
  EMAIL_SENDER=your@email.com         # Used as From / reply identity

EmailJS template: {{subject}}, {{{content}}}, {{to_email}}, {{bcc}},
{{from_name}}, {{from_header}}, {{from_email}}, {{sender_name}},
{{topic}}. Note: With personal email services (e.g. GMX), EmailJS often
does not
show the From display name. For "NewsBot" as sender, use SMTP instead
(EMAIL_PROVIDER=smtp with EMAIL_SENDER_NAME=NewsBot).

Recipient Management:
--------------------
Recipients are managed through the Django admin interface at
/admin/newsserver/subscriber/
Each recipient can subscribe to multiple news configs via the
NewsConfig model.

Command Line Override:
---------------------
Use --email-receivers to override database recipients:
- With addresses: sends to sender (To) and specified addresses (BCC)
- Without addresses: sends to sender only (no BCC recipients)

Example .env (SMTP):
-------------------
EMAIL_ENABLED=true
EMAIL_PROVIDER=smtp
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_SENDER=mybot@gmail.com
EMAIL_SENDER_NAME=My News Bot
EMAIL_PASSWORD=abcd efgh ijkl mnop

Example .env (EmailJS):
----------------------
EMAIL_ENABLED=true
EMAIL_PROVIDER=emailjs
EMAILJS_SERVICE_ID=service_xxx
EMAILJS_TEMPLATE_ID=template_xxx
EMAILJS_USER_ID=user_xxx
EMAILJS_PRIVATE_KEY=your_private_key
EMAIL_SENDER=mybot@gmail.com
"""

import logging
import os
import smtplib
from dataclasses import dataclass
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

import requests
from css_inline import CSSInliner
from dotenv import load_dotenv

from newsbot.models import AnalysisData
from utilities.django_models import NewsConfig, Subscriber

EMAILJS_SEND_URL = "https://api.emailjs.com/api/v1.0/email/send"
HTTP_FORBIDDEN = 403
_TRUTHY_VALUES = frozenset({"true", "1", "yes"})

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _env_is_truthy(key: str, default: str = "false") -> bool:
    """Return True when the environment variable value is truthy."""
    return os.getenv(key, default).lower().strip() in _TRUTHY_VALUES


def _env_str(key: str, default: str = "") -> str:
    """Return a stripped environment variable value."""
    return os.getenv(key, default).strip()


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass
class SMTPConfig:
    """SMTP configuration."""

    smtp_server: str
    smtp_port: int
    use_ssl: bool
    sender_email: str
    sender_password: str
    cancellation_email: str


@dataclass
class EmailJSConfig:
    """EmailJS configuration for sending via EmailJS API."""

    service_id: str
    template_id: str
    user_id: str
    private_key: str | None
    sender_email: str


# ----------------------------------------------------------------------
# Recipients & subject
# ----------------------------------------------------------------------


def get_recipients_for_config(config_key: str) -> list[str]:
    """
    Get email addresses of active subscribers for a specific config.

    Args:
        config_key: The config key (e.g., "technology")

    Returns:
        List of email addresses

    """
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
            "Error querying database for config key '%s'.",
            config_key,
        )
        return []
    else:
        return emails


def _build_subject(topic: str, analysis_data: AnalysisData) -> str:
    """Build email subject from analysis data and topic."""
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
    """
    Get recipient emails from override or database.

    Args:
        analysis_data: Analysis data dictionary
        config_key: Config key for database lookup

    Returns:
        List of recipient email addresses.

    """
    override = analysis_data.get("email_receivers_override")

    if override is not None:
        if override:
            logger.info(
                "Using overridden email recipients: %s",
                ", ".join(override),
            )
        else:
            logger.info(
                "Email will be sent to sender only (no BCC recipients) "
                "via --email-receivers with no arguments",
            )
        return override

    recipients = get_recipients_for_config(config_key)
    if not recipients:
        logger.info(
            "No active subscribers for config key '%s'. "
            "Email will be sent to sender only.",
            config_key,
        )
    return recipients


# ----------------------------------------------------------------------
# Report processing
# ----------------------------------------------------------------------


def get_available_newsletters() -> list[str]:
    """
    Get display names of all active newsletters from the database.

    Returns:
        List of active newsletter display names sorted alphabetically

    """
    try:
        return list(
            NewsConfig.objects.filter(is_active=True)
            .order_by("display_name")
            .values_list("display_name", flat=True),
        )
    except Exception:
        logger.exception("Error querying database for newsletters.")
        return []


def _get_manage_subscriptions_url() -> str:
    """News-schedule page URL, or empty if NEWSSERVER_BASE_URL unset."""
    return _env_str("NEWSSERVER_BASE_URL") or ""


def replace_placeholders_in_report(
    report_html: str,
    sender_email: str,
) -> str:
    """
    Replace placeholders in the report HTML with actual values.

    Placeholders:
    - PLACEHOLDER_EMAIL_ADDRESS
    - PLACEHOLDER_NEWSLETTERS
    - PLACEHOLDER_MANAGE_SUBSCRIPTIONS_LINK (manage/cancel link if
      NEWSSERVER_BASE_URL is set)
    """
    report_html = report_html.replace(
        "PLACEHOLDER_EMAIL_ADDRESS",
        sender_email,
    )

    newsletters = get_available_newsletters()
    newsletter_text = f"{', '.join(newsletters)}." if newsletters else ""
    report_html = report_html.replace(
        "PLACEHOLDER_NEWSLETTERS", newsletter_text,
    )

    manage_url = _get_manage_subscriptions_url()
    if manage_url:
        manage_link = f'<a href="{manage_url}">click here</a>'
    else:
        manage_link = (
            f'contact us at <a href="mailto:{sender_email}">'
            f"{sender_email}</a> to manage your subscriptions"
        )
    placeholder = "PLACEHOLDER_MANAGE_SUBSCRIPTIONS_LINK"
    return report_html.replace(placeholder, manage_link)


def _process_report_html(report_path: Path, sender_email: str) -> str | None:
    """
    Read, inline CSS, and replace placeholders in a report HTML file.

    Args:
        report_path: Path to the report file
        sender_email: Sender email for placeholder replacement

    Returns:
        Processed HTML content or None if the file could not be read.

    """
    try:
        report_html = report_path.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to read report file.")
        return None

    report_html = CSSInliner().inline(report_html)
    return replace_placeholders_in_report(report_html, sender_email)


# ----------------------------------------------------------------------
# MIME helpers
# ----------------------------------------------------------------------


def create_mime_text(
    sender_email: str,
    sender_name: str,
    recipient_emails: list[str],
    cancellation_email: str,
    topic: str,
    analysis_data: AnalysisData,
) -> MIMEMultipart:
    """Create the MIME email message."""
    msg = MIMEMultipart()
    msg["From"] = formataddr((str(Header(sender_name, "utf-8")), sender_email))
    msg["To"] = msg["From"]
    msg["Bcc"] = ", ".join(recipient_emails)
    list_unsubscribe_parts = [
        f"<mailto:{cancellation_email}?subject=Unsubscribe>",
    ]
    manage_url = _get_manage_subscriptions_url()
    if manage_url:
        list_unsubscribe_parts.append(f"<{manage_url}>")
    msg["List-Unsubscribe"] = ", ".join(list_unsubscribe_parts)
    msg["List-ID"] = "newsletter.gmx.com"
    msg["Precedence"] = "bulk"
    msg["Subject"] = _build_subject(topic, analysis_data)
    return msg


# ----------------------------------------------------------------------
# Provider configuration loaders
# ----------------------------------------------------------------------


def _get_smtp_config() -> SMTPConfig | None:
    """
    Get and validate SMTP configuration from environment.

    Returns:
        SMTPConfig if all required vars are set, else None.

    """
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")

    if not sender_email or not sender_password:
        logger.warning(
            "Email not configured: EMAIL_SENDER and EMAIL_PASSWORD required",
        )
        return None

    return SMTPConfig(
        smtp_server=os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
        smtp_port=int(os.getenv("EMAIL_SMTP_PORT", "587")),
        use_ssl=_env_is_truthy("EMAIL_USE_SSL"),
        sender_email=sender_email,
        sender_password=sender_password,
        cancellation_email=os.getenv("EMAIL_FOR_CANCELLATION", sender_email),
    )


def _get_emailjs_config() -> EmailJSConfig | None:
    """
    Get and validate EmailJS configuration from environment.

    Returns:
        EmailJSConfig if all required vars are set, else None.

    """
    service_id = _env_str("EMAILJS_SERVICE_ID")
    template_id = _env_str("EMAILJS_TEMPLATE_ID")
    user_id = _env_str("EMAILJS_USER_ID")
    sender_email = _env_str("EMAIL_SENDER")

    if not all((service_id, template_id, user_id, sender_email)):
        logger.warning(
            "EmailJS not configured: EMAIL_PROVIDER=emailjs requires "
            "EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, EMAILJS_USER_ID, "
            "EMAIL_SENDER",
        )
        return None

    return EmailJSConfig(
        service_id=service_id,
        template_id=template_id,
        user_id=user_id,
        private_key=_env_str("EMAILJS_PRIVATE_KEY") or None,
        sender_email=sender_email,
    )


# ----------------------------------------------------------------------
# Sending
# ----------------------------------------------------------------------


def _send_via_smtp(
    msg: MIMEMultipart,
    smtp_config: SMTPConfig,
    recipient_emails: list[str],
) -> None:
    """
    Send email via SMTP.

    Args:
        msg: MIME message to send
        smtp_config: SMTP configuration
        recipient_emails: List of recipient emails (for logging)

    """
    server_cls = smtplib.SMTP_SSL if smtp_config.use_ssl else smtplib.SMTP

    try:
        with server_cls(
            smtp_config.smtp_server,
            smtp_config.smtp_port,
        ) as server:
            if not smtp_config.use_ssl:
                server.starttls()
            server.login(smtp_config.sender_email, smtp_config.sender_password)
            server.send_message(msg)

        logger.info(
            "Email sent successfully to %d recipient(s): %s",
            len(recipient_emails),
            ", ".join(recipient_emails),
        )
    except Exception:
        logger.exception("Failed to send email")


def _send_via_emailjs(
    emailjs_config: EmailJSConfig,
    subject: str,
    html_body: str,
    recipient_emails: list[str],
    sender_name: str,
    topic: str,
) -> None:
    """
    Send email via EmailJS REST API.

    One email: To = sender, BCC = recipients (matches SMTP bulk).
    Template params: subject, content, to_email, bcc, from_name,
    from_email, from_header (RFC "Name <email>"), sender_name, topic.

    """
    from_name = (
        sender_name or _env_str("EMAIL_SENDER_NAME", "NewsBot") or "NewsBot"
    )
    from_header = formataddr((from_name, emailjs_config.sender_email))

    bcc_list = [
        e for e in recipient_emails if e != emailjs_config.sender_email
    ]

    payload: dict = {
        "service_id": emailjs_config.service_id,
        "template_id": emailjs_config.template_id,
        "user_id": emailjs_config.user_id,
        "template_params": {
            "subject": subject,
            "content": html_body,
            "sender_name": sender_name,
            "from_name": from_name,
            "from_email": emailjs_config.sender_email,
            "from_header": from_header,
            "topic": topic,
            "to_email": emailjs_config.sender_email,
            "bcc": ", ".join(bcc_list),
        },
    }
    if emailjs_config.private_key:
        payload["accessToken"] = emailjs_config.private_key

    try:
        resp = requests.post(
            EMAILJS_SEND_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        _log_emailjs_errors(resp, emailjs_config)
        resp.raise_for_status()

        logger.info(
            "Email sent via EmailJS to %d recipient(s): To=%s, BCC=%s",
            1 + len(bcc_list),
            emailjs_config.sender_email,
            ", ".join(bcc_list) or "(none)",
        )
    except requests.RequestException:
        logger.exception("EmailJS send failed")
        raise


def _log_emailjs_errors(
    resp: requests.Response,
    emailjs_config: EmailJSConfig,
) -> None:
    """Log diagnostic hints when the EmailJS API returns an error."""
    if resp.ok:
        return

    body = resp.text or ""
    logger.error(
        "EmailJS API error %s: %s",
        resp.status_code,
        body or "(no body)",
    )

    if resp.status_code != HTTP_FORBIDDEN:
        return

    if "non-browser" in body.lower():
        logger.warning(
            "Enable server-side API in EmailJS: Account > Security > "
            "allow non-browser applications (dashboard.emailjs.com)",
        )
    elif not emailjs_config.private_key:
        logger.warning(
            "403 from EmailJS often means the private key is required "
            "for server-side sends. Set EMAILJS_PRIVATE_KEY in .env",
        )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def execute(report_path: Path, analysis_data: AnalysisData) -> None:
    """
    Send the analysis report via email.

    Args:
        report_path: Path to the generated report file
        analysis_data: Dictionary containing analysis metadata

    """
    report_path = report_path.parent / "email_reports" / report_path.name

    config_name = analysis_data.get("config_name", "News")
    config_key = analysis_data.get("config_key", "default")

    load_dotenv()

    if not _env_is_truthy("EMAIL_ENABLED"):
        logger.debug("Email hook disabled (EMAIL_ENABLED not set to true)")
        return

    topic = config_name
    sender_name = f"The {topic} NewsBot"

    recipient_emails = _get_recipient_emails(analysis_data, config_key)

    sender_email = _env_str("EMAIL_SENDER")
    if not sender_email:
        logger.warning("EMAIL_SENDER required for email report placeholders")
        return

    email_body = _process_report_html(report_path, sender_email)
    if not email_body:
        return

    provider = _env_str("EMAIL_PROVIDER", "smtp").lower()

    if provider == "emailjs":
        emailjs_config = _get_emailjs_config()
        if not emailjs_config:
            return
        _send_via_emailjs(
            emailjs_config,
            subject=_build_subject(topic, analysis_data),
            html_body=email_body,
            recipient_emails=recipient_emails,
            sender_name=sender_name,
            topic=topic,
        )
        return

    smtp_config = _get_smtp_config()
    if not smtp_config:
        return

    msg = create_mime_text(
        smtp_config.sender_email,
        sender_name,
        recipient_emails,
        smtp_config.cancellation_email,
        topic,
        analysis_data,
    )
    msg.attach(MIMEText(email_body, "html"))
    _send_via_smtp(msg, smtp_config, recipient_emails)
