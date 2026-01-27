"""
Email Hook - Send analysis reports via email.

Recipients are now managed in the Django database using the Subscriber
model. SMTP configuration is still maintained in .env files.

Configuration (via environment variables in email/*.env files):
---------------------------------------------------------------
EMAIL_ENABLED=true                    # Whether to enable email sending
EMAIL_SMTP_SERVER=smtp.gmail.com      # SMTP server address
EMAIL_SMTP_PORT=587                   # 587 for TLS, 465 for SSL
EMAIL_USE_SSL=false                   # Use SSL instead of TLS
EMAIL_SENDER=your@email.com           # Sender email address
EMAIL_SENDER_NAME=NewsBot             # Display name for sender
EMAIL_PASSWORD=your_app_password      # Email password

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

Example .env file:
-----------------
EMAIL_ENABLED=true
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_SENDER=mybot@gmail.com
EMAIL_SENDER_NAME=My News Bot
EMAIL_PASSWORD=abcd efgh ijkl mnop
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

from css_inline import CSSInliner
from dotenv import load_dotenv

from newsbot.models import AnalysisData
from utilities.django_models import NewsConfig, Subscriber


@dataclass
class SMTPConfig:
    """SMTP configuration."""

    smtp_server: str
    smtp_port: int
    use_ssl: bool
    sender_email: str
    sender_password: str
    cancellation_email: str

logger = logging.getLogger(__name__)


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
            f"Found {len(emails)} recipients for config key '{config_key}'"
            f"{': ' + ', '.join(emails) if emails else ''}",
        )

    except Exception:
        logger.exception(
            f"Error querying database for config key '{config_key}'.",
        )
        return []
    else:
        return emails


def create_mime_text(
    sender_email: str,
    sender_name: str,
    recipient_emails: list[str],
    cancellation_email: str,
    topic: str,
    analysis_data: AnalysisData,
) -> MIMEMultipart:
    """Create the MIME email message."""
    # Create message
    msg = MIMEMultipart()
    # Properly encode sender name with non-ASCII characters (like Ö)
    msg["From"] = formataddr((str(Header(sender_name, "utf-8")), sender_email))
    msg["To"] = msg["From"]  # Send to sender
    msg["Bcc"] = ", ".join(recipient_emails)  # Put recipients in BCC
    msg["List-Unsubscribe"] = (
        f"<mailto:{cancellation_email}?subject=Unsubscribe>"
    )
    msg["List-ID"] = "newsletter.gmx.com"
    msg["Precedence"] = "bulk"

    # Generate subject based on analysis data
    stories_count = analysis_data.get("stories_count", "")
    from_date = analysis_data.get("from_date")
    to_date = analysis_data.get("to_date")
    date_range = ""
    if from_date and to_date:
        date_range = (
            f"{from_date.strftime('%d %b')} - {to_date.strftime('%d %b')}"
        )

    msg["Subject"] = f"{stories_count} Top Stories in {topic} - {date_range}"

    return msg


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
            .values_list(
                "display_name",
                flat=True,
            ),
        )
    except Exception:
        logger.exception("Error querying database for newsletters.")
        return []


def replace_placeholders_in_report(
    report_html: str,
    sender_email: str,
) -> str:
    """
    Replace placeholders in the report HTML with actual values.

    Placeholders:
    - PLACEHOLDER_EMAIL_ADDRESS
    - PLACEHOLDER_NEWSLETTERS
    """
    # Replace sender email placeholder
    report_html = report_html.replace(
        "PLACEHOLDER_EMAIL_ADDRESS",
        sender_email,
    )

    # Get available newsletters from database
    available_newsletters = get_available_newsletters()

    return report_html.replace(
        "PLACEHOLDER_NEWSLETTERS",
        f"{', '.join(available_newsletters)}."
        if available_newsletters
        else "",
    )


def _get_smtp_config() -> SMTPConfig | None:
    """
    Get and validate SMTP configuration from environment.

    Returns:
        Dictionary with SMTP config or None if invalid.

    """
    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    use_ssl = os.getenv("EMAIL_USE_SSL", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")

    if not sender_email or not sender_password:
        logger.warning(
            "Email not configured: EMAIL_SENDER and EMAIL_PASSWORD required",
        )
        return None

    cancellation_email = os.getenv("EMAIL_FOR_CANCELLATION", sender_email)

    return SMTPConfig(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        use_ssl=use_ssl,
        sender_email=sender_email,
        sender_password=sender_password,
        cancellation_email=cancellation_email,
    )


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
    email_receivers_override = analysis_data.get("email_receivers_override")

    if email_receivers_override is not None:
        recipient_emails = email_receivers_override
        if not recipient_emails:
            logger.info(
                "Email will be sent to sender only (no BCC recipients) "
                "via --email-receivers with no arguments",
            )
        else:
            logger.info(
                "Using overridden email recipients: "
                f"{', '.join(recipient_emails)}",
            )
        return recipient_emails

    # Get recipients from Django database based on config_key
    recipient_emails = get_recipients_for_config(config_key)

    if not recipient_emails:
        logger.info(
            f"No active subscribers found for config key '{config_key}'. "
            "Email will be sent to sender only.",
        )

    return recipient_emails


def _process_report_html(report_path: Path, sender_email: str) -> str | None:
    """
    Read and process report HTML file.

    Args:
        report_path: Path to the report file
        sender_email: Sender email for placeholder replacement

    Returns:
        Processed HTML content or None if failed

    """
    try:
        with report_path.open("r", encoding="utf-8") as f:
            report_html = f.read()
    except Exception:
        logger.exception("Failed to read report file.")
        return None

    # Make CSS inline for better email client compatibility
    inliner = CSSInliner()
    report_html = inliner.inline(report_html)

    # Replace placeholders in the report
    return replace_placeholders_in_report(report_html, sender_email)


def _send_email(
    msg: MIMEMultipart,
    smtp_config: SMTPConfig,
    recipient_emails: list[str],
) -> None:
    """
    Send email via SMTP.

    Args:
        msg: MIME message to send
        smtp_config: SMTP configuration dictionary
        recipient_emails: List of recipient emails

    """
    smtp_server = smtp_config.smtp_server
    smtp_port = smtp_config.smtp_port
    use_ssl = smtp_config.use_ssl
    sender_email = smtp_config.sender_email
    sender_password = smtp_config.sender_password

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

        logger.info(
            f"Email sent successfully to {len(recipient_emails)} "
            "recipient(s): "
            f"{', '.join(recipient_emails)}",
        )

    except Exception:
        logger.exception("Failed to send email")


def execute(report_path: Path, analysis_data: AnalysisData) -> None:
    """
    Send the analysis report via email.

    Args:
        report_path: Path to the generated report file
        analysis_data: Dictionary containing analysis metadata

    """
    # Get the email report
    report_path = report_path.parent / "email_reports" / report_path.name

    # Get the config name and key from analysis_data
    config_name = analysis_data.get("config_name", "News")
    config_key = analysis_data.get("config_key", "default")

    # Load .env file for SMTP configuration (still needed for server
    # settings)
    load_dotenv()

    # Check if email is enabled
    if os.getenv("EMAIL_ENABLED", "false").lower() not in ("true", "1", "yes"):
        logger.debug("Email hook disabled (EMAIL_ENABLED not set to true)")
        return

    topic = config_name

    # Get SMTP configuration from environment
    smtp_config = _get_smtp_config()
    if not smtp_config:
        return

    sender_email = smtp_config.sender_email
    sender_name = f"{topic} NewsBot"
    cancellation_email = smtp_config.cancellation_email

    # Get recipient emails
    recipient_emails = _get_recipient_emails(analysis_data, config_key)

    # Create MIME message
    msg = create_mime_text(
        sender_email,
        sender_name,
        recipient_emails,
        cancellation_email,
        topic,
        analysis_data,
    )

    # Process report HTML
    email_body = _process_report_html(report_path, sender_email)
    if not email_body:
        return

    # Attach the email body as HTML
    msg.attach(MIMEText(email_body, "html"))

    # Send email
    _send_email(msg, smtp_config, recipient_emails)
