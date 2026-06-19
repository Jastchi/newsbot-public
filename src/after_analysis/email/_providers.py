"""Email sending implementations for SMTP, Resend, and EmailJS."""

import logging
import smtplib
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

import requests

from utilities.email import send_via_resend

from ._config import EmailJSConfig, ResendConfig, SMTPConfig
from ._report import _replace_unsubscribe_placeholder
from ._tokens import _get_unsubscribe_url_for_subscriber

EMAILJS_SEND_URL = "https://api.emailjs.com/api/v1.0/email/send"
HTTP_FORBIDDEN = 403

logger = logging.getLogger(__name__)


def _build_list_unsubscribe_headers(
    cancellation_email: str,
    unsubscribe_url: str,
) -> dict[str, str]:
    parts = [f"<mailto:{cancellation_email}?subject=Unsubscribe>"]
    if unsubscribe_url:
        parts.append(f"<{unsubscribe_url}>")
    headers: dict[str, str] = {
        "List-Unsubscribe": ", ".join(parts),
        "List-ID": "newsletter.thenewsbot.net",
        "Precedence": "bulk",
    }
    if unsubscribe_url:
        headers["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    return headers


def _build_smtp_message(
    sender_email: str,
    sender_name: str,
    recipient_email: str,
    cancellation_email: str,
    subject: str,
    unsubscribe_url: str,
) -> MIMEMultipart:
    """Build a MIME message addressed to a single recipient."""
    msg = MIMEMultipart()
    msg["From"] = formataddr((str(Header(sender_name, "utf-8")), sender_email))
    msg["To"] = recipient_email
    for key, value in _build_list_unsubscribe_headers(
        cancellation_email, unsubscribe_url,
    ).items():
        msg[key] = value
    msg["Subject"] = subject
    return msg


def _send_via_smtp(
    smtp_config: SMTPConfig,
    subject: str,
    base_html: str,
    recipient_emails: set[str],
    sender_name: str,
) -> None:
    """Send one individual email per subscriber via SMTP."""
    server_cls = smtplib.SMTP_SSL if smtp_config.use_ssl else smtplib.SMTP
    try:
        with server_cls(
            smtp_config.smtp_server, smtp_config.smtp_port,
        ) as server:
            if not smtp_config.use_ssl:
                server.starttls()
            server.login(smtp_config.login_email, smtp_config.sender_password)
            sent = 0
            for subscriber_email in recipient_emails:
                unsubscribe_url = _get_unsubscribe_url_for_subscriber(
                    subscriber_email,
                )
                html = _replace_unsubscribe_placeholder(
                    base_html,
                    smtp_config.sender_email,
                    subscriber_email,
                    unsubscribe_url,
                )
                msg = _build_smtp_message(
                    smtp_config.sender_email,
                    sender_name,
                    subscriber_email,
                    smtp_config.cancellation_email,
                    subject,
                    unsubscribe_url,
                )
                msg.attach(MIMEText(html, "html"))
                try:
                    server.send_message(msg)
                    sent += 1
                except Exception:
                    logger.exception(
                        "Failed to send email to %s", subscriber_email,
                    )
            logger.info(
                "Email sent to %d/%d recipient(s): %s",
                sent,
                len(recipient_emails),
                ", ".join(recipient_emails),
            )
    except Exception:
        logger.exception("Failed to connect or authenticate with SMTP server")


def _send_via_resend(
    resend_config: ResendConfig,
    subject: str,
    base_html: str,
    recipient_emails: set[str],
    sender_name: str,
) -> None:
    """Send one individual email per subscriber via Resend."""
    from_header = formataddr(
        (str(Header(sender_name, "utf-8")), resend_config.sender_email),
    )
    sent = 0
    for subscriber_email in recipient_emails:
        unsubscribe_url = _get_unsubscribe_url_for_subscriber(subscriber_email)
        html = _replace_unsubscribe_placeholder(
            base_html,
            resend_config.sender_email,
            subscriber_email,
            unsubscribe_url,
        )
        headers = _build_list_unsubscribe_headers(
            resend_config.cancellation_email, unsubscribe_url,
        )
        try:
            send_via_resend(
                resend_config.api_key,
                from_header,
                [subscriber_email],
                subject,
                html=html,
                headers=headers,
            )
            sent += 1
        except Exception:
            logger.exception("Resend send failed for %s", subscriber_email)
    logger.info(
        "Email sent via Resend to %d/%d recipient(s): %s",
        sent,
        len(recipient_emails),
        ", ".join(recipient_emails),
    )


def _send_via_emailjs(
    emailjs_config: EmailJSConfig,
    subject: str,
    html_body: str,
    recipient_emails: set[str],
    sender_name: str,
    topic: str,
) -> None:
    """Send via EmailJS REST API (BCC bulk pattern)."""
    from_header = formataddr((sender_name, emailjs_config.sender_email))
    bcc_list = [
        e for e in recipient_emails if e != emailjs_config.sender_email
    ]
    html_body = _replace_unsubscribe_placeholder(
        html_body, emailjs_config.sender_email,
    )
    payload: dict = {
        "service_id": emailjs_config.service_id,
        "template_id": emailjs_config.template_id,
        "user_id": emailjs_config.user_id,
        "template_params": {
            "subject": subject,
            "content": html_body,
            "sender_name": sender_name,
            "from_name": sender_name,
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
        "EmailJS API error %s: %s", resp.status_code, body or "(no body)",
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
            "403 from EmailJS often means the private key is required. "
            "Set EMAILJS_PRIVATE_KEY in .env",
        )
