"""Report HTML processing and placeholder replacement."""

import html
import logging
from pathlib import Path

from css_inline import CSSInliner

from utilities.django_models import NewsConfig

from ._tokens import (
    _get_manage_subscriptions_url,
    _get_unsubscribe_url,
    _get_unsubscribe_url_for_subscriber,
)

logger = logging.getLogger(__name__)


def get_available_newsletters() -> list[str]:
    """Return active newsletter display names, sorted alphabetically."""
    try:
        return list(
            NewsConfig.objects.filter(is_active=True)
            .order_by("display_name")
            .values_list("display_name", flat=True),
        )
    except Exception:
        logger.exception("Error querying database for newsletters.")
        return []


def _replace_global_placeholders(
    report_html: str,
    sender_email: str,
    config_key: str = "default",
    report_name: str = "",
) -> str:
    """Replace global report placeholders with actual values."""
    report_html = report_html.replace(
        "PLACEHOLDER_EMAIL_ADDRESS", sender_email,
    )

    newsletters = get_available_newsletters()
    newsletter_text = f"{', '.join(newsletters)}." if newsletters else ""
    report_html = report_html.replace(
        "PLACEHOLDER_NEWSLETTERS", newsletter_text,
    )

    manage_url = _get_manage_subscriptions_url()
    if manage_url:
        web_report_url = f"{manage_url.rstrip('/')}/config/{config_key}/"
        if report_name:
            web_report_url += f"?report={report_name}"
        web_report_link = web_report_url
    else:
        web_report_link = "#"
    return report_html.replace("PLACEHOLDER_WEB_REPORT_LINK", web_report_link)


def _replace_unsubscribe_placeholder(
    report_html: str,
    sender_email: str,
    subscriber_email: str = "",
    unsubscribe_url: str = "",
) -> str:
    """
    Replace the manage-subscriptions placeholder.

    Offers two options: a link to manage individual subscriptions and a
    one-click link to unsubscribe from all newsletters at once.
    Pass `unsubscribe_url` to reuse a pre-computed URL and skip the
    redundant HMAC computation.
    """
    if not unsubscribe_url:
        unsubscribe_url = (
            _get_unsubscribe_url_for_subscriber(subscriber_email)
            if subscriber_email
            else _get_unsubscribe_url()
        )
    manage_url = _get_manage_subscriptions_url()
    if unsubscribe_url:
        links = []
        if manage_url:
            links.append(
                f'<a href="{html.escape(manage_url, quote=True)}">'
                f"update your preferences</a>",
            )
        links.append(
            f'<a href="{html.escape(unsubscribe_url, quote=True)}">'
            f"unsubscribe from all newsletters</a>",
        )
        manage_link = " or ".join(links)
    else:
        escaped_email = html.escape(sender_email)
        escaped_mailto = html.escape(sender_email, quote=True)
        manage_link = (
            f'contact us at <a href="mailto:{escaped_mailto}">'
            f"{escaped_email}</a> to manage your subscriptions"
        )
    return report_html.replace(
        "PLACEHOLDER_MANAGE_SUBSCRIPTIONS_LINK", manage_link,
    )


def replace_placeholders_in_report(
    report_html: str,
    sender_email: str,
    config_key: str = "default",
    report_name: str = "",
) -> str:
    """Replace all report HTML placeholders with actual values."""
    report_html = _replace_global_placeholders(
        report_html, sender_email, config_key, report_name,
    )
    return _replace_unsubscribe_placeholder(report_html, sender_email)


def _prepare_base_html(
    report_path: Path,
    sender_email: str,
    config_key: str = "default",
) -> str | None:
    """
    Read, inline CSS, and replace non-subscriber placeholders.

    PLACEHOLDER_MANAGE_SUBSCRIPTIONS_LINK is left unreplaced so each
    subscriber receives their own signed unsubscribe URL.
    """
    try:
        report_html = report_path.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to read report file.")
        return None
    report_html = CSSInliner().inline(report_html)
    return _replace_global_placeholders(
        report_html, sender_email, config_key, report_path.name,
    )
