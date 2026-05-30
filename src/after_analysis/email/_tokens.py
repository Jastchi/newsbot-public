"""Signed unsubscribe token generation and URL helpers."""

import hashlib
import hmac
import urllib.parse

from ._config import _env_str


def _get_manage_subscriptions_url() -> str:
    return _env_str("NEWSSERVER_BASE_URL") or ""


def _get_unsubscribe_url() -> str:
    base = _env_str("NEWSSERVER_BASE_URL")
    return f"{base.rstrip('/')}/unsubscribe/" if base else ""


def _generate_unsubscribe_token(email: str) -> str:
    """Return HMAC-SHA256 token for the email, or '' if no secret."""
    secret = _env_str("UNSUBSCRIBE_TOKEN_SECRET")
    if not secret:
        return ""
    return hmac.new(
        secret.encode(), email.lower().encode(), hashlib.sha256,
    ).hexdigest()


def _get_unsubscribe_url_for_subscriber(email: str) -> str:
    """Return a signed per-subscriber unsubscribe URL, or ''."""
    base = _get_unsubscribe_url()
    if not base:
        return ""
    params: dict[str, str] = {"email": email}
    token = _generate_unsubscribe_token(email)
    if token:
        params["token"] = token
    return f"{base}?{urllib.parse.urlencode(params)}"
