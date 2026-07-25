"""Signed unsubscribe token generation and URL helpers."""

import hashlib
import hmac
import urllib.parse

from ._config import _env_str


def _get_app_base_url() -> str:
    """
    Return the base URL of the host that serves the Django app.

    In a subdomain split the app (and routes like ``/unsubscribe/``)
    lives on ``APP_HOST``, while ``NEWSSERVER_BASE_URL`` points at the
    canonical/marketing host, which serves only the landing page and
    would 404 on app routes. Prefer ``APP_HOST`` when set; otherwise
    fall back to ``NEWSSERVER_BASE_URL`` (single-host deployments).
    """
    app_host = _env_str("APP_HOST")
    if app_host:
        return f"https://{app_host}"
    return _env_str("NEWSSERVER_BASE_URL")


def _get_manage_subscriptions_url() -> str:
    return _get_app_base_url() or ""


def _get_unsubscribe_url() -> str:
    base = _get_app_base_url()
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
