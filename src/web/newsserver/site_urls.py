"""Canonical site URL helpers for SEO and outbound links."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings

if TYPE_CHECKING:
    from django.http import HttpRequest


def normalize_site_domain(raw: str) -> str:
    """Return a bare hostname without scheme, www, path, or slash."""
    domain = raw.strip().lower()
    for prefix in ("https://", "http://"):
        if domain.startswith(prefix):
            domain = domain[len(prefix) :]
            break
    domain = domain.split("/", 1)[0]
    domain = domain.removeprefix("www.")
    return domain.rstrip(".")


def canonical_site_url_from_env(
    site_domain: str = "",
    news_server_base_url: str = "",
) -> str:
    """Build https:// canonical base URL from env-style settings."""
    domain = normalize_site_domain(site_domain)
    if not domain and news_server_base_url:
        domain = normalize_site_domain(news_server_base_url)
    return f"https://{domain}" if domain else ""


def request_host(request: HttpRequest) -> str:
    """Return the request hostname without port."""
    return request.get_host().split(":", 1)[0].lower()


def raw_request_host(request: HttpRequest) -> str:
    """Return HTTP_HOST without validation (for redirect middleware)."""
    host = request.META.get("HTTP_HOST", "")
    return host.split(":", 1)[0].lower() if host else ""


def is_canonical_host(request: HttpRequest) -> bool:
    """Return whether the request uses the configured canonical host."""
    canonical = settings.CANONICAL_SITE_URL
    if not canonical:
        return True
    return request_host(request) == settings.SITE_DOMAIN


def _canonical_path_prefix() -> str:
    return f"{settings.CANONICAL_SITE_URL}{settings.FORCE_SCRIPT_NAME}"


def build_canonical_absolute_uri(
    request: HttpRequest,
    path: str,
) -> str:
    """Build an absolute URL using the configured canonical site."""
    if settings.CANONICAL_SITE_URL:
        return f"{_canonical_path_prefix()}{path}"
    if request.get_host():
        return request.build_absolute_uri(path)
    return path


def build_canonical_page_url(request: HttpRequest) -> str:
    """Return the canonical URL for the current page."""
    if settings.CANONICAL_SITE_URL:
        url = f"{_canonical_path_prefix()}{request.path}"
        query = request.META.get("QUERY_STRING", "")
        if query:
            url = f"{url}?{query}"
        return url
    return request.build_absolute_uri()


def site_origin(request: HttpRequest) -> str:
    """Return site origin for absolute asset URLs in meta tags."""
    if settings.CANONICAL_SITE_URL:
        return settings.CANONICAL_SITE_URL
    return f"{request.scheme}://{request.get_host()}"
