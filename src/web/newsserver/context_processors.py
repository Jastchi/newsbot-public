"""Template context processors for the newsserver app."""

import re
import secrets

from django.http import HttpRequest

from newsbot.color_utils import derive_color_palette

from .palettes import published_palettes
from .services.log_service import LogService
from .site_urls import build_canonical_page_url, site_origin

_HEX6 = re.compile(r"[0-9a-fA-F]{6}")


def _query_hex(request: HttpRequest, key: str) -> str:
    """Return a hex colour from a query param, or '' if invalid."""
    value = request.GET.get(key, "")
    return f"#{value}" if _HEX6.fullmatch(value) else ""


def _theme_context(
    primary: str,
    secondary: str,
    middle: str,
) -> dict[str, str]:
    """Build the CSS-variable context for a given palette."""
    palette = derive_color_palette(primary, secondary, middle or None)
    return {
        "site_primary": primary,
        "site_secondary": secondary,
        "site_middle": middle,
        "site_tint": palette["hero_color_tint"],
        "site_border": palette["hero_color_border"],
        "site_shadow": palette["hero_shadow"],
        "site_muted": palette["hero_color_muted"],
    }


def canonical_urls(request: HttpRequest) -> dict[str, str]:
    """Expose canonical page URL and site origin for meta tags."""
    return {
        "canonical_url": build_canonical_page_url(request),
        "site_origin": site_origin(request),
    }


def has_logs(_request: HttpRequest) -> dict[str, bool]:
    """Add has_logs to template context: True if any log files exist."""
    logs = LogService.get_log_files()
    return {"has_logs": bool(logs)}


def site_theme(request: HttpRequest) -> dict[str, str]:
    """
    Expose a NewsConfig colour palette as CSS variables.

    The chosen config is stored in the session so the colour stays
    consistent while navigating between tabs.  A hard browser refresh
    (Cache-Control: no-cache) picks a new random config.
    """
    # The public landing page supplies its own palette (LandingView)
    # and never reads the site_* vars, so skip the sampling, the DB
    # query, and the session write here — the latter would otherwise
    # force a Set-Cookie on an otherwise-cacheable public page.
    match = request.resolver_match
    if match is not None and match.url_name == "landing":
        return {}

    # An explicit palette can arrive via query params (e.g. the
    # landing page's "Explore the app" link). Honour it and persist
    # it to the session so the app keeps the same colour scheme the
    # visitor saw on the landing page.
    query_primary = _query_hex(request, "tp")
    query_secondary = _query_hex(request, "ts")
    if query_primary and query_secondary:
        query_middle = _query_hex(request, "tm")
        theme = {
            "primary": query_primary,
            "secondary": query_secondary,
            "middle": query_middle,
        }
        # Only touch the session when the palette actually changes, so a
        # bookmarked/refreshed themed URL doesn't write the session (and
        # hit the session store) on every request.
        if request.session.get("site_theme") != theme:
            request.session["site_theme"] = theme
        return _theme_context(query_primary, query_secondary, query_middle)

    cache_control = request.META.get("HTTP_CACHE_CONTROL", "")
    is_refresh = "no-cache" in cache_control

    stored = request.session.get("site_theme")

    if stored and not is_refresh and "middle" in stored:
        primary = stored["primary"]
        secondary = stored["secondary"]
        middle = stored["middle"]
    else:
        chosen = secrets.choice(published_palettes())
        primary = chosen["p"]
        secondary = chosen["s"]
        middle = chosen["m"]
        request.session["site_theme"] = {
            "primary": primary,
            "secondary": secondary,
            "middle": middle,
        }

    return _theme_context(primary, secondary, middle)
