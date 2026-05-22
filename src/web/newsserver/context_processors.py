"""Template context processors for the newsserver app."""

import secrets

from django.http import HttpRequest

from newsbot.color_utils import derive_color_palette

from .models import NewsConfig
from .services.log_service import LogService


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
    cache_control = request.META.get("HTTP_CACHE_CONTROL", "")
    is_refresh = "no-cache" in cache_control

    stored = request.session.get("site_theme")

    if stored and not is_refresh and "middle" in stored:
        primary = stored["primary"]
        secondary = stored["secondary"]
        middle = stored["middle"]
    else:
        configs = list(
            NewsConfig.objects.filter(
                is_active=True,
                published_for_subscription=True,
            ).values(
                "hero_color_primary",
                "hero_color_secondary",
                "hero_color_middle",
            ),
        )
        if configs:
            picked = secrets.choice(configs)
            primary = picked["hero_color_primary"]
            secondary = picked["hero_color_secondary"]
            middle = picked["hero_color_middle"] or ""
        else:
            primary = "#5b6ee8"
            secondary = "#8b52d4"
            middle = ""
        request.session["site_theme"] = {
            "primary": primary,
            "secondary": secondary,
            "middle": middle,
        }

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
