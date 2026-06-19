"""
Shared palette sampling for the public site and the in-app theme.

Both ``LandingView`` and the ``site_theme`` context processor draw a
colour scheme from the published ``NewsConfig`` rows (with a built-in
fallback for a fresh DB / local dev). This module is the single source
of truth for that query and that fallback list.
"""

from __future__ import annotations

from .models import NewsConfig

# Fallback palettes used when no published NewsConfig exists (fresh DB /
# local dev). The landing page also ships these to the page as the
# server-provided palette set.
FALLBACK_PALETTES: list[dict[str, str]] = [
    {"p": "#5b6ee8", "s": "#8b52d4", "m": "#7060da"},
    {"p": "#0f766e", "s": "#1d4ed8", "m": "#0e7490"},
    {"p": "#b4304a", "s": "#7c2d12", "m": "#9a2f3e"},
    {"p": "#0d9488", "s": "#4d7c0f", "m": "#15803d"},
    {"p": "#6d28d9", "s": "#2563eb", "m": "#4f46e5"},
    {"p": "#4338ca", "s": "#0891b2", "m": "#3730a3"},
]


def published_palettes() -> list[dict[str, str]]:
    """
    Return published NewsConfig palettes, or the fallbacks if none.

    Each palette is ``{"p", "s", "m"}``; ``m`` is the configured middle
    colour or ``""`` when unset, so callers decide how to default it.
    """
    rows = NewsConfig.objects.filter(
        is_active=True,
        published_for_subscription=True,
    ).values(
        "hero_color_primary",
        "hero_color_secondary",
        "hero_color_middle",
    )
    palettes = [
        {
            "p": row["hero_color_primary"],
            "s": row["hero_color_secondary"],
            "m": row["hero_color_middle"] or "",
        }
        for row in rows
    ]
    return palettes or FALLBACK_PALETTES
