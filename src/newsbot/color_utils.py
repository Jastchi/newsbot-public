"""Color derivation utilities for email templates."""

import colorsys

_L_MID = 0.5


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_hls(hex_color: str) -> tuple[float, float, float]:
    r, g, b = _hex_to_rgb(hex_color)
    return colorsys.rgb_to_hls(r / 255, g / 255, b / 255)


def _hls_to_hex(h: float, lightness: float, s: float) -> str:
    lightness = max(0.0, min(1.0, lightness))
    s = max(0.0, min(1.0, s))
    r, g, b = colorsys.hls_to_rgb(h, lightness, s)
    return _rgb_to_hex(round(r * 255), round(g * 255), round(b * 255))


def derive_dark_palette(primary: str, secondary: str) -> dict[str, str]:
    """
    Derive a dark-mode color palette for the admin preview toggle.

    Args:
        primary: Primary hex color.
        secondary: Secondary hex color.

    Returns:
        Dict of CSS color strings for dark-mode overrides.

    """
    p_h, _p_l, p_s = _hex_to_hls(primary)
    s_h, _s_l, s_s = _hex_to_hls(secondary)
    r_p, g_p, b_p = _hex_to_rgb(primary)

    # Scale dark saturation proportionally to input saturation so that
    # near-grey primaries produce neutral dark colours instead of
    # amplifying an unreliable hue to a vivid (e.g. green) tint.
    dk_s = min(p_s * 0.60, 0.20)

    bg          = _hls_to_hex(p_h, 0.08, dk_s)          # near-black page bg
    card        = _hls_to_hex(p_h, 0.13, dk_s * 0.90)   # story card bg
    tint        = _hls_to_hex(p_h, 0.10, dk_s)           # tinted panel bg
    text        = _hls_to_hex(p_h, 0.88, dk_s * 0.90)   # body text
    text_muted  = _hls_to_hex(p_h, 0.58, dk_s * 0.60)   # secondary text
    border      = _hls_to_hex(p_h, 0.20, dk_s * 1.10)   # subtle border
    primary_dk  = _hls_to_hex(p_h, 0.72, p_s * 0.85)  # lightened primary
    muted       = _hls_to_hex(s_h, 0.55, s_s * 0.55)  # muted label
    accent      = _hls_to_hex(s_h, 0.65, s_s * 0.75)  # accent numbers

    return {
        "dk_bg":         bg,
        "dk_card":       card,
        "dk_tint":       tint,
        "dk_text":       text,
        "dk_text_muted": text_muted,
        "dk_border":     border,
        "dk_primary":    primary_dk,
        "dk_muted":      muted,
        "dk_accent":     accent,
        "dk_shadow":     f"rgba({r_p},{g_p},{b_p},0.45)",
        "dk_story_shadow": f"rgba({r_p},{g_p},{b_p},0.30)",
    }


def derive_color_palette(
    primary: str,
    secondary: str,
    middle: str | None = None,
) -> dict[str, str]:
    """
    Derive a full email color palette from brand hex colors.

    Supports two or three input colors.

    Args:
        primary: Primary hex color (gradient start, accents, links).
        secondary: Secondary hex color (gradient end).
        middle: Optional middle hex color for the gradient midpoint.

    Returns:
        Dict of template variable names to CSS color strings.

    """
    p_h, p_l, p_s = _hex_to_hls(primary)
    s_h, s_l, s_s = _hex_to_hls(secondary)
    r_p, g_p, b_p = _hex_to_rgb(primary)

    # Dark shade of primary for headlines (e.g. #312e81 from #5b6ee8)
    headline = _hls_to_hex(p_h, p_l * 0.55, p_s * 0.60)

    # Light/desaturated secondary for muted labels
    muted = _hls_to_hex(s_h, s_l + (1 - s_l) * 0.35, s_s * 0.64)

    # Near-white tint of primary for section backgrounds (e.g. #f7f6fd)
    tint = _hls_to_hex(p_h, 0.97, 0.50)

    # Light secondary tint for borders (e.g. #ece9f8)
    border = _hls_to_hex(s_h, 0.94, 0.50)

    # Slightly darker secondary tint for stronger borders (e.g. #ddd9f8)
    border_dark = _hls_to_hex(s_h, 0.91, 0.50)

    # Medium secondary accent for label numbers and callouts
    accent = _hls_to_hex(s_h, s_l * 1.10, s_s * 0.90)

    # Dark desaturated primary for subdued link text (e.g. #3d3560)
    link_dark = _hls_to_hex(p_h, 0.29, 0.25)

    # Complementary hue for primary button labels — hue rotated 180°,
    # lightness inverted so it always contrasts with the button
    # background.
    comp_h = (p_h + 0.5) % 1.0
    comp_l = 0.92 if p_l <= _L_MID else 0.12
    btn_label = _hls_to_hex(comp_h, comp_l, min(p_s * 0.90, 0.80))

    return {
        "hero_color_primary": primary,
        "hero_color_middle": middle or "",
        "hero_color_secondary": secondary,
        "hero_color_headline": headline,
        "hero_color_muted": muted,
        "hero_color_tint": tint,
        "hero_color_border": border,
        "hero_color_border_dark": border_dark,
        "hero_shadow": f"rgba({r_p},{g_p},{b_p},0.15)",
        "hero_story_shadow": f"rgba({r_p},{g_p},{b_p},0.07)",
        "hero_color_accent": accent,
        "hero_color_link_dark": link_dark,
        "hero_color_btn_label": btn_label,
    }
