"""
Generate the OG social preview image for thenewsbot.com.

Output (final): src/web/newsserver/static/newsserver/og-preview.png
Output (previews): scripts/og-previews/<name>.png

Fonts are downloaded automatically to /tmp if not already present.
Requires: Pillow  (pip install pillow)

Usage:
    # Generate and save the final image:
    python scripts/generate_og_image.py

    # Generate a named preview (saved to scripts/og-previews/):
    python scripts/generate_og_image.py --preview my-test

    # Override gradient parameters:
    python scripts/generate_og_image.py --preview try1 --cx 0.75
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Config

W, H = 1200, 630

# Gradient: ellipsoidal glow, bright at centre fading to dark at edges.
DARK_COLOR = (10, 8, 3)      # near-black olive — outer edges
GLOW_COLOR = (148, 122, 28)  # antique gold    — glow centre

GLOW_CX = 0.78   # centre X as fraction of width  (0 = left, 1 = right)
GLOW_CY = 0.50   # centre Y as fraction of height (0 = top,  1 = bottom)
GLOW_RX = 550    # ellipse radius in pixels (horizontal)
GLOW_RY = 380    # ellipse radius in pixels (vertical)
GLOW_POWER = 1.2    # falloff steepness (higher = sharper visible edge)

# Text
TITLE = "The Newsbot"
DESC_LINE1 = "Curated news digests with sentiment insights, on the"
DESC_LINE2 = "topics you choose, delivered weekly to your inbox."
URL_TEXT = "thenewsbot.com"

# Accent line and description colours (derived from glow by default)
_acc = tuple(min(255, int(c * 1.4)) for c in GLOW_COLOR)
_desc = tuple(min(255, int(c * 1.1 + 30)) for c in GLOW_COLOR)
ACCENT_COLOR = "#{:02x}{:02x}{:02x}".format(*_acc)
DESC_COLOR = "#{:02x}{:02x}{:02x}".format(*_desc)

# Fonts (downloaded to the system temp directory on first run)
FONT_SERIF_URL = (
    "https://fonts.gstatic.com/s/dmseriftext/v12"
    "/FeVQS0BTqb0h60ACL5la2bxii28wYQ.ttf"
)
FONT_SANS_URL = (
    "https://fonts.gstatic.com/s/dmsans/v15"
    "/rP2Hp2ywxg089UriCZOIHTWEBlwu8Q.ttf"
)
_FONT_DIR = Path(tempfile.gettempdir())
FONT_SERIF_PATH = _FONT_DIR / "DMSerifDisplay.ttf"
FONT_SANS_PATH = _FONT_DIR / "DMSans-Regular.ttf"

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
ICON_PATH = REPO_ROOT / "src/web/newsserver/static/newsserver/og-image.png"
OUTPUT = REPO_ROOT / "src/web/newsserver/static/newsserver/og-preview.png"
PREVIEW_DIR = SCRIPT_DIR / "og-previews"

# Fonts


def ensure_font(path: Path, url: str) -> None:
    """Download a font file when it is not already cached locally."""
    if path.exists():
        return

    parsed = urlparse(url)
    if parsed.scheme != "https":
        msg = f"Refusing to download from non-HTTPS URL: {url}"
        raise ValueError(msg)

    print(f"Downloading {path.name}…")
    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    path.write_bytes(response.content)


# Gradient


def make_gradient(
    cx_frac: float,
    cy_frac: float,
    rx: int,
    ry: int,
    power: float,
) -> Image.Image:
    """Build the elliptical gold glow background."""
    cx = int(W * cx_frac)
    cy = int(H * cy_frac)
    img = Image.new("RGB", (W, H))
    px = img.load()
    if px is None:
        msg = "Failed to load pixel data for gradient image"
        raise RuntimeError(msg)
    for y in range(H):
        for x in range(W):
            norm = math.sqrt((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2)
            t = min(norm, 1.0) ** power
            px[x, y] = (
                int(GLOW_COLOR[0] + (DARK_COLOR[0] - GLOW_COLOR[0]) * t),
                int(GLOW_COLOR[1] + (DARK_COLOR[1] - GLOW_COLOR[1]) * t),
                int(GLOW_COLOR[2] + (DARK_COLOR[2] - GLOW_COLOR[2]) * t),
            )
    return img


# Icon


def paste_icon(img: Image.Image) -> None:
    """Paste the inverted white site icon on the right side."""
    icon_size = 340
    icon_src = Image.open(ICON_PATH).convert("RGBA").resize(
        (icon_size, icon_size),
        Image.Resampling.LANCZOS,
    )
    inv = ImageOps.invert(icon_src.convert("L"))
    white_rgba = (255, 255, 255, 255)
    white_icon = Image.new("RGBA", (icon_size, icon_size), white_rgba)
    white_icon.putalpha(inv)
    ix = W - icon_size - 100
    iy = (H - icon_size) // 2
    img.paste(white_icon, (ix, iy), white_icon)


# Text


def draw_text(img: Image.Image) -> None:
    """Render the title, description, accent line, and URL."""
    draw = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(str(FONT_SERIF_PATH), 108)
    f_desc = ImageFont.truetype(str(FONT_SANS_PATH), 26)
    f_url = ImageFont.truetype(str(FONT_SANS_PATH), 22)

    draw.rectangle([60, 185, 200, 191], fill=ACCENT_COLOR)
    draw.text((60, 200), TITLE, font=f_title, fill="white")
    draw.text((60, 348), DESC_LINE1, font=f_desc, fill=DESC_COLOR)
    draw.text((60, 382), DESC_LINE2, font=f_desc, fill=DESC_COLOR)
    draw.text((60, H - 48), URL_TEXT, font=f_url, fill=DESC_COLOR)


# Main


def build(
    cx_frac: float,
    cy_frac: float,
    rx: int,
    ry: int,
    power: float,
) -> Image.Image:
    """Compose the gradient background, icon, and text layers."""
    img = make_gradient(cx_frac, cy_frac, rx, ry, power)
    paste_icon(img)
    draw_text(img)
    return img


def main() -> None:
    """Parse CLI arguments and write the OG preview image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        metavar="NAME",
        help="Save as a named preview instead of the final image",
    )
    parser.add_argument("--cx", type=float, default=GLOW_CX)
    parser.add_argument("--cy", type=float, default=GLOW_CY)
    parser.add_argument("--rx", type=int, default=GLOW_RX)
    parser.add_argument("--ry", type=int, default=GLOW_RY)
    parser.add_argument("--power", type=float, default=GLOW_POWER)
    args = parser.parse_args()

    ensure_font(FONT_SERIF_PATH, FONT_SERIF_URL)
    ensure_font(FONT_SANS_PATH, FONT_SANS_URL)

    img = build(args.cx, args.cy, args.rx, args.ry, args.power)

    if args.preview:
        PREVIEW_DIR.mkdir(exist_ok=True)
        out = PREVIEW_DIR / f"{args.preview}.png"
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        out = OUTPUT

    img.save(out, optimize=True)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
