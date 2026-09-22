#!/usr/bin/env python3
"""Build the multi-size Mun Cyber Eye icon from the logo PNG.

Customers receive ``app/static/img/mun-cyber-eye.ico`` already built.
Run this from the repo root only when the logo artwork changes:

    python scripts/build_icon.py
"""

from __future__ import annotations

import struct
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
LOGO = ROOT / "app" / "static" / "img" / "mun-cyber-eye-logo.png"
ICO = ROOT / "app" / "static" / "img" / "mun-cyber-eye.ico"
SIZES = (16, 24, 32, 48, 64, 128, 256)
# Console background, so the transparent pupil stays dark on a light desktop.
TILE = (11, 18, 32, 255)


def _eye_crop(src: Image.Image) -> Image.Image:
    """Square crop of the eye, centered in the wide logo."""
    crop = src.convert("RGBA").crop(src.getbbox())
    width, height = crop.size
    side = min(width, height)
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    return crop.crop((left, top, left + side, top + side))


def _render(eye: Image.Image, size: int) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), TILE)
    margin = max(1, round(size * 0.06))
    inner = size - margin * 2
    fitted = eye.resize((inner, inner), Image.Resampling.LANCZOS)
    canvas.paste(fitted, (margin, margin), fitted)
    radius = max(2, round(size * 0.18))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(canvas, (0, 0))
    out.putalpha(mask)
    return out


def _mark_png_entries_readable(path: Path) -> None:
    """Windows shortcuts expect planes=1 and 32-bit entries even for PNG payloads."""
    data = bytearray(path.read_bytes())
    count = struct.unpack_from("<H", data, 4)[0]
    for index in range(count):
        base = 6 + index * 16
        struct.pack_into("<HH", data, base + 4, 1, 32)
    path.write_bytes(data)


def build_icon(logo: Path = LOGO, dest: Path = ICO) -> Path:
    eye = _eye_crop(Image.open(logo))
    icons = [_render(eye, size) for size in SIZES]
    dest.parent.mkdir(parents=True, exist_ok=True)
    icons[-1].save(
        dest,
        format="ICO",
        sizes=[(size, size) for size in SIZES],
        append_images=icons[:-1],
    )
    _mark_png_entries_readable(dest)
    return dest


def main() -> None:
    path = build_icon()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
