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


def _visible(pixel: tuple[int, int, int, int]) -> bool:
    red, green, blue, alpha = pixel
    return alpha > 24 and (red > 16 or green > 16 or blue > 16)


def _mark_bands(src: Image.Image) -> list[tuple[int, int, int, int]]:
    """Vertical bands of the mark as (left, top, right, bottom).

    Gaps of a few pixels (the pupil, or antialiasing) stay inside one band.
    A wordmark under the eye is a separate, shorter band.
    """
    image = src.convert("RGBA")
    pixels = image.load()
    width, height = image.size
    spans: list[tuple[int, int] | None] = []
    for y in range(height):
        left = width
        right = -1
        for x in range(width):
            if _visible(pixels[x, y]):
                if x < left:
                    left = x
                right = x
        spans.append((left, right) if right >= 0 else None)

    bands: list[tuple[int, int, int, int]] = []
    start: int | None = None
    last = 0
    min_x = width
    max_x = 0
    for y, span in enumerate(spans):
        if span is None:
            continue
        if start is None or y - last > 10:
            if start is not None:
                bands.append((min_x, start, max_x + 1, last + 1))
            start = y
            min_x, max_x = span
        else:
            min_x = min(min_x, span[0])
            max_x = max(max_x, span[1])
        last = y
    if start is not None:
        bands.append((min_x, start, max_x + 1, last + 1))
    return bands


def _eye_crop(src: Image.Image) -> Image.Image:
    """Square crop of the eye, centered on the mark.

    The installed logo may include the wordmark. The icon keeps the taller
    eye band so 16px shortcuts stay a recognizable eye.
    """
    bands = _mark_bands(src)
    if not bands:
        crop = src.convert("RGBA").crop(src.getbbox() or (0, 0, *src.size))
    else:
        left, top, right, bottom = max(bands, key=lambda band: (band[3] - band[1], band[2] - band[0]))
        crop = src.convert("RGBA").crop((left, top, right, bottom))
    width, height = crop.size
    side = min(width, height)
    offset_x = max(0, (width - side) // 2)
    offset_y = max(0, (height - side) // 2)
    return crop.crop((offset_x, offset_y, offset_x + side, offset_y + side))


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
