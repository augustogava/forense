#!/usr/bin/env python3
"""
Downloads the canary token preview PNG and writes:
  - Byte-identical PNG copy
  - PNG re-saved with the canary URL in a PNG tEXt chunk (same pixels; file hash may differ)
  - JPEG with the URL in the JPEG COM (comment) segment

Raster formats cannot trigger an HTTP request when opened like an SVG in a browser; the URL is
embedded for metadata / workflows where another system loads it. Use --svg only if you need
that behaviour.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, PngImagePlugin

LOG = logging.getLogger(__name__)

CANARY_DEFAULT = (
    "http://canarytokens.com/about/fton7gb9a0nkp73l7kbf3yelu/preview.png"
)


def fetch_bytes(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; CanaryEmbed/1.0)"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def write_svg(path: Path, url: str, width: int, height: int) -> None:
    safe_url = (
        url.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
    )
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image width="{width}" height="{height}" xlink:href="{safe_url}" href="{safe_url}"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Save canary preview as PNG/JPG with URL embedded in file metadata",
    )
    parser.add_argument(
        "--url",
        default=CANARY_DEFAULT,
        help="Canary preview.png URL",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory",
    )
    parser.add_argument(
        "--png-identical-name",
        default="canary_preview_identical.png",
        help="Byte-identical PNG copy of the download",
    )
    parser.add_argument(
        "--png-meta-name",
        default="canary_preview_url_meta.png",
        help="PNG with same pixels; URL in tEXt chunk (CanaryURL)",
    )
    parser.add_argument(
        "--jpg-name",
        default="canary_preview_url_meta.jpg",
        help="JPEG with URL in COM comment segment",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality 1-95",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also write SVG that loads the remote URL (browser fetch)",
    )
    parser.add_argument(
        "--svg-name",
        default="canary_preview_embed.svg",
        help="SVG output filename when --svg",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_identical = out_dir / args.png_identical_name
    png_meta = out_dir / args.png_meta_name
    jpg_path = out_dir / args.jpg_name

    try:
        data = fetch_bytes(args.url)
    except urllib.error.URLError as e:
        LOG.error("Failed to fetch URL: %s", e)
        return 1
    except TimeoutError:
        LOG.error("Request timed out")
        return 1

    png_identical.write_bytes(data)
    LOG.info("Wrote identical PNG (%d bytes): %s", len(data), png_identical)

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        w, h = img.size
    except Exception as e:
        LOG.error("Could not read image: %s", e)
        return 1

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("CanaryURL", args.url)
    img.save(png_meta, format="PNG", pnginfo=pnginfo, compress_level=6)
    LOG.info("Wrote PNG with tEXt CanaryURL: %s", png_meta)

    if img.mode == "RGB":
        jpg_img = img
    elif img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        base = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        base.paste(rgba, mask=rgba.split()[3])
        jpg_img = base
    else:
        jpg_img = img.convert("RGB")

    q = max(1, min(95, args.jpeg_quality))
    jpg_img.save(
        jpg_path,
        format="JPEG",
        quality=q,
        comment=args.url.encode("utf-8"),
        optimize=True,
    )
    LOG.info("Wrote JPEG with COM comment (URL): %s", jpg_path)

    if args.svg:
        svg_path = out_dir / args.svg_name
        write_svg(svg_path, args.url, w, h)
        LOG.info("Wrote SVG (remote load): %s", svg_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
