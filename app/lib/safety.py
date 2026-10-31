"""Upload validation (DASHBOARD.md rule 7).

jpg / png / webp only, at most 5 MB, decoded with PIL from memory, never
written to disk or logged.  The returned image is rebuilt from pixel data
only, so EXIF and every other metadata block are dropped.  No URL fetching:
the dashboard makes no runtime network calls.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 40_000_000           # refuse decompression bombs early
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png", "webp")


class UploadError(ValueError):
    """A one-line reason the upload was refused (shown in the UI)."""


def load_upload(data: bytes, filename: str = "") -> Image.Image:
    """Validate and decode an uploaded image; return a metadata-free RGB copy."""
    if len(data) > MAX_BYTES:
        raise UploadError(f"file is {len(data) / 1e6:.1f} MB; the limit is "
                          f"{MAX_BYTES / 1e6:.0f} MB.")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if filename and ext not in ALLOWED_EXTENSIONS:
        raise UploadError(f"'.{ext}' files are not accepted; use jpg, png or webp.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            w, h = probe.size
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise UploadError("the file could not be decoded as an image.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(f"image format {fmt} is not accepted; use jpg, png or webp.")
    if w * h > MAX_PIXELS:
        raise UploadError(f"image is {w}x{h} pixels; the limit is {MAX_PIXELS:,} pixels.")
    with Image.open(io.BytesIO(data)) as img:
        pixels = np.asarray(img.convert("RGB"))
    return Image.fromarray(pixels)          # pixels only: no EXIF, ICC or text chunks
