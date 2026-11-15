"""Input validation for uploaded images.

The goal is to fail early with a clear message instead of passing garbage to
the model (corrupt files, wrong types, blank images, decompression bombs...).
"""
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_FORMATS,
    MAX_DIMENSION,
    MAX_UPLOAD_BYTES,
    MIN_DIMENSION,
    MIN_PIXEL_STD,
)


class ValidationError(ValueError):
    """Raised when an uploaded file is not a usable image."""


def validate_upload(filename, data: bytes) -> Image.Image:
    """Validate raw upload bytes and return a loaded RGB PIL image."""
    if not filename or not str(filename).strip():
        raise ValidationError("No file selected.")

    ext = str(filename).rsplit(".", 1)[-1].lower() if "." in str(filename) else ""
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValidationError(f"Unsupported file type '.{ext}'. Allowed: {allowed}.")

    if not data:
        raise ValidationError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"File is too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."
        )

    # Verify the bytes really are an image (extension alone can lie).
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError:
        raise ValidationError("Image dimensions are unreasonably large.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ValidationError("File is corrupted or is not a valid image.")

    if img.format not in ALLOWED_FORMATS:
        raise ValidationError(f"Unsupported image format '{img.format}'.")

    width, height = img.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ValidationError(
            f"Image is too small ({width}x{height}). Minimum is "
            f"{MIN_DIMENSION}x{MIN_DIMENSION} pixels."
        )
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ValidationError(
            f"Image is too large ({width}x{height}). Maximum is "
            f"{MAX_DIMENSION}x{MAX_DIMENSION} pixels."
        )

    img = img.convert("RGB")
    if float(np.asarray(img, dtype=np.float32).std()) < MIN_PIXEL_STD:
        raise ValidationError(
            "Image appears blank or uniform. Please upload a microscope image "
            "of a blood cell."
        )
    return img
