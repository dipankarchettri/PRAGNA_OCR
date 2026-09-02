"""
Image Preprocessor and Ingestion
Handles loading, auto-orientation, resolution normalization, enhancement,
and validation of image documents.
"""

import os
from typing import Optional
from PIL import Image, ImageOps, ImageEnhance

SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}

# Tesseract's LSTM engine normalizes each text line to a fixed height, so it needs
# roughly 300 DPI to resolve detail. Kannada is far more sensitive to this than Latin:
# ottakshara (subscript conjuncts) sit below the base consonant and collapse into a
# smudge at low resolution, which no amount of dictionary correction can recover.
# A typical book page scanned at 300 DPI lands around 2400px on its long edge.
TARGET_LONG_EDGE = 2400

# Never blow an image up more than this. Beyond ~4x we are inventing pixels, not
# recovering detail, and memory/OCR time grow quadratically.
MAX_UPSCALE_FACTOR = 4.0


def is_image_file(filepath: str) -> bool:
    """Check if the given path has a supported image extension."""
    ext = os.path.splitext(filepath.lower())[1]
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def normalize_resolution(
    img: Image.Image,
    target_long_edge: int = TARGET_LONG_EDGE,
    max_scale: float = MAX_UPSCALE_FACTOR
) -> Image.Image:
    """
    Upscale a low-resolution page image so Tesseract sees enough detail to resolve
    Kannada conjuncts. Only ever scales *up* -- images already at or above the target
    are returned untouched, so this is safe to apply to high-DPI PDF rasterizations.

    Returns the image unchanged when no scaling is warranted.
    """
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= 0 or long_edge >= target_long_edge:
        return img

    scale = min(target_long_edge / long_edge, max_scale)
    if scale <= 1.01:
        return img

    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    # LANCZOS keeps glyph edges as clean as resampling allows; bicubic visibly softens
    # the thin strokes that distinguish similar Kannada consonants.
    return img.resize(new_size, Image.LANCZOS)


def load_and_preprocess_image(
    image_path: str,
    enhance_contrast: bool = False,
    upscale: bool = True
) -> Image.Image:
    """
    Load an image, fix EXIF orientation, normalize resolution, and optionally
    boost contrast for OCR.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = Image.open(image_path)
    # Fix orientation according to EXIF tag
    img = ImageOps.exif_transpose(img)

    # Convert RGBA / P to RGB
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    if upscale:
        img = normalize_resolution(img)

    if enhance_contrast:
        # Subtle contrast boost for faded scanned docs
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

    return img
