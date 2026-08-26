"""
Image Preprocessor and Ingestion
Handles loading, auto-orientation, enhancement, and validation of image documents.
"""

import os
from typing import Optional
from PIL import Image, ImageOps, ImageEnhance

SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}


def is_image_file(filepath: str) -> bool:
    """Check if the given path has a supported image extension."""
    ext = os.path.splitext(filepath.lower())[1]
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def load_and_preprocess_image(
    image_path: str,
    enhance_contrast: bool = False
) -> Image.Image:
    """
    Load an image, fix EXIF orientation, and optionally boost contrast for OCR.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = Image.open(image_path)
    # Fix orientation according to EXIF tag
    img = ImageOps.exif_transpose(img)
    
    # Convert RGBA / P to RGB
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    if enhance_contrast:
        # Subtle contrast boost for faded scanned docs
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

    return img
