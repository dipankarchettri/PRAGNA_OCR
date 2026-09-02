"""
Image Preprocessor and Ingestion
Handles loading, auto-orientation, resolution normalization, enhancement,
and validation of image documents.
"""

import os
from typing import Optional
import numpy as np
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


# Skew below this is not worth correcting -- the rotation interpolation
# itself softens glyph edges slightly, so a fractional-degree "correction"
# on an already-straight scan is a net loss, not a gain.
MIN_SKEW_DEGREES = 0.3

# Cap how far we'll ever auto-rotate. A real scan skew is rarely more than a
# couple of degrees; letting the search range go wider risks the projection-
# profile method latching onto a spurious angle on a page that's mostly
# photos/diagrams rather than text rows (e.g. a physics textbook's figures).
MAX_SKEW_DEGREES = 8.0


# Fraction of each edge discarded before scoring a rotation. Rotating with
# expand=False fills the corners with white, and that wedge GROWS with the
# angle -- so it adds row-to-row variance that has nothing to do with text
# alignment, and the score climbs steadily as the angle gets more extreme. On a
# low-contrast page, where the real text signal is weak, that artifact
# dominates and the search runs to the edge of its range. Scoring only the
# common interior area removes the artifact and makes angles comparable.
# sin(8 degrees) is 0.14, so 10% covers the full search range.
_SKEW_SCORE_MARGIN = 0.10


def _otsu_threshold(arr: np.ndarray) -> int:
    """Otsu's method: the grey level that best separates ink from paper."""
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128
    levels = np.arange(256)
    w_bg = np.cumsum(hist)
    w_fg = total - w_bg
    valid = (w_bg > 0) & (w_fg > 0)
    if not valid.any():
        return 128
    sum_all = float((hist * levels).sum())
    sum_bg = np.cumsum(hist * levels)
    mean_bg = np.divide(sum_bg, w_bg, out=np.zeros_like(sum_bg), where=w_bg > 0)
    mean_fg = np.divide(sum_all - sum_bg, w_fg, out=np.zeros_like(sum_bg), where=w_fg > 0)
    between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -1.0
    return int(np.argmax(between))


def _text_row_variance(binary_arr: np.ndarray, angle: float) -> float:
    """
    Rotate a BINARIZED array by `angle` and score how sharply text rows
    separate from the gaps between them: well-aligned text makes the row-wise
    ink counts peak and trough, skewed text smears them together.

    The input must already be binarized. This function's docstring used to say
    "binarized" while every caller passed raw grayscale, and on a low-contrast
    page (one measured here had std 15 and never got brighter than 192) the row
    sums then track paper shading rather than text, leaving no real peak for
    the search to find.
    """
    img = Image.fromarray(binary_arr)
    rotated = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)
    arr = np.asarray(rotated, dtype=np.float64)

    h, w = arr.shape
    my, mx = int(h * _SKEW_SCORE_MARGIN), int(w * _SKEW_SCORE_MARGIN)
    if h - 2 * my > 1 and w - 2 * mx > 1:
        arr = arr[my:h - my, mx:w - mx]

    row_sums = (255.0 - arr).sum(axis=1)
    return float(row_sums.var())


def detect_skew_angle(img: Image.Image) -> float:
    """
    Estimate a scanned page's rotation via the projection-profile method:
    search a small angle range for the rotation that maximizes text-row
    contrast. Works on a small downscaled grayscale copy purely for
    detection speed -- the returned angle is applied to the full-resolution
    image separately. Returns 0.0 if no clearly-better angle is found or if
    the page is too sparse (e.g. mostly blank/graphical) to trust.
    """
    gray = img.convert('L')
    # Downscale for a fast search -- skew angle doesn't need full detail.
    w, h = gray.size
    scale = min(1.0, 1000.0 / max(w, h))
    if scale < 1.0:
        gray = gray.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.uint8)

    # Binarize before searching. The projection profile is meant to count ink
    # per row; on raw grayscale it instead measures paper shading, which on a
    # faded or low-contrast page swamps the text signal entirely.
    threshold = _otsu_threshold(arr)
    binary = np.where(arr > threshold, np.uint8(255), np.uint8(0))

    # Coarse pass across the full allowed range, then a fine pass around the
    # best coarse angle -- cheaper than a single fine-grained sweep.
    coarse_angles = np.arange(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES + 0.01, 0.5)
    best_coarse = max(coarse_angles, key=lambda a: _text_row_variance(binary, a))

    # Clamped: an unclamped fine pass around a boundary coarse angle can return
    # up to MAX_SKEW_DEGREES + 0.5, rotating further than the cap that exists
    # precisely to bound the damage a wrong angle can do.
    lo = max(-MAX_SKEW_DEGREES, best_coarse - 0.5)
    hi = min(MAX_SKEW_DEGREES, best_coarse + 0.5)
    scored = [(a, _text_row_variance(binary, a)) for a in np.arange(lo, hi + 0.01, 0.1)]
    best_angle, best_score = max(scored, key=lambda x: x[1])
    baseline_score = _text_row_variance(binary, 0.0)

    # A best angle sitting on the edge of the search range means the score was
    # still climbing when the range ran out -- i.e. no peak was found and this
    # is not a measurement of skew at all. Observed on a real page: a straight
    # but low-contrast scan scored best at the boundary, got rotated 8.5
    # degrees, and Tesseract went from reading it correctly to returning
    # nothing whatsoever. Declining to rotate is always the safe answer.
    if abs(best_angle) >= MAX_SKEW_DEGREES - 0.05:
        return 0.0

    # Require the corrected angle to meaningfully beat doing nothing --
    # guards against chasing noise on pages with little real text.
    if baseline_score <= 0 or best_score < baseline_score * 1.02:
        return 0.0
    if abs(best_angle) < MIN_SKEW_DEGREES:
        return 0.0
    return round(float(best_angle), 2)


def deskew_image(img: Image.Image) -> Image.Image:
    """
    Detect and correct page rotation. Returns the image unchanged (no
    resize/recompute) when the detected skew is negligible.
    """
    angle = detect_skew_angle(img)
    if angle == 0.0:
        return img
    fill = 'white' if img.mode == 'RGB' else 255
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)


def preprocess_for_ocr(img: Image.Image, enhance_contrast: bool = False, deskew: bool = True) -> Image.Image:
    """
    Shared pre-OCR pipeline applied regardless of source (standalone image
    file or a rasterized PDF page): straighten page rotation, then
    optionally boost contrast for faded scans. Resolution normalization is
    deliberately NOT included here -- a PDF page is already rasterized at
    the caller's chosen DPI, so upscaling doesn't apply the same way it
    does to an arbitrary source image (see load_and_preprocess_image).

    enhance_contrast defaults to False: confirmed on a real document (a
    clean, already-good-contrast modern exam-paper scan) that a blanket
    +20% contrast boost can catastrophically break Tesseract's output --
    coherent, near-perfect OCR degraded to complete word-salad, not a
    minor accuracy hit. The boost was only ever intended to help genuinely
    faded/low-contrast scans; applying it unconditionally to every image
    regardless of whether it's actually faded is what made it dangerous.
    Only pass True for a source already known to be faded.
    """
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    if deskew:
        img = deskew_image(img)

    if enhance_contrast:
        # Subtle contrast boost for faded scanned docs
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

    return img


def load_and_preprocess_image(
    image_path: str,
    enhance_contrast: bool = False,
    upscale: bool = True,
    deskew: bool = True
) -> Image.Image:
    """
    Load an image, fix EXIF orientation, normalize resolution, straighten
    skew, and optionally boost contrast for OCR.
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

    return preprocess_for_ocr(img, enhance_contrast=enhance_contrast, deskew=deskew)
