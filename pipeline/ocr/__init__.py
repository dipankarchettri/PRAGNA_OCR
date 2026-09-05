from .tesseract_engine import (
    is_tesseract_available,
    get_available_languages,
    ocr_image,
    ocr_image_with_layout,
    SUPPORTED_LANGUAGES,
    DEFAULT_PSM,
    DEFAULT_OEM
)

# Surya is optional and lives in its own virtualenv (see surya_engine's module
# docstring). Import it lazily -- importing this package must not require it,
# and surya_engine itself only shells out, so it is cheap to import.
from .surya_engine import (
    is_surya_available,
    ocr_images_with_layout as surya_ocr_images_with_layout
)

OCR_ENGINES = ('tesseract', 'surya')

__all__ = [
    'is_tesseract_available',
    'get_available_languages',
    'ocr_image',
    'ocr_image_with_layout',
    'is_surya_available',
    'surya_ocr_images_with_layout',
    'OCR_ENGINES',
    'SUPPORTED_LANGUAGES',
    'DEFAULT_PSM',
    'DEFAULT_OEM'
]
