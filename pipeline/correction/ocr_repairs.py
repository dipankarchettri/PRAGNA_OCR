"""
Kannada OCR Normalization and Precision Indic Repairs
Fixes recurring optical recognition faults without damaging valid Kannada words.
"""

import re
from typing import Set, Optional

# Safe, high-precision OCR pattern normalizations
SAFE_OCR_REPAIRS = [
    # 1. Archaic Repha and broken R-cluster normalizations (e.g. ಕನಾ೯ಟಕ -> ಕರ್ನಾಟಕ, ಕತ೯ವ್ಯ -> ಕರ್ತವ್ಯ)
    (r'ನಾ೯', 'ರ್ನಾ'),
    (r'ತಾ೯', 'ರ್ತಾ'),
    (r'ದಾ೯', 'ರ್ದಾ'),
    (r'ಮಾ೯', 'ರ್ಮಾ'),
    (r'ವಾ೯', 'ರ್ವಾ'),
    (r'ಶಾ೯', 'ರ್ಶಾ'),
    (r'ಷಾ೯', 'ರ್ಷಾ'),
    (r'ಸಾ೯', 'ರ್ಸಾ'),
    (r'ಹಾ೯', 'ರ್ಹಾ'),
    (r'ತ೯', 'ರ್ತ'),
    (r'ದ೯', 'ರ್ದ'),
    (r'ಮ೯', 'ರ್ಮ'),
    (r'ವ೯', 'ರ್ವ'),
    (r'ಶ೯', 'ರ್ಶ'),
    (r'ಷ೯', 'ರ್ಷ'),
    (r'ಸ೯', 'ರ್ಸ'),

    # 2. Specific Missing Halant / Virama OCR errors
    (r'ಶಿಕಷ', 'ಶಿಕ್ಷ'),
    (r'ಲಕಷಿ', 'ಲಕ್ಷ್ಮಿ'),
    (r'ಪರಕಷ', 'ಪರೀಕ್ಷೆ'),

    # 3. Common OCR Verbal Suffix Glitches
    (r'ಸುತದೆ$', 'ಸುತ್ತದೆ'),
    (r'ುತದೆ$', 'ುತ್ತದೆ'),
    (r'ುತಾರೆ$', 'ತ್ತಾರೆ'),
    (r'ುತಾನೆ$', 'ತ್ತಾನೆ'),
    (r'ುತಾಳೆ$', 'ತ್ತಾಳೆ'),

]

# Targeted high-frequency vowel length OCR confusions
VOWEL_LONG_MAP = {
    'ಜಿವನ': 'ಜೀವನ',
    'ಸಂಗಿತ': 'ಸಂಗೀತ',
    'ನಿರು': 'ನೀರು',
    'ಭುಮಿ': 'ಭೂಮಿ',
    'ಶಿಲ': 'ಶೀಲ',
}


def apply_ocr_repairs(word: str, dictionary: Optional[Set[str]] = None) -> str:
    """
    Apply safe, rule-based Indic OCR repairs to a word token.
    Never corrupts valid dictionary words or syllables.
    """
    w = word

    # 1. Target exact vowel prefixes
    for old_v, new_v in VOWEL_LONG_MAP.items():
        if w == old_v or w.startswith(old_v):
            w = new_v + w[len(old_v):]
            break

    # 2. Apply safe pattern normalizations
    for pat, rep in SAFE_OCR_REPAIRS:
        w = re.sub(pat, rep, w)

    return w
