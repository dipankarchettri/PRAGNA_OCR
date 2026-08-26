"""
Kannada OCR Normalization and Indic Repairs
Fixes recurring optical recognition faults:
- Missing Virama/Halant (U+0CCD) conjunct corruptions
- Consonant and ottu collapsing
- Short/long vowel optical confusion
- Corrupted prefix forms (e.g., initial ಪ -> ಪ್ರ)
"""

import re
from typing import Set, Tuple, Optional

# Regex repairs for common OCR character merges and misclassifications
OCR_REGEX_REPAIRS = [
    # Missing halants / conjunct repairs
    (r'ಶಿಕಷ', 'ಶಿಕ್ಷ'),
    (r'ಕಷ', 'ಕ್ಷ'),
    (r'ಕಶ', 'ಕ್ಷ'),
    (r'ಕಸ', 'ಕ್ಷ'),
    (r'ವಯ', 'ವ್ಯ'),
    (r'ತರ', 'ತ್ರ'),
    (r'ಲಯ', 'ಲ್ಯ'),
    (r'ಶ್ರಿ', 'ಶ್ರೀ'),
    
    # Double consonant / ottu collapses
    (r'ಸಸ', 'ಸ್ಸ'),
    (r'ಚಚ', 'ಚ್ಚ'),
    (r'ಟಟ', 'ಟ್ಟ'),
    (r'ಡಡ', 'ಡ್ಡ'),
    (r'ತತ', 'ತ್ತ'),
    (r'ದದ', 'ದ್ದ'),
    (r'ನನ', 'ನ್ನ'),
    (r'ಪಪ', 'ಪ್ಪ'),
    (r'ಬಬ', 'ಬ್ಬ'),
    (r'ಮಮ', 'ಮ್ಮ'),
    (r'ರರ', 'ರ್ರ'),
    (r'ಲಲ', 'ಲ್ಲ'),
    
    # Common OCR verbal and nominal suffix errors
    (r'ಸುತದೆ$', 'ಸುತ್ತದೆ'),
    (r'ುತದೆ$', 'ುತ್ತದೆ'),
    (r'ುತಾರೆ$', 'ತ್ತಾರೆ'),
    (r'ುತಾನೆ$', 'ತ್ತಾನೆ'),
    (r'ುತಾಳೆ$', 'ತ್ತಾಳೆ'),
    (r'ವನು$', 'ವನ್ನು'),
    (r'ಯನು$', 'ಯನ್ನು'),
    (r'ರನು$', 'ರನ್ನು'),
    (r'ಗನು$', 'ಗಳನ್ನು'),
    (r'ಜನನು$', 'ಜನರನ್ನು'),
]

# Frequent vowel length optical confusions in OCR
VOWEL_LONG_REPAIRS = [
    ('ಜಿವನ', 'ಜೀವನ'),
    ('ಸಂಗಿತ', 'ಸಂಗೀತ'),
    ('ನಿರು', 'ನೀರು'),
    ('ಭುಮಿ', 'ಭೂಮಿ'),
    ('ರೂಪ', 'ರೂಪ'),
    ('ಶಿಲ', 'ಶೀಲ'),
    ('ದಿನ', 'ದಿನ'),
]


def apply_ocr_repairs(word: str, dictionary: Optional[Set[str]] = None) -> str:
    """
    Apply rule-based Indic OCR repairs to a word token.
    """
    w = word

    # 1. Direct Vowel Corrections
    for old_v, new_v in VOWEL_LONG_REPAIRS:
        if w.startswith(old_v):
            w = new_v + w[len(old_v):]
            break

    # 2. Regex Pattern Normalizations
    for pat, rep in OCR_REGEX_REPAIRS:
        w = re.sub(pat, rep, w)

    # 3. Check for initial 'ಪ' -> 'ಪ್ರ' prefix restoration
    if dictionary and w.startswith('ಪ') and not w.startswith('ಪ್ರ'):
        pra_cand = 'ಪ್ರ' + w[1:]
        if pra_cand in dictionary:
            w = pra_cand

    return w
