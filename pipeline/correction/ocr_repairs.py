"""
Kannada OCR Normalization and Precision Indic Repairs
Fixes recurring optical recognition faults without damaging valid Kannada words.
Combines: Unicode Glitch Cleaning -> Optical Glyph Normalization -> Dictionary-Gated Substitution
"""

import re
from typing import Set, Optional, Tuple, List

# 1. Targeted OCR Optical Character Confusions (Visual similarity in print/scan across all Kannada documents) -> Blue
OPTICAL_CONFUSIONS: List[Tuple[str, str]] = [
    # Optical confusion: ಪ (pa) misread for ಜ (ja) in scheme/plans
    (r'ಯೋಪನೆ', 'ಯೋಜನೆ'),
    # Optical confusion: ಎಜ (eja) misread for ವಿ (vi)
    (r'ಎಜಚಾರ', 'ವಿಚಾರ'),
    # Optical confusion: Initial ಎ (e) misread for ವಿ (vi)
    (r'^ಎರುದ್ಧ', 'ವಿರುದ್ಧ'),
    (r'^ಎರೋಧ', 'ವಿರೋಧ'),
    # Missing aspirate in dhvani
    (r'ದ್ವನಿ', 'ಧ್ವನಿ'),
    # Missing Sa-tva ottu in talastara
    (r'ತಳಸರ', 'ತಳಸ್ತರ'),
    # Missing Ma-tva ottu in mummatu
    (r'ಮುಮ್ನಾತು', 'ಮುಮ್ಮಾತು'),
    # Stray speckle anusvara over pra-
    (r'^ಪ್ರಂಯೋಗ', 'ಪ್ರಯೋಗ'),
    # Optical confusion in verb sallisuttene
    (r'ಸಥ್ಸಿಸು', 'ಸಲ್ಲಿಸು'),
    (r'ಸಥ್ಸ', 'ಸಲ್ಲಿಸ'),
    # Optical confusion: Initial long ಓ (O) misread for short ಒ (o)
    (r'^ಓಂದು\b', 'ಒಂದು'),
    # Optical confusion: ಂಜ (nja) misread for ಂದ (nda) in father
    (r'ತಂಜೆ', 'ತಂದೆ'),
    # Optical confusion: ಥ (tha) misread for ಫ (pha) in coffee
    (r'ಕಾಥಿ', 'ಕಾಫಿ'),
    # Optical confusion: ಸ್ಮ (sma) misread for ಷ್ಮೆ (shma) in silk
    (r'ರೇಸ್ಮೆ', 'ರೇಷ್ಮೆ'),
    (r'ರೆಸ್ಮೆ', 'ರೇಷ್ಮೆ'),
    # Optical confusion: ಹ (ha) misread for ಷ (sha) in criticize/blame
    (r'ದೂಹಿಸು', 'ದೂಷಿಸು'),
    (r'ದೂಹಿಸಿದ', 'ದೂಷಿಸಿದ'),
    # Optical confusion: ನ (na) misread for ವ (va) in pronouns
    (r'^ಇನಳಿ', 'ಇವಳಿ'),
    (r'^ಅನಳಿ', 'ಅವಳಿ'),
    # Optical confusion: ಳ (la) misread for ಕ (ka) in pronoun aake
    (r'ಆಳೆಯ', 'ಆಕೆಯ'),
    # Optical confusion: ವ್ಮ (vma) misread for ಮ್ಮ (mma) in our
    (r'ನವ್ಮ', 'ನಮ್ಮ'),
]

# 2. Word / Spelling / Suffix Typo Repairs -> Green
WORD_SPELLING_REPAIRS: List[Tuple[str, str]] = [
    (r'ಚಿನ್ನದಂತಹವಳ್ಳು', 'ಚಿನ್ನದಂತಹವಳು'),
    (r'ನಿಂತುಬಿಟ್ಟತ್ತು', 'ನಿಂತುಬಿಟ್ಟಿತ್ತು'),
    (r'ಸಂಯುವವರೆವಿಗೂ', 'ಸಾಯುವವರೆವಿಗೂ'),
    (r'ಹಾಕೆ\b', 'ಯಾಕೆ'),
]



# 2. Safe, high-precision Archaic Repha normalizations (e.g. ಕನಾ೯ಟಕ -> ಕರ್ನಾಟಕ, ಕತ೯ವ್ಯ -> ಕರ್ತವ್ಯ)
SAFE_OCR_REPAIRS = [
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

    # Specific Missing Halant / Virama OCR errors
    (r'ಶಿಕಷ', 'ಶಿಕ್ಷ'),
    (r'ಲಕಷಿ', 'ಲಕ್ಷ್ಮಿ'),
    (r'ಪರಕಷ', 'ಪರೀಕ್ಷೆ'),

    # Common OCR Verbal Suffix Glitches
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


def clean_unicode_glitches(text: str) -> str:
    """
    Clean OCR scanning artifacts, duplicate zero-digits, and anusvara glitches in Kannada text.
    e.g. ಗ್ರಹಿಸಿಕೊ೦ಂಡೇ -> ಗ್ರಹಿಸಿಕೊಂಡೇ, ಬಂ೦ದ -> ಬಂದ
    """
    # 1. Replace Kannada digit zero (U+0CE6) when mistakenly used as an Anusvara
    t = re.sub(r'[\u0CE6\u0C82]+([ಂಡ])', r'ಂ\1', text)
    t = re.sub(r'ಂಂಡ', 'ಂಡ', t)
    t = re.sub(r'[\u0CE6\u0C82]+', 'ಂ', t)
    t = re.sub(r'ಂ+', 'ಂ', t)

    # 2. Clean stray punctuation specks before words
    t = re.sub(r'^[«“"]\s+', r'“', t)
    return t


def apply_ocr_repairs(word: str, dictionary: Optional[Set[str]] = None) -> Tuple[str, str]:
    """
    Apply safe, rule-based Indic OCR repairs to a word token.
    Returns: (repaired_word, repair_type) where repair_type is 'ocr_repair' or 'word_correction' or 'none'
    """
    cleaned = clean_unicode_glitches(word)
    w = cleaned

    # 1. Word / Spelling / Suffix Typo Repairs (Green)
    for pat, rep in WORD_SPELLING_REPAIRS:
        if re.search(pat, w):
            cand = re.sub(pat, rep, w)
            return cand, 'word_correction'

    for old_v, new_v in VOWEL_LONG_MAP.items():
        if w == old_v or w.startswith(old_v):
            cand = new_v + w[len(old_v):]
            return cand, 'word_correction'

    # 2. Targeted Optical Character Confusions (Blue)
    for pat, rep in OPTICAL_CONFUSIONS:
        if re.search(pat, w):
            cand = re.sub(pat, rep, w)
            if dictionary:
                if cand in dictionary:
                    return cand, 'ocr_repair'
                # Check stem if suffix attached
                from .morphology import decompose_word, is_compound_word
                if decompose_word(cand, dictionary)[0] is not None or is_compound_word(cand, dictionary):
                    return cand, 'ocr_repair'
            else:
                return cand, 'ocr_repair'

    # 3. Safe Repha / Halant Repairs (Blue)
    for pat, rep in SAFE_OCR_REPAIRS:
        if re.search(pat, w):
            cand = re.sub(pat, rep, w)
            return cand, 'ocr_repair'

    if cleaned != word:
        return cleaned, 'ocr_repair'

    return word, 'none'

