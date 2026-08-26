"""
Kannada Indic Script Normalization & Universal OCR Repairs
Handles systematic Indic Unicode normalizations, zero-digit/anusvara glitches,
and archaic Repha encoding transformations across all Kannada documents.
"""

import re
from typing import Set, Optional, Tuple

# Universal Indic Archaic Repha Regex:
# In legacy fonts and OCR engines, digit ೯ (U+0CEF) following a consonant + optional vowel sign
# represents an archaic Repha (e.g., ಕನಾ೯ಟಕ -> ಕರ್ನಾಟಕ, ಕತ೯ವ್ಯ -> ಕರ್ತವ್ಯ, ಮಾ೯ -> ರ್ಮಾ)
REPHA_REGEX = re.compile(r'([\u0C85-\u0CB9][\u0CBE-\u0CD6]?)೯')


def clean_unicode_glitches(text: str) -> str:
    """
    Clean OCR scanning artifacts, duplicate zero-digits, and anusvara glitches in Kannada text.
    e.g. ಗ್ರಹಿಸಿಕೊ೦ಂಡೇ -> ಗ್ರಹಿಸಿಕೊಂಡೇ, ಬಂ೦ದ -> ಬಂದ
    """
    if not text:
        return text

    # 1. Replace Kannada digit zero (U+0CE6) when mistakenly used as an Anusvara
    t = re.sub(r'[\u0CE6\u0C82]+([ಂಡ])', r'ಂ\1', text)
    t = re.sub(r'ಂಂಡ', 'ಂಡ', t)
    t = re.sub(r'[\u0CE6\u0C82]+', 'ಂ', t)
    t = re.sub(r'ಂ+', 'ಂ', t)

    # 2. Clean invalid independent vowel + dependent matra OCR glitches (e.g. ಎಿ -> ಅ)
    t = re.sub(r'ಎ[ಿೀ]', 'ಅ', t)

    # 3. Clean stray punctuation specks before words
    t = re.sub(r'^[«“"]\s+', r'“', t)
    return t


def normalize_indic_repha(text: str) -> str:
    """
    Universally normalize archaic Repha OCR/font encodings across all Kannada consonants.
    e.g. ಕನಾ೯ಟಕ -> ಕರ್ನಾಟಕ, ಕತ೯ವ್ಯ -> ಕರ್ತವ್ಯ, ಸ೯ -> ರ್ಸ
    """
    if not text or '೯' not in text:
        return text
    return REPHA_REGEX.sub(r'ರ್\1', text)


def normalize_script(text: str) -> str:
    """
    Apply full universal Indic script normalization.
    """
    return normalize_indic_repha(clean_unicode_glitches(text))


def apply_ocr_repairs(word: str, dictionary: Optional[Set[str]] = None) -> Tuple[str, str]:
    """
    Apply safe, rule-based Indic script normalization to a word token.
    Returns: (repaired_word, repair_type)
    """
    norm = normalize_script(word)
    if norm != word:
        return norm, 'ocr_repair'
    return word, 'none'


