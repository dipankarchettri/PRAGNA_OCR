"""
Weighted Edit Distance & Optical Confusion Matrices for Kannada OCR Post-Processing
Levenshtein distance with tailored substitution costs for visually similar Kannada glyphs,
vowel signs, and multi-character optical OCR ligatures.
"""

from typing import Dict, Tuple, FrozenSet, List

# Visually similar / commonly confused Kannada characters and glyph shapes in OCR
CONFUSION_PAIRS: Dict[Tuple[str, str], float] = {
    # 1. Aspirated vs. Unaspirated Consonants (identical outer curves)
    ('ಕ', 'ಖ'): 0.25,
    ('ಗ', 'ಘ'): 0.25,
    ('ಚ', 'ಛ'): 0.25,
    ('ಜ', 'ಝ'): 0.25,
    ('ಟ', 'ಠ'): 0.25,
    ('ಡ', 'ಢ'): 0.25,
    ('ತ', 'ಥ'): 0.25,
    ('ದ', 'ಧ'): 0.25,
    ('ಪ', 'ಫ'): 0.25,
    ('ಬ', 'ಭ'): 0.25,

    # 2. Visually Confusable Base Consonants (loop, descender, and contour similarities)
    ('ಪ', 'ವ'): 0.25,
    ('ಬ', 'ವ'): 0.30,
    ('ನ', 'ವ'): 0.25,
    ('ಮ', 'ಯ'): 0.25,
    ('ಹ', 'ಪ'): 0.25,
    ('ಹ', 'ಷ'): 0.25,
    ('ಹ', 'ಯ'): 0.20,
    ('ಶ', 'ತ'): 0.22,
    ('ಲ', 'ಳ'): 0.25,
    ('ಳ', 'ಕ'): 0.22,
    ('ಳ', 'ಗ'): 0.35,
    ('ಣ', 'ನ'): 0.25,
    ('ಶ', 'ಷ'): 0.25,
    ('ಸ', 'ಷ'): 0.25,
    ('ಶ', 'ಸ'): 0.25,
    ('ರ', 'ಠ'): 0.25,
    ('ರ', 'ದ'): 0.30,
    ('ಜ', 'ಪ'): 0.25,
    ('ತ', 'ನ'): 0.35,
    ('ಥ', 'ಫ'): 0.25,
    ('ದ', 'ಞ'): 0.35,

    # 3. Independent Vowels & Vowel Flips
    ('ಅ', 'ಆ'): 0.25,
    ('ಇ', 'ಈ'): 0.25,
    ('ಉ', 'ಊ'): 0.25,
    ('ಎ', 'ಏ'): 0.25,
    ('ಒ', 'ಓ'): 0.20,
    ('ಎ', 'ವಿ'): 0.25,

    # 4. Multi-character / Subscript Ligature OCR Scans (High precision)
    ('ಎಜ', 'ವಿ'): 0.22,
    ('ವ್ಮ', 'ಮ್ಮ'): 0.20,
    ('ಮ್ನ', 'ಮ್ಮ'): 0.20,
    ('ಂಜ', 'ಂದ'): 0.20,
    ('ಸ್ಮ', 'ಷ್ಮೆ'): 0.20,
    ('ರೆಸ್ಮ', 'ರೇಷ್ಮ'): 0.20,
    ('ರೇಸ್ಮ', 'ರೇಷ್ಮ'): 0.20,
    ('ದ್ವ', 'ಧ್ವ'): 0.20,
    ('ಸರ', 'ಸ್ತರ'): 0.20,
    ('ಸಥ್ಸ', 'ಸಲ್ಲಿಸ'): 0.20,
    ('ಸಂಯುವ', 'ಸಾಯುವ'): 0.22,
}

_COST_MAP: Dict[FrozenSet[str], float] = {}
for (a, b), cost in CONFUSION_PAIRS.items():
    _COST_MAP[frozenset([a, b])] = cost

# Bidirectional lookup mapping for single & multi-character substitutions
GLYPH_CONFUSIONS: Dict[str, List[Tuple[str, float]]] = {}
for (a, b), cost in CONFUSION_PAIRS.items():
    GLYPH_CONFUSIONS.setdefault(a, [])
    if not any(x[0] == b for x in GLYPH_CONFUSIONS[a]):
        GLYPH_CONFUSIONS[a].append((b, cost))
    GLYPH_CONFUSIONS.setdefault(b, [])
    if not any(x[0] == a for x in GLYPH_CONFUSIONS[b]):
        GLYPH_CONFUSIONS[b].append((a, cost))


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return _COST_MAP.get(frozenset([a, b]), 1.0)


def weighted_edit_distance(s1: str, s2: str, max_dist: float = 4.0) -> float:
    """
    Compute weighted Levenshtein distance between two strings with early termination.
    """
    m, n = len(s1), len(s2)

    # Quick length check
    if abs(m - n) > max_dist:
        return max_dist + 1.0

    prev = [float(i) for i in range(n + 1)]
    curr = [0.0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = float(i)
        row_min = curr[0]
        for j in range(1, n + 1):
            cost = _sub_cost(s1[i - 1], s2[j - 1])
            curr[j] = min(
                prev[j] + 1.0,       # deletion
                curr[j - 1] + 1.0,   # insertion
                prev[j - 1] + cost   # substitution
            )
            row_min = min(row_min, curr[j])
        if row_min > max_dist:
            return max_dist + 1.0
        prev, curr = curr, prev

    return prev[n]

