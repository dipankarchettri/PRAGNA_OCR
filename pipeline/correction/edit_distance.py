"""
Weighted Edit Distance for Kannada OCR Post-Processing
Levenshtein distance with tailored substitution costs for visually similar Kannada glyphs.
"""

from typing import Dict, Tuple, FrozenSet

# Visually similar / commonly confused Kannada characters in OCR
CONFUSION_PAIRS: Dict[Tuple[str, str], float] = {
    ('ಕ', 'ಖ'): 0.3,
    ('ಗ', 'ಘ'): 0.3,
    ('ಚ', 'ಛ'): 0.3,
    ('ಜ', 'ಝ'): 0.3,
    ('ಟ', 'ಠ'): 0.3,
    ('ಡ', 'ಢ'): 0.3,
    ('ತ', 'ಥ'): 0.3,
    ('ದ', 'ಧ'): 0.3,
    ('ಪ', 'ಫ'): 0.3,
    ('ಬ', 'ಭ'): 0.3,
    ('ಣ', 'ನ'): 0.4,
    ('ಶ', 'ಷ'): 0.4,
    ('ಸ', 'ಷ'): 0.4,
    ('ಬ', 'ವ'): 0.4,
    ('ಲ', 'ಳ'): 0.4,
    ('ಅ', 'ಆ'): 0.5,
    ('ಇ', 'ಈ'): 0.5,
    ('ಉ', 'ಊ'): 0.5,
    ('ಎ', 'ಏ'): 0.5,
    ('ಒ', 'ಓ'): 0.5,
}

_COST_MAP: Dict[FrozenSet[str], float] = {}
for (a, b), cost in CONFUSION_PAIRS.items():
    _COST_MAP[frozenset([a, b])] = cost


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
