"""
Weighted Edit Distance & Optical Confusion Matrices for Kannada OCR Post-Processing
Levenshtein distance with tailored substitution costs for visually similar Kannada glyphs,
vowel signs, and multi-character optical OCR ligatures.
"""

from typing import Dict, Tuple, FrozenSet, List

from .graphemes import aksharas, split_cluster

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

    # 2b. Added from an empirical audit (aligning real raw-Tesseract output
    # against hand-verified ground truth on 4 held-out book pages) rather
    # than visual-similarity intuition like the rest of this table -- see
    # git history. Only pairs that recurred 2+ times, including across
    # different source documents, were kept; single-occurrence diffs from
    # that same audit were discarded as too likely to be one-off noise
    # (e.g. from the single worst-quality page in the sample) rather than a
    # generalizable confusion.
    ('ಮ', 'ವ'): 0.25,
    ('ಯ', 'ವ'): 0.25,
    ('ಜ', 'ಚ'): 0.25,
    ('ಲ', 'ರ'): 0.25,

    # 3. Independent Vowels & Vowel Flips
    ('ಅ', 'ಆ'): 0.25,
    ('ಇ', 'ಈ'): 0.25,
    ('ಉ', 'ಊ'): 0.25,
    ('ಎ', 'ಏ'): 0.25,
    ('ಒ', 'ಓ'): 0.20,
    ('ಎ', 'ವಿ'): 0.25,

    # 4. Multi-character / Subscript Ligature OCR Scans (High precision)
    ('ವ್ಮ', 'ಮ್ಮ'): 0.20,
    ('ಮ್ನ', 'ಮ್ಮ'): 0.20,
    ('ಂಜ', 'ಂದ'): 0.20,
    ('ಸ್ಮ', 'ಷ್ಮ'): 0.20,
    ('ಸ್ಮೆ', 'ಷ್ಮೆ'): 0.20,
    ('ದ್ವ', 'ಧ್ವ'): 0.20,

    # 4b. From the same empirical audit as 2b -- the single strongest
    # pattern found: subscript/conjunct ya (್ಯ) misread as subscript ka
    # (್ಕ), recurring independently in two different source documents
    # (e.g. ದುಷ್ಯಂತ -> ದುಷ್ಕಂತ, ...ಸ್ಯ -> ...ಸ್ಕ three separate times on one
    # page alone). Consistent with this project's own documented finding
    # that Kannada ottakshara/subscript conjuncts are the most fragile part
    # of the script under OCR (see the --dpi rationale in CLAUDE.md).
    ('್ಯ', '್ಕ'): 0.20,
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


# Cost of two clusters that share a base character but carry different
# dependent signs -- ಕಿ vs ಕೀ, or ಕ vs ಕಿ. This is a misread or dropped
# diacritic, the single most common thing OCR loses on this script, and it is
# emphatically not the same event as reading one consonant as another.
#
# Set to match what the generators already charge for the same edit: item 6 of
# generate_kannada_candidates prices a vowel-length swap at 0.25. Before this,
# the verifier scored that identical edit at 1.0, so generator and verifier
# disagreed by 4x about the cost of the most common error in the corpus.
#
# MEASURED INERT, and worth saying so rather than implying it is tuned: swept
# {0.15, 0.25, 0.40, 0.60} across all three benches and every column was
# identical to four decimal places. This distance only feeds
# MAX_CORRECTION_EDIT_DISTANCE, a reject-or-accept gate that almost nothing
# reaches; it does not participate in ranking, which runs on the generators'
# own base costs. The value becomes load-bearing only if candidate ranking
# starts using it (Phase 3d), so it is set to the defensible number rather
# than an empirically chosen one.
MATRA_SUB_COST = 0.25


def _cluster_sub_cost(a: str, b: str) -> float:
    """
    Substitution cost between two orthographic clusters (see graphemes.py).

    Three tiers, cheapest first:

      1. The pair is listed in CONFUSION_PAIRS verbatim. This is what finally
         makes the multi-character entries (ವ್ಮ/ಮ್ಮ, ಸ್ಮೆ/ಷ್ಮೆ, ್ಯ/್ಕ ...)
         reachable: _sub_cost only ever compared single code points, so those
         9 rows priced a substitution at 0.20 in the generator while the
         verifier scored the same edit at ~2.0 and could reject it outright.
      2. Same dependent signs, confusable bases -- ಸ್ vs ಷ್ inherits the cost
         the table records for ಸ vs ಷ. Deriving from the base rather than
         requiring a literal row is what keeps the table free of the
         combinatorial explosion of every consonant x every matra.
      3. Same base, different signs -- a diacritic error, MATRA_SUB_COST.

    A cluster substitution never costs more than 1.0, the price of replacing
    one glyph with an unrelated one.
    """
    if a == b:
        return 0.0

    cost = _COST_MAP.get(frozenset([a, b]))
    if cost is not None:
        return cost

    base_a, marks_a = split_cluster(a)
    base_b, marks_b = split_cluster(b)
    mark_penalty = 0.0 if marks_a == marks_b else MATRA_SUB_COST

    if base_a == base_b:
        return mark_penalty

    base_cost = _COST_MAP.get(frozenset([base_a, base_b]))
    if base_cost is not None:
        return min(1.0, base_cost + mark_penalty)

    return 1.0


def weighted_edit_distance(s1: str, s2: str, max_dist: float = 4.0) -> float:
    """
    Weighted Levenshtein distance in *aksharas*, not code points.

    The unit matters. ಕಾಫಿ is 4 code points but 2 clusters, and a single
    misread syllable ಕಿ -> ಖೀ is one cluster substitution but two code-point
    substitutions. Measuring in code points meant every threshold expressed in
    this metric -- MAX_CORRECTION_EDIT_DISTANCE above all -- was really
    counting "how many combining marks did this touch", which for an abugida
    is not a measure of how different two words look.
    """
    a1, a2 = aksharas(s1), aksharas(s2)
    m, n = len(a1), len(a2)

    # Quick length check
    if abs(m - n) > max_dist:
        return max_dist + 1.0

    prev = [float(i) for i in range(n + 1)]
    curr = [0.0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = float(i)
        row_min = curr[0]
        for j in range(1, n + 1):
            cost = _cluster_sub_cost(a1[i - 1], a2[j - 1])
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

