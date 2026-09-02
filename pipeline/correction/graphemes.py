"""
Akshara (grapheme cluster) segmentation for Kannada.

Everything in the correction engine used to count Python code points, which is
the wrong unit for an Indic script and quietly distorted two things:

  edit distance  ಕಿ -> ಖೀ is ONE misread akshara, but it is two code-point
                 substitutions, so MAX_CORRECTION_EDIT_DISTANCE = 2.0 -- meant
                 to allow "one or two glyph errors" -- was already exhausted by
                 a single wrong syllable.
  candidate      deleting code point 0 of ಕಾಫಿ yields "ಾಫಿ", a string starting
  generation     with an orphan combining mark. That is not a possible OCR
                 reading of anything; it is a wasted dictionary lookup at best.

`aksharas()` returns the units the script is actually written in, so both of
those become "one edit means one visible glyph".

WHAT `\\X` ACTUALLY GIVES US HERE -- worth being precise, because it is not the
textbook akshara. The installed regex module carries Unicode 15.0 data, and
conjunct-cluster support (UAX #29 rule GB9c) only arrived in 15.1. So a
consonant cluster splits at the virama boundary:

    ಶಿಕ್ಷಣವು  ->  ['ಶಿ', 'ಕ್', 'ಷ', 'ಣ', 'ವು']      not ['ಶಿ', 'ಕ್ಷ', 'ಣ', 'ವು']

These are orthographic half-forms. That is fine, and arguably better, for what
this module is used for: each unit still begins with a base character and
carries its own dependent signs, so no operation can strip a matra off its
consonant or leave one stranded, and a subscript conjunct stays a separate
editable unit -- which matches how OCR actually fails on Kannada, where the
ottakshara is misread independently of the consonant above it.

If the regex data ever advances to 15.1+, clusters will start including
conjuncts and distances over conjunct-heavy words will shift. That is a
behaviour change, not a crash, and the cost table in edit_distance.py derives
costs from cluster *bases* rather than whole clusters precisely so it degrades
gracefully if that happens.
"""

from typing import List

import regex

_GRAPHEME_RE = regex.compile(r'\X')

# Dependent vowel signs, anusvara, visarga, virama, nukta -- everything that
# attaches to a base rather than standing on its own.
COMBINING_MARKS = frozenset(
    '಼'          # nukta
    'ಾಿೀುೂೃೄ'   # matras aa..vocalic rr
    'ೆೇೈೊೋೌ'          # matras e..au
    '್'          # virama
    'ೕೖ'    # length marks
    'ಂಃ'    # anusvara, visarga
)


def aksharas(s: str) -> List[str]:
    """Split into orthographic clusters. Never returns empty strings."""
    if not s:
        return []
    return _GRAPHEME_RE.findall(s)


def akshara_len(s: str) -> int:
    return len(aksharas(s))


def split_cluster(cluster: str) -> tuple:
    """
    (base, marks) for one cluster -- the character that carries the shape, and
    the dependent signs hanging off it. Used by the cost table so that ಸ್ vs ಷ್
    can inherit the cost the table records for ಸ vs ಷ.
    """
    if not cluster:
        return '', ''
    return cluster[0], cluster[1:]


def strip_marks(cluster: str) -> str:
    """The cluster's base character, with all dependent signs removed."""
    return ''.join(ch for ch in cluster if ch not in COMBINING_MARKS)
