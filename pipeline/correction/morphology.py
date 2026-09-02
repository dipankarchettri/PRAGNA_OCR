"""
Kannada Morphology & Sandhi Engine
Handles agglutinative Kannada suffix stripping, stem analysis, and Sandhi rules.
"""

from typing import Set, Tuple, Optional, Dict

# Dependent vowel signs (matras) -- a root ending in one of these is
# vowel-final, which changes which suffixes can attach directly (see
# decompose_word's oblique-case-suffix guard below).
KANNADA_VOWEL_SIGNS = set('ಾಿೀುೂೃೆೇೈೊೋೌ')

# Suffix normalizations for common OCR scanning corruptions
SUFFIX_NORMALIZATIONS: Dict[str, str] = {
    'ುತದೆ': 'ುತ್ತದೆ',
    'ಸುತದೆ': 'ಸುತ್ತದೆ',
    'ುತಾರೆ': 'ತ್ತಾರೆ',
    'ುತಾನೆ': 'ತ್ತಾನೆ',
    'ುತಾಳೆ': 'ತ್ತಾಳೆ',
    'ವಳ್ಳು': 'ವಳು',
    'ಅಂತಹವಳ್ಳು': 'ಅಂತಹವಳು',
    'ಟ್ಟತ್ತು': 'ಟ್ಟಿತ್ತು',
}

BROKEN_SUFFIXES = set(SUFFIX_NORMALIZATIONS.keys())

# Comprehensive sorted list of Kannada suffix morphemes (longest match first)
SUFFIXES = [
    # Compound case & associative suffixes
    'ಯೊಂದಿಗೆ', 'ನೊಂದಿಗೆ', 'ಗಳೊಂದಿಗೆ', 'ಗಳಲ್ಲಿನ', 'ಗಳಲ್ಲಿ', 'ಗಳನ್ನು', 'ಗಳಿಂದ',
    'ಗಳಿಗೆ', 'ಗಳಿಗಾಗಿ', 'ಗಳಿಗೂ', 'ಗಳಿಗಿಂತ', 'ಗಳ', 'ಗಳು',
    'ಯಲ್ಲಿನ', 'ನಲ್ಲಿನ', 'ದಲ್ಲಿನ', 'ಯಲ್ಲಿದ್ದ', 'ನಲ್ಲಿದ್ದ', 'ದಲ್ಲಿದ್ದ',
    'ಯಲ್ಲಿ', 'ನಲ್ಲಿ', 'ದಲ್ಲಿ', 'ಅಲ್ಲಿ', 'ಯೊಡನೆ', 'ನೊಡನೆ', 'ದೊಡನೆ',
    
    # Relative / Adjectival participle suffixes
    'ಅಂತಹವಳು', 'ಅಂತಹವನು', 'ಅಂತಹವರು', 'ಅಂತಹ', 'ಇಂತಹ', 'ಆದಂತಹ',
    'ವರೆವಿಗೂ', 'ವರೆಗೆ', 'ತನಕ',

    # Plurals, honorifics & pronominal suffixes
    'ಅವರುಗಳು', 'ಇವರುಗಳು', 'ಅವರು', 'ಇವರು', 'ಅವರ', 'ಇವರ',
    'ವರು', 'ಯರು', 'ಅರು', 'ರು', 'ಂದಿರು',

    # Accusative & Genitive
    'ಯನ್ನು', 'ನನ್ನು', 'ವನ್ನು', 'ನ್ನು', 'ಅನ್ನು', 'ವನು', 'ಯನು',
    'ಯಿಂದ', 'ನಿಂದ', 'ದಿಂದ', 'ಇಂದ',
    
    # Dative & Purposive
    'ಯಿಗೆ', 'ನಿಗೆ', 'ದಿಗೆ', 'ಕ್ಕಾಗಿಯೊ', 'ಕ್ಕಾಗಿಯೂ', 'ಕ್ಕಾಗಿ', 'ಗಾಗಿ', 'ಕ್ಕೆ', 'ಗೆ',
    'ಯಾಗಿಯೂ', 'ವಾಗಿಯೂ', 'ಯಾಗಿನ', 'ವಾಗಿನ', 'ಯಾಗಿ', 'ವಾಗಿ',
    
    # Numeral / Personal
    'ಯೊಬ್ಬ', 'ನೊಬ್ಬ', 'ದೊಬ್ಬ', 'ಒಬ್ಬ', 'ಯೊಬ್ಬಳು', 'ನೊಬ್ಬಳು',
    'ಯಲ್ಲ', 'ನಲ್ಲ', 'ದಲ್ಲ', 'ಅಲ್ಲ',
    'ಆಗಿದೆ', 'ಆಗಿ', 'ಆಗಿದ್ದಾರೆ', 'ಆಗಿದ್ದವು',
    'ಆದರೂ', 'ಆದರೆ', 'ಇದ್ದರೆ', 'ಇದ್ದರೂ',
    
    # Emphatic & Euphonic
    'ಯೂ', 'ನೂ', 'ದೂ', 'ವೂ', 'ಉ', 'ಒ', 'ವು', 'ಯು', 'ನು',
    'ಯ', 'ನ', 'ದ', 'ಅ',

    # Periphrastic passive/impersonal ("ಲಾಗು" auxiliary): root + ಲು + ಆಗು
    # sandhi-fuses to root + ಲಾಗು..., e.g. ಎನ್ನು+ಲಾಗುತ್ತದೆ -> ಎನ್ನಲಾಗುತ್ತದೆ,
    # ಇರಿಸು+ಲಾಗಿದೆ -> ಇರಿಸಲಾಗಿದೆ. Not decomposable by the plain 'ಆಗಿದೆ'/
    # 'ಆಗಿ' suffixes above since those need a literal independent ಆ, which
    # this sandhi form never has.
    'ಲಾಗುತ್ತಿದೆ', 'ಲಾಗುತ್ತದೆ', 'ಲಾಗುವುದು', 'ಲಾಗುತ್ತವೆ', 'ಲಾಗಿದೆ', 'ಲಾಯಿತು',
    
    # Verb inflections & Auxiliaries (Present / Past / Future / Perfect)
    'ಿಸುತ್ತವೆ', 'ಿಸುತ್ತಾನೆ', 'ಿಸುತ್ತಾಳೆ', 'ಿಸುತ್ತಾರ', 'ಿಸುತ್ತದೆ', 'ಿಸುತ್ತೀರಿ', 'ಿಸುತ್ತೇನೆ', 'ಿಸುತ್ತೇವೆ',
    'ುತ್ತವೆ', 'ತ್ತಾನೆ', 'ತ್ತಾಳೆ', 'ತ್ತಾರೆ', 'ುತ್ತದೆ', 'ತ್ತೀರಿ', 'ತ್ತೇನೆ', 'ತ್ತೇವೆ',
    'ುತದೆ', 'ುತಾರೆ', 'ುತಾನೆ', 'ುತಾಳೆ', 'ಸುತದೆ',
    'ಿಸಿದರು', 'ಿಸಿದನು', 'ಿಸಿದಳು', 'ಿಸಿದವು', 'ಿಸಿತು', 'ಿದರು', 'ಿದನು', 'ಿದಳು', 'ಿದವು', 'ಿತು',
    'ವಿತ್ತು', 'ಯಿತ್ತು', 'ಇತ್ತು', 'ದ್ದವು', 'ದ್ದರು', 'ದ್ದನು', 'ದ್ದಳು',
    'ಿಸಲು', 'ಿಲು', 'ುವುದು', 'ಿಕೊಂಡು',

    # Negative perfect participle ("not having done X"): verb root + ಇರ + ದ,
    # surface-realized as root + ಿರದ after the root's citation-form ು elides
    # before the following vowel (e.g. ಬಳಸು+ಇರದ -> ಬಳಸಿರದ, ಮಾಡು+ಇರದ ->
    # ಮಾಡಿರದ, ಹೋಗು+ಇರದ -> ಹೋಗಿರದ). A standard, productive negation pattern,
    # not specific to any one verb.
    'ಿರದ'
]

SUFFIXES.sort(key=len, reverse=True)

# Suffixes that only ever attach to a verb root (never a bare nominal
# particle). Scoped separately from SUFFIXES because the bare-consonant-stem
# fallback in decompose_word (root[:-1], dropping a verb root's trailing
# vowel, e.g. ಬೇಡು -> ಬೇಡ) is only linguistically valid here -- applying it
# for every suffix risks colliding with an unrelated dictionary particle
# that happens to share the same 3-letter prefix (e.g. stripping 'ದೂ' from
# ಎಂಬುದೂ down to root ಎಂಬು would wrongly match "ಎಂಬ", a fixed particle, not
# a verb stem taking 'ದೂ').
VERB_SUFFIXES = {
    'ಿಸುತ್ತವೆ', 'ಿಸುತ್ತಾನೆ', 'ಿಸುತ್ತಾಳೆ', 'ಿಸುತ್ತಾರ', 'ಿಸುತ್ತದೆ', 'ಿಸುತ್ತೀರಿ', 'ಿಸುತ್ತೇನೆ', 'ಿಸುತ್ತೇವೆ',
    'ುತ್ತವೆ', 'ತ್ತಾನೆ', 'ತ್ತಾಳೆ', 'ತ್ತಾರೆ', 'ುತ್ತದೆ', 'ತ್ತೀರಿ', 'ತ್ತೇನೆ', 'ತ್ತೇವೆ',
    'ುತದೆ', 'ುತಾರೆ', 'ುತಾನೆ', 'ುತಾಳೆ', 'ಸುತದೆ',
    'ಿಸಿದರು', 'ಿಸಿದನು', 'ಿಸಿದಳು', 'ಿಸಿದವು', 'ಿಸಿತು', 'ಿದರು', 'ಿದನು', 'ಿದಳು', 'ಿದವು', 'ಿತು',
    'ವಿತ್ತು', 'ಯಿತ್ತು', 'ಇತ್ತು', 'ದ್ದವು', 'ದ್ದರು', 'ದ್ದನು', 'ದ್ದಳು',
    'ಿಸಲು', 'ಿಲು', 'ುವುದು', 'ಿಕೊಂಡು',
    'ಲಾಗುತ್ತಿದೆ', 'ಲಾಗುತ್ತದೆ', 'ಲಾಗುವುದು', 'ಲಾಗುತ್ತವೆ', 'ಲಾಗಿದೆ', 'ಲಾಯಿತು',
    'ಿರದ',
}


def join_root_suffix(root: str, suf: str) -> str:
    """
    Reconstruct full word by applying Kannada Sandhi euphonic rules.
    """
    if not suf:
        return root

    # Normalize suffix if broken OCR suffix
    suf = SUFFIX_NORMALIZATIONS.get(suf, suf)

    # Verb sandhi: root ending in 'ಿಸು' or 'ಿಸಿ'
    if root.endswith('ಿಸು') or root.endswith('ಿಸಿ'):
        base = root[:-3]
        if suf in ('ಿಸುತ್ತದೆ', 'ುತ್ತದೆ'):
            return base + 'ಿಸುತ್ತದೆ'
        elif suf in ('ಿಸುತ್ತಾರೆ', 'ತ್ತಾರೆ'):
            return base + 'ಿಸುತ್ತಾರೆ'
        elif suf in ('ಿಸುತ್ತಾನೆ', 'ತ್ತಾನೆ'):
            return base + 'ಿಸುತ್ತಾನೆ'
        elif suf in ('ಿಸುತ್ತಾಳೆ', 'ತ್ತಾಳೆ'):
            return base + 'ಿಸುತ್ತಾಳೆ'
        elif suf in ('ಿಸುತ್ತವೆ', 'ುತ್ತವೆ'):
            return base + 'ಿಸುತ್ತವೆ'
        elif suf in ('ಿಸುತ್ತೇನೆ', 'ತ್ತೇನೆ'):
            return base + 'ಿಸುತ್ತೇನೆ'
        elif suf in ('ಿಸುತ್ತೇವೆ', 'ತ್ತೇವೆ'):
            return base + 'ಿಸುತ್ತೇವೆ'
        elif suf in ('ಿಸಿದರು', 'ಿದರು'):
            return base + 'ಿಸಿದರು'
        elif suf in ('ಿಸಿದನು', 'ಿದನು'):
            return base + 'ಿಸಿದನು'
        elif suf in ('ಿಸಿದಳು', 'ಿದಳು'):
            return base + 'ಿಸಿದಳು'
        elif suf in ('ಿಸಿದವು', 'ಿದವು'):
            return base + 'ಿಸಿದವು'
        elif suf in ('ಿಸಿತು', 'ಿತು'):
            return base + 'ಿಸಿತು'

    # General Verb & Noun sandhi: root ending in 'ು'
    if root.endswith('ು'):
        base = root[:-1]
        if suf.startswith('ತ್ತ'):
            return base + 'ು' + suf
        if suf in ('ಿಸುತ್ತದೆ', 'ುತ್ತದೆ'):
            return base + 'ಿಸುತ್ತದೆ'
        # Vowel beginning / Lopa Sandhi (e.g. ು + ಿ... -> ಿ...)
        if suf.startswith(('ಿ', 'ೀ', 'ೆ', 'ೇ', 'ಅ', 'ಇ', 'ಉ', 'ಎ', 'ಒ')):
            return base + suf
        if suf == 'ಗೆ':
            return base + 'ಿಗೆ'
        
    # Vowel sandhi with glide 'ಯ'
    if (root.endswith('ಿ') or root.endswith('ೆ')) and suf.startswith('ಯ'):
        return root + suf

    return root + suf


def decompose_word(word: str, dictionary: Set[str], exact_only: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Decompose a surface Kannada word into a valid (root, suffix) pair if possible.

    exact_only=True skips the vowel-ending-guessing stems_to_try fallback
    below, trusting only a root that is *literally* in the dictionary with
    no modification. Candidate generation (resolve_valid_surface_form) wants
    the fuzzy fallback -- it's what finds a correction root at all. But
    is_valid_surface_word, which hard-bypasses all correction the instant it
    returns True, must not: as the dictionary grows, an unrelated real word
    can coincidentally match a guessed vowel-ending stem (e.g. root[:-1]+'ಿ')
    and silently certify a genuinely OCR-corrupted word as already-correct.
    """
    if word in dictionary:
        return word, ''

    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf):
            root = word[:-len(suf)]

            # Suffix normalization
            correct_suf = SUFFIX_NORMALIZATIONS.get(suf, suf)

            # 'ದ'/'ನ'/'ಅ' are oblique-stem case markers that only attach to
            # a consonant-final stem (e.g. ಮನ+ದ needs a linking vowel first,
            # giving ಮನೆಯ, not bare ಮನ+ದ). Unlike 'ಯ' -- which has its own
            # glide-sandhi rule for vowel-final stems in join_root_suffix --
            # pasting these directly onto a root that already ends in a
            # vowel matra produces a non-word, e.g. ಇಲ್ಲಿ (here, vowel-final)
            # + ದ reads as "ಇಲ್ಲಿದ", which isn't Kannada, even though both
            # ಇಲ್ಲಿ and the surface form ಇಲ್ಲಿದ happen to arise from real
            # dictionary/suffix pieces.
            if suf in ('ದ', 'ನ', 'ಅ') and root and root[-1] in KANNADA_VOWEL_SIGNS:
                continue

            if root in dictionary:
                return root, correct_suf

            if exact_only:
                # Narrow exception, not the broad fuzzy fallback below: for
                # a verb-only suffix, also accept the root's standard ು-final
                # citation form (e.g. ಬಳಸಿರದ -> strip ಿರದ -> ಬಳಸ -> ಬಳಸು,
                # which the dictionary actually lists; the bare consonant
                # stem ಬಳಸ usually isn't listed on its own). This is a single
                # structurally predictable reconstruction -- how Kannada
                # dictionaries conventionally cite verbs -- not a guess among
                # several vowel endings, so it doesn't reopen the coincidental-
                # collision risk exact_only exists to close.
                if suf in VERB_SUFFIXES and root + 'ು' in dictionary:
                    return root + 'ು', correct_suf
                continue

            # Try common vowel ending variants of the substantive stem (must be at least 2 chars)
            stems_to_try = [root, root + 'ಿ', root + 'ು', root + 'ಾ', root + 'ೆ']
            if len(root) >= 3 and root.endswith('ಿಸ'):
                stems_to_try.append(root + 'ು')
            elif len(root) > 3:
                stems_to_try.append(root[:-1] + 'ಿ')
                stems_to_try.append(root[:-1] + 'ೆ')
                stems_to_try.append(root[:-1] + 'ು')
                # root[:-1] alone (dropping the trailing vowel matra with
                # nothing) covers verb roots whose bare consonant-final form
                # is itself the dictionary entry, e.g. ಬೇಡು+ತ್ತಾನೆ decomposes
                # with root=ಬೇಡು, and "ಬೇಡ" (not "ಬೇಡು") is what's listed.
                # Scoped to verb suffixes only -- for a non-verb suffix this
                # is unsound and risks colliding with an unrelated nominal
                # particle that happens to share the same shortened prefix.
                if suf in VERB_SUFFIXES:
                    stems_to_try.append(root[:-1])

            # Initials/single-letter prefixes that should never act as inflected verb/noun stems
            INITIALS = {'ವಿ', 'ಡಾ', 'ಶ್ರೀ', 'ಪ್ರೊ', 'ಎ', 'ಆ', 'ಈ', 'ಏ', 'ಓ', 'ಒ', 'ಎಸ್', 'ಜಿ', 'ಕೆ', 'ಪಿ', 'ಟಿ', 'ಎಂ', 'ಎಲ್', 'ಆರ್', 'ಬಿ', 'ಸಿ', 'ಡಿ', 'ಹೆಚ್', 'ಎನ್'}

            for st in stems_to_try:
                if len(st) >= 2 and st not in INITIALS and st in dictionary:
                    return st, correct_suf

    return None, None


def is_compound_word(word: str, dictionary: Set[str]) -> bool:
    """
    Check if word is a valid Kannada compound (Samasa) formed by joining substantive dictionary stems.
    """
    if len(word) < 4:
        return False

    # Common prefixes in Kannada
    kannada_prefixes = ['ಮರು', 'ಅನು', 'ಪ್ರತಿ', 'ಉಪ', 'ಸಹ', 'ಅಸಹ', 'ಸು', 'ದುರ್', 'ವಿ', 'ಮಹಾ', 'ಏಕ', 'ಸರ್ವ', 'ಆದಿ', 'ಅಂತರ್']
    for pfx in kannada_prefixes:
        if word.startswith(pfx) and len(word) > len(pfx) + 1:
            rest = word[len(pfx):]
            if rest in dictionary:
                return True

    # Split into 2 substantive components (neither can be a pure suffix or < 2 chars)
    for i in range(2, len(word) - 2):
        part1 = word[:i]
        part2 = word[i:]

        if part1 in SUFFIXES or part2 in SUFFIXES:
            continue

        if part1 in dictionary and part2 in dictionary:
            return True

        if (part1 + 'ು' in dictionary or part1 + 'ಾ' in dictionary or part1 + 'ಿ' in dictionary) and part2 in dictionary:
            return True

    return False



