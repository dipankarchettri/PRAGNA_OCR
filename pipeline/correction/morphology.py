"""
Kannada Morphology & Sandhi Engine
Handles agglutinative Kannada suffix stripping, stem analysis, and Sandhi rules.
"""

from typing import Set, Tuple, Optional

# Comprehensive sorted list of Kannada suffix morphemes (longest match first)
SUFFIXES = [
    # Compound case suffixes
    'ಯೊಂದಿಗೆ', 'ನೊಂದಿಗೆ', 'ಗಳೊಂದಿಗೆ', 'ಗಳಲ್ಲಿನ', 'ಗಳಲ್ಲಿ', 'ಗಳನ್ನು', 'ಗಳಿಂದ',
    'ಗಳಿಗೆ', 'ಗಳಿಗಾಗಿ', 'ಗಳಿಗೂ', 'ಗಳಿಗಿಂತ', 'ಗಳ', 'ಗಳು',
    'ಯಲ್ಲಿನ', 'ನಲ್ಲಿನ', 'ದಲ್ಲಿನ', 'ಯಲ್ಲಿದ್ದ', 'ನಲ್ಲಿದ್ದ', 'ದಲ್ಲಿದ್ದ',
    'ಯಲ್ಲಿ', 'ನಲ್ಲಿ', 'ದಲ್ಲಿ', 'ಅಲ್ಲಿ', 'ಯೊಡನೆ', 'ನೊಡನೆ', 'ದೊಡನೆ',
    
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
    
    # Emphatic & Euphonic
    'ಯೂ', 'ನೂ', 'ದೂ', 'ವೂ', 'ಉ', 'ಒ', 'ವು', 'ಯು',
    'ಯ', 'ನ', 'ದ', 'ಅ',
    
    # Verb inflections (Present / Past / Future)
    'ಿಸುತ್ತವೆ', 'ಿಸುತ್ತಾನೆ', 'ಿಸುತ್ತಾಳೆ', 'ಿಸುತ್ತಾರ', 'ಿಸುತ್ತದೆ', 'ಿಸುತ್ತೀರಿ', 'ಿಸುತ್ತೇನೆ', 'ಿಸುತ್ತೇವೆ',
    'ುತ್ತವೆ', 'ತ್ತಾನೆ', 'ತ್ತಾಳೆ', 'ತ್ತಾರೆ', 'ುತ್ತದೆ', 'ತ್ತೀರಿ', 'ತ್ತೇನೆ', 'ತ್ತೇವೆ',
    'ುತದೆ', 'ುತಾರೆ', 'ುತಾನೆ', 'ುತಾಳೆ', 'ಸುತದೆ',
    'ಿಸಿದರು', 'ಿಸಿದನು', 'ಿಸಿದಳು', 'ಿಸಿದವು', 'ಿಸಿತು', 'ಿದರು', 'ಿದನು', 'ಿದಳು', 'ಿದವು', 'ಿತು',
    'ಿಸಲು', 'ಿಲು', 'ುವುದು', 'ಿಕೊಂಡು'
]

SUFFIXES.sort(key=len, reverse=True)


def join_root_suffix(root: str, suf: str) -> str:
    """
    Reconstruct full word by applying Kannada Sandhi euphonic rules.
    """
    if not suf:
        return root

    # Verb sandhi: root ending in 'ಿಸು', 'ಿಸಿ', or 'ು' + verbal suffix
    if (root.endswith('ಿಸು') or root.endswith('ಿಸಿ')) and (suf in ('ಿಸುತ್ತದೆ', 'ುತ್ತದೆ', 'ುತದೆ', 'ಸುತದೆ')):
        return root[:-3] + 'ಿಸುತ್ತದೆ'
    elif root.endswith('ು') and (suf in ('ಿಸುತ್ತದೆ', 'ುತ್ತದೆ', 'ುತದೆ', 'ಸುತದೆ')):
        return root[:-1] + 'ಿಸುತ್ತದೆ'


    # Noun sandhi: root ending in ು + ಗೆ -> ಿಗೆ (e.g. ಮನಸ್ಸು + ಗೆ -> ಮನಸ್ಸಿಗೆ)
    if root.endswith('ು') and suf == 'ಗೆ':
        return root[:-1] + 'ಿಗೆ'
        
    # Vowel sandhi with glide 'ಯ'
    if (root.endswith('ಿ') or root.endswith('ೆ')) and suf.startswith('ಯ'):
        return root + suf

    return root + suf


def decompose_word(word: str, dictionary: Set[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Decompose a surface Kannada word into a valid (root, suffix) pair if possible.
    """
    if word in dictionary:
        return word, ''

    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf):
            root = word[:-len(suf)]

            # Suffix normalization
            correct_suf = suf
            if suf in ('ುತದೆ', 'ಸುತದೆ'):
                correct_suf = 'ಿಸುತ್ತದೆ'
            elif suf == 'ವನು':
                correct_suf = 'ವನ್ನು'
            elif suf == 'ಯನು':
                correct_suf = 'ಯನ್ನು'

            if root in dictionary:
                return root, correct_suf

            # Try common vowel ending variants of the stem
            stems_to_try = [root, root + 'ಿ', root + 'ು', root + 'ಾ', root + 'ೆ']
            if len(root) > 1:
                stems_to_try.append(root[:-1] + 'ಿ')
                stems_to_try.append(root[:-1] + 'ೆ')
                stems_to_try.append(root[:-1] + 'ು')

            for st in stems_to_try:
                if st in dictionary:
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


