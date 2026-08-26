"""
Kannada Tokenizer
Splits input text into Kannada and non-Kannada tokens, preserving positions,
spaces, and special characters for exact reconstruction.
"""

import re

# Unicode range for Kannada digits: U+0CE6 - U+0CEF
# Unicode range for Kannada letters & signs: U+0C80 - U+0CE5, plus ZWJ and ZWNJ
KANNADA_DIGIT_RE = re.compile(r'[\u0CE6-\u0CEF]+')
KANNADA_WORD_RE = re.compile(r'[\u0C80-\u0CE5\u200C\u200D]+')
SPLIT_RE = re.compile(r'([\u0CE6-\u0CEF]+|[\u0C80-\u0CE5\u200C\u200D]+|[^\u0C80-\u0CFF\u200C\u200D]+)')


def tokenize(text: str) -> list[dict]:
    """
    Tokenize text into a list of token dictionaries:
      {'type': 'kannada' | 'number' | 'other', 'value': str, 'start': int, 'end': int}
    """
    tokens = []
    for m in SPLIT_RE.finditer(text):
        val = m.group()
        start, end = m.start(), m.end()
        if KANNADA_DIGIT_RE.fullmatch(val):
            tokens.append({'type': 'number', 'value': val, 'start': start, 'end': end})
        elif KANNADA_WORD_RE.fullmatch(val):
            tokens.append({'type': 'kannada', 'value': val, 'start': start, 'end': end})
        else:
            tokens.append({'type': 'other', 'value': val, 'start': start, 'end': end})
    return tokens



def reconstruct(tokens: list[dict]) -> str:
    """Rebuild text string from token dicts."""
    return ''.join(t['value'] for t in tokens)
