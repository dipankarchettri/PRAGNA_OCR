"""
Kannada Dictionary Loader
Loads Hunspell kn_IN.dic + kn_IN.aff rules and maintains an indexed vocabulary.
"""

import os
import re
from typing import Set, List, Optional

KANNADA_CHARS_RE = re.compile(r'^[\u0C80-\u0CFF\u200C\u200D]+$')

_dictionary: Set[str] = set()
_word_list: List[str] = []


def is_kannada_word(word: str) -> bool:
    return bool(KANNADA_CHARS_RE.match(word))


def load_and_expand_dic(dic_path: str, aff_path: Optional[str] = None) -> Set[str]:
    """
    Parse Hunspell .dic and .aff files, expanding prefix/suffix affix rules.
    """
    expanded_words = set()

    # 1. Parse Affix Rules (.aff)
    sfx_rules = {}
    if aff_path and os.path.exists(aff_path):
        with open(aff_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or not line.startswith('SFX'):
                    continue
                parts = line.split()
                if len(parts) >= 4 and parts[2] in ('Y', 'N'):
                    continue
                if len(parts) >= 4:
                    flag = parts[1]
                    strip = parts[2] if parts[2] != '0' else ''
                    add_part = parts[3].split('/')[0] if parts[3] != '0' else ''
                    if flag not in sfx_rules:
                        sfx_rules[flag] = []
                    sfx_rules[flag].append((strip, add_part))

    # 2. Parse Dictionary (.dic)
    if os.path.exists(dic_path):
        with open(dic_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0 and line.isdigit():
                    continue
                if not line:
                    continue

                parts = line.split('/')
                base = parts[0].strip()
                if base and is_kannada_word(base):
                    expanded_words.add(base)

                if len(parts) > 1:
                    flags = parts[1].split(',')
                    for flag in flags:
                        flag = flag.strip()
                        if flag in sfx_rules:
                            for strip, add in sfx_rules[flag]:
                                if strip:
                                    if base.endswith(strip):
                                        w = base[:-len(strip)] + add
                                        if w and is_kannada_word(w):
                                            expanded_words.add(w)
                                else:
                                    w = base + add
                                    if w and is_kannada_word(w):
                                        expanded_words.add(w)

    global _dictionary, _word_list
    _dictionary = expanded_words
    _word_list = sorted(expanded_words)
    return expanded_words




def load_dictionary(dic_path: Optional[str] = None) -> Set[str]:
    """
    Helper to load default dictionary from project data directory.
    """
    if dic_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dic_path = os.path.join(base_dir, 'data', 'kn_IN.dic')
    
    aff_path = os.path.join(os.path.dirname(dic_path), 'kn_IN.aff')
    return load_and_expand_dic(dic_path, aff_path)


def get_dictionary() -> Set[str]:
    return _dictionary


def get_word_list() -> List[str]:
    return _word_list
