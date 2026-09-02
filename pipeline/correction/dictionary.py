"""
Kannada Dictionary Loader

Loads Hunspell kn_IN.dic + kn_IN.aff rules and maintains two distinct
vocabularies, which is the important thing about this module:

  membership set  -- every entry plus its affix expansions. A word in here is
                     never flagged as an error. Answers "is this real Kannada?"
  target set      -- only entries with real corpus attestation and enough
                     length to be worth proposing. Answers "may the corrector
                     rewrite some other word INTO this one?"

The split exists because the extended dictionary is 141,115 entries but
**70,778 of them have a corpus frequency of zero** and 522 are two code points
or fewer (ಅಕ, ಅಘ, ಖಸ...). Every one of those is a target the weighted-edit
search can snap a perfectly good word onto. Dropping them outright would be
worse -- a rare real word would then be flagged as an error and "corrected"
into a common one -- so they stay valid, they just stop being destinations.

Frequencies come from data/kn_freq.tsv (`word[/flags]<TAB>count`), built from
the same corpus as the n-gram model.
"""

import os
import re
from typing import Dict, List, Optional, Set

KANNADA_CHARS_RE = re.compile(r'^[ಀ-೿‌‍]+$')

# Minimum corpus attestation for a word to be offered as a correction target.
#
# Swept over 300 synthetic-noise lines (uncorrected CER 0.0155 / WER 0.1710).
# `minlen` is MIN_TARGET_LENGTH; (0, 0) is the old ungated behaviour:
#
#   minfreq  minlen      CER      WER   fixed  broke   precision
#         0       0   0.0089   0.0901     195    155       0.557
#         0       3   0.0089   0.0900     194    153       0.559
#         2       3   0.0088   0.0895     193    148       0.566   <- chosen
#         5       3   0.0089   0.0900     190    147       0.564
#        20       3   0.0088   0.0898     187    143       0.567
#       100       3   0.0089   0.0909     180    142       0.559
#         5       4   0.0089   0.0901     187    144       0.565
#
# The curve is flat -- be honest about the size of this: it buys ~7 fewer
# broken words per 300 lines, not a step change. 2 is the best point on CER,
# WER and retained fixes simultaneously, and tightening further trades away
# more real fixes than it prevents breaks (at 100 both CER and precision get
# worse again).
#
# 2 is also the same boundary MIN_CORPUS_ATTESTATION uses, for the same
# reason: ngram.add_vocabulary() pads every known dictionary word to a count
# of exactly 1, so ">= 2" is precisely "attested in real corpus text rather
# than merely present in the word list".
MIN_TARGET_FREQUENCY = 2

# Minimum length, in code points, for a correction target. Frequency alone
# cannot gate this: the single letter ರ has a corpus frequency of 1,064,175
# (corpus tokenization artifacts, not real one-letter words), which would make
# it one of the most attractive targets in the entire vocabulary. 4 was also
# measured and is slightly worse on every column than 3.
MIN_TARGET_LENGTH = 3

_dictionary: Set[str] = set()
_word_list: List[str] = []
_frequencies: Dict[str, int] = {}
_targets: Set[str] = set()


def is_kannada_word(word: str) -> bool:
    return bool(KANNADA_CHARS_RE.match(word))


def _parse_aff(aff_path: Optional[str]) -> Dict[str, List]:
    """Parse SFX rules out of a Hunspell .aff. (This file has no PFX rules.)"""
    sfx_rules: Dict[str, List] = {}
    if not (aff_path and os.path.exists(aff_path)):
        return sfx_rules

    with open(aff_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or not line.startswith('SFX'):
                continue
            parts = line.split()
            # `SFX <flag> <Y|N> <count>` is a rule-group header, not a rule.
            if len(parts) >= 4 and parts[2] in ('Y', 'N'):
                continue
            if len(parts) >= 4:
                flag = parts[1]
                strip = parts[2] if parts[2] != '0' else ''
                # Hunspell writes "add nothing" as `0`, but as `0/1000,1001`
                # when the added morph carries continuation flags. Comparing
                # the whole field against '0' missed that second form, so
                # `add` became the literal ASCII '0', the generated word failed
                # is_kannada_word, and the expansion was silently dropped --
                # 11 of the 247 rules in data/kn_IN.aff, every one of them a
                # pure-strip rule that consequently never fired.
                add_field = parts[3].split('/')[0]
                add_part = add_field if add_field != '0' else ''
                sfx_rules.setdefault(flag, []).append((strip, add_part))
    return sfx_rules


def load_and_expand_dic(dic_path: str, aff_path: Optional[str] = None) -> Set[str]:
    """Parse Hunspell .dic and .aff files, expanding suffix affix rules."""
    expanded_words = set()
    sfx_rules = _parse_aff(aff_path)

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
                    for flag in parts[1].split(','):
                        flag = flag.strip()
                        if flag not in sfx_rules:
                            continue
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


def load_frequencies(freq_path: Optional[str] = None) -> Dict[str, int]:
    """
    Load `word[/flags]<TAB>count` into the frequency table and derive the
    correction-target set from it.

    Missing file is not an error: the table stays empty, corpus_frequency()
    falls back to the n-gram unigram counts, and every dictionary word remains
    a valid target -- i.e. exactly the old behaviour.
    """
    global _frequencies, _targets

    if freq_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        freq_path = os.path.join(base_dir, 'data', 'kn_freq.tsv')

    frequencies: Dict[str, int] = {}
    if os.path.exists(freq_path):
        with open(freq_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                word, _, count = line.partition('\t')
                if not count:
                    continue
                # Entries carry the same /FLAGS suffix as the .dic.
                word = word.split('/')[0].strip()
                if not word:
                    continue
                try:
                    frequencies[word] = int(count.strip())
                except ValueError:
                    continue

    _frequencies = frequencies
    _targets = {
        w for w, c in frequencies.items()
        if c >= MIN_TARGET_FREQUENCY and len(w) >= MIN_TARGET_LENGTH
    }
    return frequencies


def load_dictionary(dic_path: Optional[str] = None) -> Set[str]:
    """Load the default dictionary, affix rules and frequency table."""
    if dic_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dic_path = os.path.join(base_dir, 'data', 'kn_IN.dic')

    aff_path = os.path.join(os.path.dirname(dic_path), 'kn_IN.aff')
    words = load_and_expand_dic(dic_path, aff_path)
    load_frequencies()
    return words


def get_dictionary() -> Set[str]:
    return _dictionary


def get_word_list() -> List[str]:
    return _word_list


def get_frequencies() -> Dict[str, int]:
    return _frequencies


def dictionary_frequency(word: str) -> int:
    """Corpus frequency from the curated table. 0 if unknown."""
    return _frequencies.get(word, 0)


def has_frequency_table() -> bool:
    return bool(_frequencies)


def is_correction_target(word: str, frequency: int) -> bool:
    """
    May the corrector rewrite some other word INTO this one?

    `frequency` is supplied by the caller rather than read from the curated
    table here, because the caller has a strictly better source. The n-gram
    model carries ~12.8M surface forms against this table's 141k, and the gap
    is exactly the inflected forms that matter:

        word          curated    n-gram
        ವಹಿಸುತ್ತದೆ           0    14,647
        ಮಂಡಲಿಗೆ              0       216

    Gating on the curated count alone would have blocked both -- and both are
    corrections this pipeline is required to make. Pass corpus_frequency().

    Two independent conditions, because neither covers the other:

      attestation  keeps the 70,778 zero-frequency dictionary entries from
                   acting as edit-distance attractors. Measured: the break
                   ಪಾರಲೌಕಿಕರಿಗೆ -> ಪಾರಲೌಕಿಕದಿಗೆ found on tests/fixtures/eval
                   goes away here, because the target has frequency 0.
      length       frequency cannot catch this one. The single letter ರ has
                   corpus frequency 1,064,175 (tokenization artifacts, not
                   real one-letter words), which would otherwise make it one
                   of the most attractive targets in the whole vocabulary.

    With no frequency table loaded at all, every word is allowed, preserving
    the original behaviour.
    """
    if not _frequencies:
        return True
    return len(word) >= MIN_TARGET_LENGTH and frequency >= MIN_TARGET_FREQUENCY
