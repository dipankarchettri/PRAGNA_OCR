"""
Kannada Dictionary Loader

Loads Hunspell kn_IN.dic + kn_IN.aff rules and maintains two distinct
vocabularies. The split is the important thing about this module:

  membership set  -- every entry plus its affix expansions, 622,407 forms.
                     A word in here is never flagged as an error. Answers
                     "is this real Kannada?"
  target set      -- the subset that is also attested in real corpus text,
                     246,861 forms. Answers "may the corrector rewrite some
                     other word INTO this one?"

The asymmetry is deliberate and the two errors are not symmetric. A real word
missing from *membership* gets flagged and "corrected" into something else --
silent corpus corruption. A junk string admitted as a *target* is somewhere the
edit-distance search can land, which is the same corruption from the other
direction. So membership is made as large as the evidence allows, and
targethood is gated on corpus attestation on top of it.

Membership is data/kn_IN.dic (589,521 entries; see CLAUDE.md for where it comes
from and why it is not in git). Attestation comes from the n-gram model's
unigram counts, which cover ~12.8M surface forms -- far more than any curated
word list, and the reason a separate frequency table is no longer kept here.

A note on sizing, since the obvious instinct is "bigger is better". A 2.5M-entry
build of this dictionary was measured too, and was worse: harvested from corpus
text that itself contains OCR output, it admitted OCR errors as entries (ಕಾಥಿ,
the misreading of ಕಾಫಿ, was a listed word at frequency 37), and membership means
"never flag this" -- so the dictionary taught the corrector that the error was
correct. Head to head at identical settings, 2.5M scored precision 0.667 with
162 broken words against 589k's 0.708 with 121, on identical real-page results.
Vocabulary quality dominates vocabulary size here.
"""

import gzip
import os
import re
from typing import Dict, List, Optional, Set

KANNADA_CHARS_RE = re.compile(r'^[ಀ-೿‌‍]+$')

# Minimum corpus attestation for a word to be offered as a correction target.
#
# Re-swept against the 622k membership set; the old value was tuned when
# membership was 182k and does not transfer. Measured on 24 clean pages, 24
# degraded pages and 300 synthetic lines:
#
#   minfreq   targets   real9(fix/brk)  synth CER  synth fix/brk  precision
#         2   246,861        11 / 1         0.0091     350 / 40      0.897
#         5   194,715        11 / 1         0.0096     284 / 120     0.703
#        10   163,046        11 / 1         0.0095     280 / 118     0.704
#
# Tightening buys nothing: real-page results are identical and the synthetic
# columns are flat-to-worse. This is the opposite of the intuition that a
# smaller target set buys precision -- it does not, because the bigram-support
# and frequency-dominance gates in corrector.py already do that filtering, and
# the frequency floor only subtracts real fixes on top. The same sweep against
# a 2.5M membership was monotone in the same direction, so the finding is not
# an artefact of this particular vocabulary.
#
# 2 is the meaningful floor rather than 0 or 1 because ngram.add_vocabulary()
# pads every known dictionary word to a count of exactly 1, so ">= 2" is
# precisely "attested in real corpus text rather than merely present in the
# word list".
MIN_TARGET_FREQUENCY = 2

# Minimum length, in code points, for a correction target.
#
# The floor exists because frequency cannot exclude single letters -- ರ alone
# has a corpus frequency of 1,064,175, from corpus tokenization artifacts
# rather than real one-letter words, which would otherwise make it one of the
# most attractive targets in the whole vocabulary.
#
# 3 beats 2 on every set measured, at every VALID_WORD_TRUST_FREQUENCY:
#
#   len   real9(fix/brk)   typeset24(fix/brk)   synth fix/brk  synth precision
#     2       11 / 1            2 / 2             351 / 51         0.873
#     3       11 / 1            2 / 1             350 / 40         0.897
#
# This was briefly set to 2 on the strength of a measurement that turned out to
# be wrong -- classify_changes compared words by substring containment, so a
# correction that TRUNCATED a word scored as a fix (ಹೊಸದು -> ಹೊಸ against
# reference "ಹೊಸದು." counted fixed). Every number that informed the change was
# inflated. See tools/correction_bench.py.
MIN_TARGET_LENGTH = 3

_dictionary: Set[str] = set()
_word_list: List[str] = []


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


def _open_dic(dic_path: str):
    """
    Open the .dic, transparently preferring a gzipped copy.

    The 589k-entry dictionary is 18.4MB raw and 2.5MB gzipped, so git carries
    the .gz and this reads it in place -- no decompression step, no build
    artifact to keep in sync, and a fresh clone works without setup.py having
    to fetch anything. A plain .dic still wins if one is present, which keeps
    local experiments (drop a different .dic in and restart) working exactly as
    before.
    """
    if os.path.exists(dic_path):
        return open(dic_path, 'r', encoding='utf-8', errors='ignore')
    if os.path.exists(dic_path + '.gz'):
        return gzip.open(dic_path + '.gz', 'rt', encoding='utf-8', errors='ignore')
    return None


def load_and_expand_dic(dic_path: str, aff_path: Optional[str] = None) -> Set[str]:
    """Parse Hunspell .dic and .aff files, expanding suffix affix rules."""
    expanded_words = set()
    sfx_rules = _parse_aff(aff_path)

    handle = _open_dic(dic_path)
    if handle is not None:
        with handle as f:
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


def load_dictionary(dic_path: Optional[str] = None) -> Set[str]:
    """Load the default dictionary and its affix rules."""
    if dic_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dic_path = os.path.join(base_dir, 'data', 'kn_IN.dic')

    aff_path = os.path.join(os.path.dirname(dic_path), 'kn_IN.aff')
    return load_and_expand_dic(dic_path, aff_path)


def get_dictionary() -> Set[str]:
    return _dictionary


def get_word_list() -> List[str]:
    return _word_list


def is_correction_target(word: str, frequency: int) -> bool:
    """
    May the corrector rewrite some other word INTO this one?

    `frequency` is supplied by the caller rather than read from a table here,
    because the caller has a strictly better source. There used to be a curated
    frequency file (data/kn_freq.tsv, 141k words); the n-gram model carries
    ~12.8M surface forms, and the gap is exactly the inflected forms that
    matter:

        word          curated    n-gram
        ವಹಿಸುತ್ತದೆ           0    14,647
        ಮಂಡಲಿಗೆ              0       216

    Gating on the curated count alone would have blocked both -- and both are
    corrections this pipeline is required to make. Pass corpus_frequency().

    Callers must skip this gate entirely when ngram.has_corpus_counts() is
    False: with no real corpus loaded every word sits at the padded count of 1,
    and a ">= 2" floor would reject the whole vocabulary rather than loosen.

    Two independent conditions, because neither covers the other:

      attestation  keeps unattested dictionary entries from acting as
                   edit-distance attractors. Of the 622,407 membership forms
                   only 247,556 clear it.
      length       frequency cannot catch this one. The single letter ರ has
                   corpus frequency 1,064,175 (tokenization artifacts, not
                   real one-letter words), which would otherwise make it one
                   of the most attractive targets in the whole vocabulary.
    """
    return len(word) >= MIN_TARGET_LENGTH and frequency >= MIN_TARGET_FREQUENCY
