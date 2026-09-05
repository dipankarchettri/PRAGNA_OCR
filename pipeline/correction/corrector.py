"""
Main Kannada Autocorrect & Post-OCR Correction Engine
Combines: Universal Script Normalization -> Space Healing -> Dynamic Indic Candidate Generation -> N-Gram Context Ranking
"""

import re
from typing import Dict, List, Tuple, Any, Optional, Set
from .dictionary import get_dictionary, get_word_list, load_dictionary, is_correction_target
from .tokenizer import tokenize, reconstruct
from .edit_distance import weighted_edit_distance, GLYPH_CONFUSIONS
from .graphemes import aksharas
from .morphology import (
    decompose_word,
    join_root_suffix,
    is_compound_word,
    suffixes_ending_with,
    SUFFIX_SET,
    SUFFIX_NORMALIZATIONS,
    BROKEN_SUFFIXES
)
from .ocr_repairs import normalize_script, clean_unicode_glitches
from .ngram import (
    train_from_word_list, score_candidate, corpus_frequency, has_bigram_support,
    has_corpus_counts,
)

# Minimum real-corpus attestation count for a word absent from the curated
# dictionary to be treated as potentially "already correct" at all (see
# FREQUENCY_DOMINANCE_RATIO below). The curated Hunspell dictionary (~46k
# entries) is nowhere near complete against ordinary Kannada inflected forms
# -- the trained corpus model covers ~12M unique surface forms and catches
# this gap directly. >=2 excludes add_vocabulary()'s padding value of
# exactly 1 for known dictionary words, requiring genuine repeated real-text
# attestation rather than a single coincidental corpus occurrence.
MIN_CORPUS_ATTESTATION = 2

# Corpus attestation at which mere dictionary membership is trusted to end the
# matter, short-circuiting candidate generation for a word entirely.
#
# A word can be "valid" without being right. is_valid_surface_word accepts
# anything morphology can decompose against a 622k-form dictionary, and OCR
# errors decompose readily -- ಜಿವನದಲ್ಲಿ, the misreading of ಜೀವನದಲ್ಲಿ, passes at
# corpus frequency 63 against the correct form's 112,670, and the test suite
# requires the pipeline to fix it. With membership alone as the gate it
# short-circuits before any candidate is generated and becomes uncorrectable by
# construction.
#
# (This was starker still on a 2.5M build of the dictionary, where ಕಾಥಿ -- the
# misreading of ಕಾಫಿ -- was a *listed entry* at frequency 37. That vocabulary
# was rejected for this reason among others; see dictionary.py.)
#
# Below this floor, a dictionary word still goes through candidate generation
# and is then protected by FREQUENCY_DOMINANCE_RATIO, which is precisely the
# check built to separate "rare but real" from "an error swamped by its own
# correct form": ಮಂಡಲಿಗೆ (freq 216) survives at 82x behind ಮಂಡಳಿಗೆ, while ಕಾಥಿ
# at 2,442x behind ಕಾಫಿ does not. So this does not weaken protection of rare
# real words; it hands them to the check designed for them.
#
# Swept against 9 real scanned pages, 24 typeset pages and 300 synthetic lines:
#
#     VWT    real9(fix/brk)  typeset(fix/brk)  synth CER  synth fix/brk  prec
#        0       11 / 1           2 / 1          0.0096     305 / 23    0.930
#     1000       11 / 1           2 / 1          0.0091     350 / 40    0.897
#     5000       11 / 3           2 / 2          0.0092     350 / 41    0.895
#
# The real pages -- the only honest gate here -- are identical at 0 and 1000,
# and clearly worse at 5000, which breaks three words instead of one. Between 0
# and 1000 the real pages cannot distinguish, so the tiebreak is that 0 fails
# outright: it leaves ಜಿವನದಲ್ಲಿ uncorrected, which test_pipeline requires, and
# at 0 this constant is a no-op anyway since every valid word short-circuits.
#
# 5000 was chosen at first on the strength of a measurement that turned out to
# be wrong -- classify_changes compared words by substring containment, so a
# correction that TRUNCATED a word counted as a fix. That inflated 5000's
# apparent gain and hid its extra breaks entirely. See
# tools/correction_bench.py for the bug.
#
# The cost is real: every word below this floor runs candidate generation,
# roughly doubling correction time, which the Phase 3a speedup paid for.
VALID_WORD_TRUST_FREQUENCY = 1000

# How many times more corpus-attested a correction candidate must be than
# the original word before suggest_kannada_word will actually apply it, once
# the original itself clears MIN_CORPUS_ATTESTATION. A huge real-world
# corpus inevitably contains some fraction of typos/OCR noise/unrelated
# loanword collisions, and the original word's raw attestation count *alone*
# can't separate that from a genuinely rare real word -- both can sit in
# similar low-hundreds ranges. What actually distinguishes them is the
# relative gap to the best correction candidate: "ಮಂಡಲಿಗೆ" (freq 198, the
# real spelling in an actual exam-board document) must be KEPT even though
# "ಮಂಡಳಿಗೆ" (freq 17,658, the generic dictionary spelling) is 89x more
# common -- that's ordinary spelling variation. But "ಸಂಗಿತ" (freq 295,
# common OCR-typo noise) must be CORRECTED to "ಸಂಗೀತ" (freq 197,520) at a
# 669x gap -- that's not plausibly variation, it's a real word being
# swamped by its own typo's corpus noise. 250x sits between the two
# measured cases (89x keep, 669x correct) with a comfortable margin on
# both sides.
FREQUENCY_DOMINANCE_RATIO = 250

# Fallback minimum real-corpus attestation for a 'word_correction_unconstrained'
# candidate (blind single-char deletion/insertion -- see generate_kannada_candidates
# item 10) when there's no prev/next Kannada word to check bigram support
# against at all (e.g. a word at a line/document boundary, or a direct
# suggest_kannada_word("...") call with no context args). Frequency
# dominance alone can't gate this mechanism -- see the item-10 comment for
# why -- so with zero context to fall back on we require the correction
# target to be unambiguously common. Reuses the same freq>=100 bar already
# established elsewhere in this project as the "safe to trust as genuinely
# common" threshold (see the Alar dictionary-growth vocabulary work).
MIN_CANDIDATE_ATTESTATION_FOR_UNSEEN_ORIGINAL = 100

# Tesseract's own per-word confidence (0-100), when available, at or above
# which a word is left untouched regardless of what the corrector would
# otherwise suggest. Calibrated against real cases from this project's own
# testing: every word that was ALREADY correct but got wrongly "corrected"
# this session (ಭಾವಃ, ಇಮೇ, ದೇಹಾ, ಅಜೋ, ಯಣ, ಅನುಸರಣವು, ಹಣ್ಣು, ಮೇಲಿ) had OCR
# confidence 90-96; the one clear genuine-error correction in the same
# sample (ಕಲವು -> ಕೆಲವು) sat at 69, and clear noise (ಕವಿಂರು) at 16. 85
# sits with margin on both sides of that split. This is a blunter, cheaper
# signal than any of the corpus-evidence gating above -- it doesn't require
# guessing at what the word means, just trusting that Tesseract was
# actually confident about what it read.
HIGH_OCR_CONFIDENCE_TRUST = 85

# Corpus attestation at which a word is treated as able to stand on its own,
# for the purpose of deciding whether the space before it is real (see
# heal_split_tokens).
#
# This deliberately does NOT use is_valid_surface_word. That routine accepts a
# word if morphology can decompose it against the dictionary, and a genuine
# OCR fragment decomposes very easily -- "ಲೆಯಲ್ಲಿ" is ಲೆ + ಯಲ್ಲಿ and
# "ವಾಗಬೇಕಿದೆ" splits just as neatly -- so as the dictionary grew past half a
# million forms the fragments started passing the very check that exists to
# identify them, and genuine splits stopped being healed.
#
# Literal membership plus attestation separates them cleanly. Measured:
#
#     stands alone (must not merge)   ಅವರು 3,874,677   ಅವರ 2,404,297   both listed
#     fragment     (must merge)       ಲೆಯಲ್ಲಿ    143   ವಾಗಬೇಕಿದೆ  131   neither listed
#
# 300 sits above the fragments and below ಪರಿಚಯವು (634), a real inflected form
# absent from the .dic that must keep its own word boundary. The asymmetry is
# intentional: a wrongly merged pair reads as one plausible token and is
# effectively undetectable downstream, whereas an unhealed split stays visibly
# broken, so ambiguity resolves toward leaving the space alone.
HEAL_STANDALONE_FREQUENCY = 300

# Strict protected tokens that should never be altered or fused
PROTECTED_TOKENS = {
    'ಶ್ರೀ', 'ಡಾ', 'ಪ್ರೊ', 'ಆ', 'ಈ', 'ಏ', 'ಮತ್ತು', 'ಹಾಗೂ', 'ಅಥವಾ',
    'ಎಂದು', 'ಎಂಬ', 'ಒಂದು', 'ಆಗಿದೆ', 'ಆಗಿದ್ದಾರೆ', 'ಆಗುವುದು', 'ಆಗಿದೆ.',
    'ಇದೆ', 'ಇದು', 'ಅಲ್ಲ', 'ಇಲ್ಲ'
}

# Systematic Vowel Matra & Independent Vowel transformation pairs (Short <-> Long)
VOWEL_MATRA_MAP = {
    'ಿ': 'ೀ', 'ೀ': 'ಿ',
    'ು': 'ೂ', 'ೂ': 'ು',
    'ೆ': 'ೇ', 'ೇ': 'ೆ',
    'ೊ': 'ೋ', 'ೋ': 'ೊ',
    'ಅ': 'ಆ', 'ಆ': 'ಅ',
    'ಇ': 'ಈ', 'ಈ': 'ಇ',
    'ಉ': 'ಊ', 'ಊ': 'ಉ',
    'ಎ': 'ಏ', 'ಏ': 'ಎ',
    'ಒ': 'ಓ', 'ಓ': 'ಒ'
}

KANNADA_CONSONANTS = set('ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲವಶಷಸಹಳೞ')

# Dependent vowel signs (matras) + anusvara/virama -- small diacritics that
# OCR frequently drops anywhere in a word, not just at word boundaries.
KANNADA_VOWEL_SIGNS = 'ಾಿೀುೂೃೆೇೈೊೋೌಂ್'

# Reject a correction if its true weighted edit distance from the
# script-normalized word exceeds this.
#
# NOW DENOMINATED IN AKSHARAS, NOT CODE POINTS -- weighted_edit_distance
# changed units, so the old value of 2.0 does not carry over and was re-swept.
# 1.0 means "at most one visibly wrong glyph", which is what this constant was
# always trying to express: a single misread akshara ಕಿ -> ಖೀ is two code-point
# substitutions, so under the old metric it already exhausted the entire budget
# and a second real error could never be tolerated.
#
# Swept {0.75, 1.0, 1.5, 2.0} over all three benches. Above 1.0 the gate is
# inert -- cluster costs put essentially every candidate below it -- and 0.75
# loses the genuine fixes on both real-OCR page sets (clean 3 -> 2, degraded
# 2 -> 1) to buy a slightly better synthetic CER. 1.0 is the largest value
# that still constrains anything and the best on both real-page sets.
#
# Measured from the *normalized* form
# (post Repha/zero-anusvara/illegal-vowel cleanup), not the raw OCR token,
# because normalize_script's own deterministic repairs can legitimately
# span several raw character positions (e.g. one Repha digit standing in
# for two characters ರ್) despite being a single trusted, always-desired
# step -- measuring from the raw token would wrongly count that against the
# floor. Real single-glyph OCR damage on top of the normalized form (a
# dropped matra, a confused consonant) lands around 1.0; two compounding
# errors land around 2.0. Past that, the "best" candidate is usually just
# an unrelated word that happens to be reachable via a low base-cost
# generation step (e.g. an out-of-vocabulary proper noun with no dictionary
# entry) -- forcing it on is worse than leaving the OCR text untouched.
MAX_CORRECTION_EDIT_DISTANCE = 1.0


def heal_split_tokens(tokens: List[Dict[str, Any]], dictionary: Set[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Intelligently heal broken intra-word spaces caused by OCR segmentation gaps.
    e.g. ಹಿನ್ನೆ + ಲೆಯಲ್ಲಿ -> ಹಿನ್ನೆಲೆಯಲ್ಲಿ, ಸಾಧ್ಯ + ವಾಗಬೇಕಿದೆ -> ಸಾಧ್ಯವಾಗಬೇಕಿದೆ
    """
    if len(tokens) < 3:
        return tokens, []

    healed_corrections = []
    new_tokens = []
    i = 0

    while i < len(tokens):
        # Look for pattern: KannadaToken + single space + KannadaToken
        if (i + 2 < len(tokens) and 
            tokens[i]['type'] == 'kannada' and 
            tokens[i+1]['value'] == ' ' and 
            tokens[i+2]['type'] == 'kannada'):
            
            w1 = tokens[i]['value']
            w2 = tokens[i+2]['value']

            # A merge is only ever justified when the second half is a
            # *fragment*. This one condition gates all three cases below, and
            # it is the single most important guard in this function.
            #
            # A genuine OCR split leaves something that is not a word --
            # "ಲೆಯಲ್ಲಿ", "ವಾಗಬೇಕಿದೆ" -- which is exactly why rejoining them is
            # safe. If w2 stands on its own as a complete valid word, the space
            # before it is ordinary word spacing, however suffix-shaped w2
            # looks: "ಇದ್ದರೆ" is a real bound suffix elsewhere, but on its own
            # it is the plain standalone word "if [there] is". Merging there
            # destroys a real word boundary, which is strictly worse than
            # leaving a split alone -- a split word is visibly broken to
            # anything reading the corpus later, while a wrongly merged pair
            # reads as one plausible token and is effectively undetectable.
            #
            # Tested with is_valid_surface_word, not bare `in dictionary`.
            # Membership alone misses every inflected form the .dic does not
            # list literally, and that is what made this fire on good text:
            # measured over tests/fixtures/eval, "ಪರಿಚಯವು ಅವರು" was merged
            # because ಪರಿಚಯವು (= ಪರಿಚಯ + ವು) has no literal entry, and
            # "ಸ್ಪರ್ಶಿಸಿ ಅವರ" because Case B only looked at w1. Across 24 clean
            # pages this class of merge broke 50 words against 2 genuine fixes;
            # gating all three cases on it cut that to 10.
            w2_stands_alone = (
                w2 in dictionary
                or corpus_frequency(w2) >= HEAL_STANDALONE_FREQUENCY
            )
            if (w1 not in PROTECTED_TOKENS and w2 not in PROTECTED_TOKENS
                    and not w2_stands_alone):
                joined = w1 + w2
                should_join = False

                # Case A: Second word is a known suffix fragment -- except a
                # bare single-letter fragment (ಯ/ದ/ನ/ಅ/ಉ/ವು), which is
                # exactly as coincidence-prone as is_valid_surface_word's
                # own short-word suffix guard exists to prevent, regardless
                # of the merged result's length: these single letters are
                # also complete, independently correct Sanskrit particles/
                # pronouns in their own right (e.g. "ಚ ಯ", or a longer
                # phrase like "ಯಮದಾಹ್ಯೋ ಯ"), so bare membership in SUFFIXES
                # alone isn't enough evidence to merge them away.
                if (w2 in SUFFIX_SET or w2.startswith(('ವಾಗಿ', 'ವಾಗಲು', 'ವಾಗುವುದು', 'ವಾಗಬೇಕಿದೆ', 'ವಾಗಿದೆ', 'ವಾಗುತ್ತದೆ'))) \
                        and w2 not in ('ಯ', 'ದ', 'ನ', 'ಅ', 'ಉ', 'ವು'):
                    should_join = True
                # Case B: the joined form is itself a real word. Given the
                # outer guard has already established w2 cannot stand alone,
                # "fragment that completes a real word when reattached" is
                # about as strong as split evidence gets.
                #
                # This deliberately says nothing about w1. It used to require
                # w1 to be invalid, on the theory that a valid w1 means the
                # space is real -- but w1 is very often a valid word *prefix*
                # that is also independently valid (ಹಿನ್ನೆ in
                # "ಹಿನ್ನೆ ಲೆಯಲ್ಲಿ" -> ಹಿನ್ನೆಲೆಯಲ್ಲಿ), so that test rejected
                # exactly the splits this function exists to repair. The w2
                # guard is what carries the precision; w1 carries none.
                elif joined in dictionary or decompose_word(joined, dictionary)[0] is not None:
                    should_join = True

                if should_join:
                    healed_corrections.append({
                        'original': f"{w1} {w2}",
                        'correction': joined,
                        'edit_distance': 0.5,
                        'start': tokens[i]['start'],
                        'end': tokens[i+2]['end']
                    })
                    merged_tok = {
                        'value': joined,
                        'type': 'kannada',
                        'start': tokens[i]['start'],
                        'end': tokens[i+2]['end']
                    }
                    new_tokens.append(merged_tok)
                    i += 3
                    continue

        new_tokens.append(tokens[i])
        i += 1

    return new_tokens, healed_corrections


def is_valid_surface_word(w: str, dictionary: Set[str]) -> bool:
    """
    Check if the surface word token is structurally valid Kannada: an exact
    dictionary entry, a valid dictionary-root + suffix decomposition, or a
    recognized compound. This is a hard, curated-source gate -- real-corpus
    attestation alone is deliberately NOT checked here (see
    MIN_CORPUS_ATTESTATION / KEEP_ORIGINAL_BASE_COST in
    generate_kannada_candidates) because raw frequency can't reliably
    distinguish a genuinely rare real word from corpus noise/collisions, so
    it needs to compete as a scored candidate rather than short-circuit
    correction entirely.
    """
    if w in dictionary:
        return True
    for bs in BROKEN_SUFFIXES:
        if bs in w:
            return False
    root, suf = decompose_word(w, dictionary, exact_only=True)
    if root is not None and suf not in BROKEN_SUFFIXES:
        # Guard against greedy 1-letter suffix stripping on short words (e.g. ಆಳೆಯ != ಆಳೆ + ಯ)
        if len(w) <= 4 and suf in ('ಯ', 'ದ', 'ನ', 'ಅ', 'ಉ', 'ವು'):
            return False
        return True
    # is_compound_word's arbitrary two-way split (any i such that word[:i]
    # and word[i:] are each independently real dictionary words) gets
    # increasingly likely to fire by pure coincidence as the dictionary
    # grows -- most short Kannada syllable pairs ARE real words on their
    # own, so splitting almost any short word finds SOME dictionary/
    # dictionary pairing without the concatenation being a real compound
    # (e.g. ಪಥ "path" + ಮಾಷ "a lentil" structurally split ಪಥಮಾಷ, which
    # isn't a real word). Requiring the whole reconstructed compound to
    # itself have real corpus attestation is what actually distinguishes a
    # genuine compound from a coincidental split.
    #
    # Attestation is tested first purely for speed -- both operands are pure
    # so the conjunction is order-independent, but corpus_frequency is one
    # dict lookup while is_compound_word is an O(len(word) * 152) scan, and
    # it is the attestation half that rejects almost every candidate. This
    # ordering was 24% of total correction runtime.
    if corpus_frequency(w) >= MIN_CORPUS_ATTESTATION and is_compound_word(w, dictionary):
        return True
    return False


def resolve_valid_surface_form(cand: str, dictionary: Set[str]) -> Tuple[Optional[str], bool]:
    """
    Validate a candidate transformation against the dictionary and morphology
    engine, reconstructing the proper inflected surface word if needed.
    Returns (surface_form_or_None, is_fuzzy). is_fuzzy is True when the only
    match found requires decompose_word's vowel-ending-guessing fallback
    (the stripped root isn't literally in the dictionary -- only root+ಿ/ು/ಾ/ೆ
    happened to be). That's the same coincidence-prone shape as the blind
    edit-distance mechanisms below: a real dictionary word (e.g. the bound
    stem ಅನುಸರಿ) can happen to be what a guessed vowel ending lands on,
    silently turning an already-correct-but-undictionaried word (e.g.
    ಅನುಸರಣ, missing from the dictionary) into something unrelated once a
    glyph substitution elsewhere in the pipeline nudges the stem shape.
    Callers should treat a fuzzy resolution with the same reduced trust as
    'word_correction_unconstrained' -- see is_valid_surface_word's
    docstring for the same risk on the detection side.
    """
    if cand in dictionary:
        return cand, False
    root, suf = decompose_word(cand, dictionary, exact_only=True)
    if root is not None:
        return join_root_suffix(root, suf), False
    root, suf = decompose_word(cand, dictionary, exact_only=False)
    if root is not None:
        return join_root_suffix(root, suf), True
    # See is_valid_surface_word: is_compound_word's structural split alone
    # is too permissive to trust as a correction target -- require the
    # whole candidate to have real corpus attestation too. Attestation is
    # tested first for the same speed reason given there.
    if corpus_frequency(cand) >= MIN_CORPUS_ATTESTATION and is_compound_word(cand, dictionary):
        return cand, False
    return None, False


def collect_kannada_candidates(
    word: str,
    dictionary: Set[str]
) -> Optional[Dict[str, Tuple[float, str]]]:
    """
    Context-free half of candidate generation: every transformation of `word`
    that resolves to a real surface form, mapped to its base cost and the
    mechanism that produced it.

    Returns None if `word` is already a valid surface word (nothing to correct).

    Split out from generate_kannada_candidates because this half depends only
    on (word, dictionary) while the ranking half also depends on the
    surrounding words, and this is where essentially all the cost is: ~52
    dictionary/morphology resolutions per word against a handful of dict
    lookups for scoring. Measured over the eval fixtures, 62.7% of calls are
    for a word already seen, so the split is what makes caching possible.
    """
    # 1. Exact / Valid surface form check.
    #
    # Validity alone is not enough to stop here any more -- see
    # VALID_WORD_TRUST_FREQUENCY. A weakly-attested dictionary word carries on
    # into candidate generation and is defended by the frequency-dominance
    # check instead, because at 2.5M entries the dictionary contains OCR errors
    # and they would otherwise be uncorrectable by construction.
    if is_valid_surface_word(word, dictionary):
        if not has_corpus_counts() or corpus_frequency(word) >= VALID_WORD_TRUST_FREQUENCY:
            return None

    candidates: List[Tuple[str, float, str]] = []

    # 2. Universal Script Normalization (Repha regex + zero-anusvara)
    norm = normalize_script(word)
    if norm != word:
        res, fuzzy = resolve_valid_surface_form(norm, dictionary)
        if res:
            candidates.append((res, 0.20, 'word_correction_unconstrained' if fuzzy else 'ocr_repair'))

    w = norm

    # 3. Subscript & Multi-character Optical Ligature Substitutions (High precision)
    # The longest key in GLYPH_CONFUSIONS is 4 code points (ಸ್ಮೆ), so the
    # k_len=5 pass this loop used to start with could never match anything.
    for k_len in (4, 3, 2):
        for i in range(len(w) - k_len + 1):
            sub = w[i:i+k_len]
            if sub in GLYPH_CONFUSIONS:
                for rep, cost in GLYPH_CONFUSIONS[sub]:
                    cand = w[:i] + rep + w[i+k_len:]
                    res, fuzzy = resolve_valid_surface_form(cand, dictionary)
                    if res:
                        candidates.append((res, cost, 'word_correction_unconstrained' if fuzzy else 'ocr_repair'))

    # 4. Suffix normalizations (e.g. ಟ್ಟತ್ತು -> ಟ್ಟಿತ್ತು, ವಳ್ಳು -> ವಳು, ಸಂಯುವ -> ಸಾಯುವ, ುತದೆ -> ುತ್ತದೆ)
    for bad_suf, good_suf in SUFFIX_NORMALIZATIONS.items():
        if bad_suf in w:
            cand = w.replace(bad_suf, good_suf)
            res, fuzzy = resolve_valid_surface_form(cand, dictionary)
            if not res and w.endswith(bad_suf) and len(w) > len(bad_suf):
                stem = w[:-len(bad_suf)]
                if stem + 'ು' in dictionary:
                    res = join_root_suffix(stem + 'ು', good_suf)
                else:
                    r_test, s_test = decompose_word(stem + 'ು', dictionary, exact_only=True)
                    if r_test is None:
                        r_test, s_test = decompose_word(stem + 'ು', dictionary, exact_only=False)
                        fuzzy = r_test is not None
                    if r_test is not None:
                        res = join_root_suffix(stem + 'ು', good_suf)
            if res:
                candidates.append((res, 0.20, 'word_correction_unconstrained' if fuzzy else 'word_correction'))

    # 5. Single-char Optical Glyph Substitutions
    for i in range(len(w)):
        sub = w[i]
        if sub in GLYPH_CONFUSIONS:
            for rep, cost in GLYPH_CONFUSIONS[sub]:
                cand = w[:i] + rep + w[i+1:]
                res, fuzzy = resolve_valid_surface_form(cand, dictionary)
                if res:
                    candidates.append((res, cost, 'word_correction_unconstrained' if fuzzy else 'ocr_repair'))

    # 6. Systematic Vowel Length Transformations (ಹ್ರಸ್ವ <-> ದೀರ್ಘ)
    for i, ch in enumerate(w):
        if ch in VOWEL_MATRA_MAP:
            cand = w[:i] + VOWEL_MATRA_MAP[ch] + w[i+1:]
            res, fuzzy = resolve_valid_surface_form(cand, dictionary)
            if res:
                candidates.append((res, 0.25, 'word_correction_unconstrained' if fuzzy else 'word_correction'))

    # 7. Stem-level substitutions for inflected words (e.g., ದೂಹಿಸಿದರು -> ದೂಷಿಸಿದರು)
    for suf in (suffixes_ending_with(w[-1]) if w else ()):
        if w.endswith(suf) and len(w) > len(suf) + 1:
            stem = w[:-len(suf)]
            for k_len in (3, 2, 1):
                for j in range(len(stem) - k_len + 1):
                    sub = stem[j:j+k_len]
                    if sub in GLYPH_CONFUSIONS:
                        for rep, cost in GLYPH_CONFUSIONS[sub]:
                            rep_stem = stem[:j] + rep + stem[j+k_len:]
                            res, fuzzy = resolve_valid_surface_form(join_root_suffix(rep_stem, suf), dictionary)
                            if not res:
                                res, fuzzy = resolve_valid_surface_form(join_root_suffix(rep_stem + 'ು', suf), dictionary)
                            if res:
                                candidates.append((res, cost, 'word_correction_unconstrained' if fuzzy else 'ocr_repair'))
            break

    # 8. Virama (Halant) Insertion between Adjacent Consonants
    for i in range(len(w) - 1):
        if w[i] in KANNADA_CONSONANTS and w[i+1] in KANNADA_CONSONANTS:
            cand = w[:i+1] + '್' + w[i+1:]
            res, fuzzy = resolve_valid_surface_form(cand, dictionary)
            if res:
                candidates.append((res, 0.30, 'word_correction_unconstrained' if fuzzy else 'ocr_repair'))
            for ext in ['ೆ', 'ಾ', 'ು', 'ಣ', 'ಮಿ', 'ಿ']:
                scand = cand + ext
                sres, sfuzzy = resolve_valid_surface_form(scand, dictionary)
                if sres:
                    candidates.append((sres, 0.40, 'word_correction_unconstrained' if sfuzzy else 'ocr_repair'))

    # 9. Anusvara Noise Speckle Removal -- same unconstrained-mechanism
    # caveat as item 10 below: a trailing anusvara is also the ordinary
    # Sanskrit case ending on a word whose bare-stem form is a common
    # Kannada loanword (ಅಕ್ಷರಂ -> ಅಕ್ಷರ, ಮಿಶ್ರಂ -> ಮಿಶ್ರ), so this can't be
    # trusted on frequency dominance alone either.
    if 'ಂ' in w:
        cand = w.replace('ಂ', '')
        res, _fuzzy = resolve_valid_surface_form(cand, dictionary)
        if res:
            candidates.append((res, 0.35, 'word_correction_unconstrained'))

    # 10. Universal 1-Edit Deletion & Boundary Insertion (Handles OCR truncation or speckle insertions)
    #
    # These two loops are the least-constrained candidate generators in the
    # whole pipeline: "delete any single character" / "append any
    # consonant" has no glyph-similarity or morphological grounding at
    # all -- it only asks whether the result happens to land on a
    # dictionary word. That's exactly why it also fires on genuine Sanskrit
    # case-ending variation (visarga/anusvara/vocalic endings absent from
    # the Kannada-only dictionary) as if it were OCR noise: deleting one
    # character of e.g. "ಭಾವಃ" or "ಇಮೇ" happens to land on the common
    # Kannada word "ಭಾವ"/"ಮೇ", and frequency-dominance alone can't tell
    # that apart from real noise -- a rare real word being outnumbered by
    # its own common near-neighbor is the *identical* statistical
    # signature either way. Tagged 'word_correction_unconstrained' (mapped
    # back to 'word_correction' before being returned to callers) so
    # suggest_kannada_word can require actual bigram context support for
    # just these two mechanisms, not the higher-precision ones below.
    if len(w) >= 3:
        ax = aksharas(w)

        # Whole-akshara deletion (a spurious glyph read out of a speckle).
        # Deleting a *code point* here used to be able to remove a base
        # consonant and leave its matra stranded -- "ಕಾಫಿ" minus code point 0
        # is "ಾಫಿ", which no OCR engine could have produced and no dictionary
        # can contain. Those lookups were pure waste.
        for i in range(len(ax)):
            cand = ''.join(ax[:i] + ax[i+1:])
            res, _fuzzy = resolve_valid_surface_form(cand, dictionary)
            if res:
                candidates.append((res, 0.40, 'word_correction_unconstrained'))

        # Dependent-sign removal: keep the akshara's base, drop the marks
        # hanging off it. This is the other half of what code-point deletion
        # used to cover -- a speckle misread as a matra, or a spurious virama
        # splitting a consonant -- but expressed so the result is always a
        # well-formed cluster.
        for i, cl in enumerate(ax):
            if len(cl) > 1:
                cand = ''.join(ax[:i] + [cl[0]] + ax[i+1:])
                res, _fuzzy = resolve_valid_surface_form(cand, dictionary)
                if res:
                    candidates.append((res, 0.40, 'word_correction_unconstrained'))

        # Terminal single-character insertion (handles truncated word endings)
        for ch in KANNADA_CONSONANTS:
            cand = w + ch
            res, _fuzzy = resolve_valid_surface_form(cand, dictionary)
            if res:
                candidates.append((res, 0.35, 'word_correction_unconstrained'))

        # Dropped diacritic: attach a vowel sign to an akshara. These are
        # optically tiny marks and the single most common thing OCR silently
        # loses.
        #
        # Attached per cluster rather than inserted at an arbitrary code-point
        # offset, which used to generate impossible strings -- a matra before
        # the consonant it modifies, or stacked after a virama -- at every
        # position in the word.
        #
        # KNOWN INCONSISTENCY, left in place deliberately. This mechanism can
        # still emit 'ocr_repair', which skips the bigram gate, even though a
        # 13-sign x every-akshara enumeration has none of the optical grounding
        # the item-10 comment above says that trust requires. Tagging it
        # unconstrained was tried and measured: on 300 synthetic lines it cost
        # 14 fixes (193 -> 179) to avoid 2 breaks (146 -> 144), dropping
        # precision 0.569 -> 0.554, and changed nothing on the 24 real-OCR
        # pages. Both metrics available say the stricter tag is the worse
        # trade, so the argument for it is currently consistency alone.
        #
        # Note the synthetic set is biased FOR this mechanism -- corrupt_line
        # drops matras deliberately -- so it flatters the loose tag. Settling
        # this needs real scanned pages with transcripts, which the repo has
        # none of. Do not "fix" the inconsistency without that evidence.
        for i, cl in enumerate(ax):
            if cl.endswith('್'):
                continue
            for vs in KANNADA_VOWEL_SIGNS:
                cand = ''.join(ax[:i] + [cl + vs] + ax[i+1:])
                res, fuzzy = resolve_valid_surface_form(cand, dictionary)
                if res:
                    candidates.append((res, 0.35, 'word_correction_unconstrained' if fuzzy else 'ocr_repair'))

    # 11. Deduplicate candidates, keeping the cheapest route to each one.
    candidate_dict: Dict[str, Tuple[float, str]] = {}
    for cand, base_cost, ctype in candidates:
        if cand not in candidate_dict or base_cost < candidate_dict[cand][0]:
            candidate_dict[cand] = (base_cost, ctype)

    return candidate_dict


# Cache for collect_kannada_candidates, keyed on the word alone -- valid
# because the function's only other input, the dictionary, is loaded once per
# process (init_pipeline is idempotent under a lock). The two paths that can
# replace it -- init_pipeline itself and correct_text's lazy-load fallback --
# both call clear_correction_caches(), and so must any test that swaps
# dictionaries mid-process.
#
# Capped rather than unbounded: a full book run sees a lot of distinct words,
# and each entry holds a small dict of candidate strings. Past the cap new
# words simply go uncached, which costs speed and nothing else.
_candidate_cache: Dict[str, Optional[Dict[str, Tuple[float, str]]]] = {}
_CANDIDATE_CACHE_LIMIT = 200_000


def clear_correction_caches() -> None:
    """Drop memoized candidate generation. Call after loading a dictionary."""
    _candidate_cache.clear()


def generate_kannada_candidates(
    word: str,
    dictionary: Set[str],
    prev_word: Optional[str] = None,
    next_word: Optional[str] = None
) -> List[Tuple[str, float, str]]:
    """
    Dynamically generate and rank valid candidate words for an unknown Kannada word token.
    Combines: Script Normalization -> Suffix Healing -> Vowel Permutation ->
              Optical Glyph Substitutions -> Virama/Ottu Transforms -> LM Context Scoring.
    Returns: List of (candidate_word, final_score, correction_type) sorted by best score.
    """
    try:
        candidate_dict = _candidate_cache[word]
    except KeyError:
        candidate_dict = collect_kannada_candidates(word, dictionary)
        if len(_candidate_cache) < _CANDIDATE_CACHE_LIMIT:
            _candidate_cache[word] = candidate_dict

    if candidate_dict is None:
        return [(word, 0.0, 'none')]

    ranked: List[Tuple[str, float, str]] = []
    for cand, (base_cost, ctype) in candidate_dict.items():
        # score_candidate already returns a bounded [0.0, 0.3] bonus.
        lm_bonus = score_candidate(cand, prev_word, next_word)
        final_score = round(base_cost - lm_bonus, 3)
        ranked.append((cand, final_score, ctype))

    # Stable sort, so score ties keep candidate_dict's insertion order -- which
    # is the generator order in collect_kannada_candidates, cached or not.
    # That ordering is load-bearing and undocumented; Phase 3d addresses it.
    ranked.sort(key=lambda x: x[1])
    return ranked


def suggest_kannada_word(word: str, prev_word: Optional[str] = None, next_word: Optional[str] = None) -> Tuple[str, float, str]:
    """
    Find best correction suggestion for a single Kannada word.
    Returns: (corrected_word, edit_distance, correction_type)
    Types: 'ocr_repair' (Blue), 'word_correction' (Green), 'hybrid' (Yellow), 'none'
    """
    dictionary = get_dictionary()

    if not dictionary:
        return word, 0.0, 'none'

    # 1. Single character or short initials (e.g. ಶ್ರೀ, ಡಾ, ಎ, ವಿ, ಎಸ್) - never corrupt
    if len(word) <= 1:
        return word, 0.0, 'none'

    # 2. Dynamic candidate generation
    candidates = generate_kannada_candidates(word, dictionary, prev_word, next_word)

    if candidates:
        top_cand, score, corr_type = candidates[0]
        if top_cand != word and corr_type != 'none':
            dist = weighted_edit_distance(word, top_cand)
            norm_dist = weighted_edit_distance(normalize_script(word), top_cand)
            if norm_dist > MAX_CORRECTION_EDIT_DISTANCE:
                return word, 0.0, 'none'

            is_unconstrained = corr_type == 'word_correction_unconstrained'
            display_type = 'word_correction' if is_unconstrained else corr_type

            # Is the destination a word worth rewriting INTO at all? This is
            # a property of the candidate alone, independent of how it was
            # reached, so it gates every mechanism -- grounded and
            # unconstrained alike -- before any of the evidence tests below.
            #
            # The extended dictionary is 141,115 entries of which 70,778 have
            # zero corpus attestation, and every one of those is somewhere for
            # the edit search to land. Measured on tests/fixtures/eval, the
            # only remaining broken word after the space-merge fix was
            # ಪಾರಲೌಕಿಕರಿಗೆ -> ಪಾರಲೌಕಿಕದಿಗೆ, a ರ/ದ glyph swap into a form with
            # frequency 0. See dictionary.is_correction_target.
            cand_freq = corpus_frequency(top_cand)
            # Skipped outright when no real corpus model is loaded: without one
            # every word carries add_vocabulary's padded count of 1, so a
            # frequency floor would reject the entire vocabulary and disable
            # correction rather than merely tightening it.
            if has_corpus_counts() and not is_correction_target(top_cand, cand_freq):
                return word, 0.0, 'none'

            # If the original word has (almost) no real corpus attestation,
            # frequency dominance can't gate ANY mechanism -- not just the
            # unconstrained ones. A high-precision glyph-confusion
            # substitution is grounded in real optical similarity, but that
            # doesn't mean this specific application is safe: OCR debris
            # from a truncated/dropped run of characters (not a real word
            # attempt at all) can still land one clean glyph-swap away from
            # an unrelated common word (e.g. a truncated "ರಾಮಾಯಣ" fragment
            # "ಯಣ" -> the common word "ಹಣ", "money") with no relationship
            # to the original whatsoever. Require real local context
            # support instead -- does the candidate actually co-occur with
            # its neighbor in the corpus -- falling back to an absolute
            # commonness floor only for the already-flagged-risky
            # unconstrained mechanisms when there's no neighboring Kannada
            # word to check bigram support against at all (a candidate
            # reached only by exact dictionary/glyph match, in a genuinely
            # context-free call, is left as-is to avoid regressing the
            # normal single-word correction case).
            word_freq = corpus_frequency(word)

            if is_unconstrained:
                # Unconstrained mechanisms need real context support
                # UNCONDITIONALLY, regardless of the original word's own
                # attestation -- this must stay independent of the
                # word_freq check below, not folded into the same
                # if/else: a rare-but-attested word (e.g. "ಭಾವಃ", freq 106)
                # outnumbered 300x+ by its common bare-stem neighbor
                # ("ಭಾವ") still clears the dominance ratio below on
                # frequency alone, which is exactly the Sanskrit
                # case-ending bug from earlier in this session -- the
                # dominance check was never capable of catching that on
                # its own, which is the entire reason this bigram check
                # exists.
                if prev_word or next_word:
                    if not has_bigram_support(top_cand, prev_word, next_word):
                        return word, 0.0, 'none'
                elif cand_freq < MIN_CANDIDATE_ATTESTATION_FOR_UNSEEN_ORIGINAL:
                    return word, 0.0, 'none'
            # Non-unconstrained candidates (is_unconstrained False) are
            # already structurally validated, not just frequency-plausible:
            # `word` already failed is_valid_surface_word to even reach
            # candidate generation, and `top_cand` was confirmed via a
            # NON-fuzzy dictionary/morphology match reached through a
            # grounded transform (optical glyph confusion, script
            # normalization, suffix pattern, virama insertion -- see the
            # fuzzy ternary in every generate_kannada_candidates item).
            # Bigram co-occurrence on top of that is systematically
            # miscalibrated: a correct RARE inflected form of a common root
            # (Kannada's agglutinative morphology produces huge numbers of
            # these) essentially never satisfies a literal bigram-adjacency
            # check just because it's rare, not because it's wrong -- e.g.
            # real "ದೂಷಿಸಿದವಳೂ" (corpus freq 1, resolved from OCR-garbled
            # "ದೂಹಿಸಿದವಳೂ" via item 7's stem-level glyph substitution)
            # succeeds in isolation but was wrongly blocked whenever real
            # sentence context was present. The frequency-dominance check
            # just below remains the safety net for these mechanisms once
            # the original word is itself attested enough to plausibly be
            # its own rare-but-real word.

            if word_freq >= MIN_CORPUS_ATTESTATION:
                if cand_freq < word_freq * FREQUENCY_DOMINANCE_RATIO:
                    return word, 0.0, 'none'
            return top_cand, dist, display_type

    # 3. Default Safe Fallback: Preserve original OCR text without corrupting
    return word, 0.0, 'none'


def correct_text(
    text: str,
    allowed_types: Optional[Set[str]] = None,
    word_confidences: Optional[List[Tuple[str, float]]] = None
) -> Dict[str, Any]:
    """
    Correct a full block of mixed Kannada text.
    Combines Unicode normalization, space healing, dynamic Indic transforms, and N-gram ranking.

    allowed_types, if given, restricts which correction *types*
    ('ocr_repair' / 'word_correction') are actually applied to the output
    text -- every correction is still found and reported in `corrections`
    regardless, only whether it's written into `corrected` is gated. This
    exists for ablation/diagnostic runs (e.g. "what does the pipeline look
    like with only glyph/script-level repairs applied, no dictionary-based
    word correction"); normal callers should leave it as None.

    word_confidences, if given, is the (word_text, tesseract_confidence)
    list for this line's OCR output (see ocr_image_with_layout's
    'word_confidences'). A word whose own OCR confidence clears
    HIGH_OCR_CONFIDENCE_TRUST is left untouched outright -- see that
    constant's docstring for why this is trusted as a real signal, not
    just corpus-frequency guesswork.
    """
    dictionary = get_dictionary()
    if not dictionary:
        load_dictionary()
        train_from_word_list(get_word_list())
        clear_correction_caches()
        # Re-read: load_and_expand_dic rebinds the module global rather than
        # mutating it, so the local `dictionary` above is still the empty set
        # that triggered this branch, and heal_split_tokens below would have
        # been handed it. Unreachable in practice -- every real entry point
        # runs init_pipeline first -- but wrong as written.
        dictionary = get_dictionary()

    conf_by_word: Dict[str, float] = {}
    if word_confidences:
        for w, c in word_confidences:
            conf_by_word[w] = c

    # Pre-pass: clean Unicode zero-digit / anusvara scanning artifacts & universal Repha
    pre_cleaned = normalize_script(text)

    tokens = tokenize(pre_cleaned)

    # Pre-pass 2: Heal split word gaps (Treated as OCR repairs)
    if allowed_types is None or 'ocr_repair' in allowed_types:
        tokens, space_corrections = heal_split_tokens(tokens, dictionary)
    else:
        space_corrections = []
    for sc in space_corrections:
        sc['type'] = 'ocr_repair'

    corrections = list(space_corrections)
    kannada_tokens = [t for t in tokens if t['type'] == 'kannada']
    kannada_count = len(kannada_tokens)

    for i, token in enumerate(tokens):
        # Only check Kannada words (leave numbers, punctuation, and English strictly untouched)
        if token['type'] != 'kannada':
            continue

        orig_word = token['value']

        if conf_by_word.get(orig_word, -1) >= HIGH_OCR_CONFIDENCE_TRUST:
            continue

        # Context extraction
        prev_kannada = None
        next_kannada = None
        for prev_t in reversed(tokens[:i]):
            if prev_t['type'] == 'kannada':
                prev_kannada = prev_t['value']
                break
        for next_t in tokens[i+1:]:
            if next_t['type'] == 'kannada':
                next_kannada = next_t['value']
                break

        corrected_word, dist, corr_type = suggest_kannada_word(orig_word, prev_kannada, next_kannada)

        if corrected_word != orig_word and corr_type != 'none':
            corrections.append({
                'original': orig_word,
                'correction': corrected_word,
                'edit_distance': round(dist, 2),
                'type': corr_type, # 'ocr_repair' (Blue), 'word_correction' (Green), or 'hybrid' (Yellow)
                'start': token['start'],
                'end': token['end']
            })
            if allowed_types is None or corr_type in allowed_types:
                token['value'] = corrected_word

    corrected_text = reconstruct(tokens)
    return {
        'original': text,
        'corrected': corrected_text,
        'has_errors': len(corrections) > 0,
        'total_words': kannada_count,
        'total_corrections': len(corrections),
        'accuracy_rate': round((1.0 - (len(corrections) / (kannada_count or 1))) * 100, 1),
        'corrections': corrections
    }



# Below this mean OCR confidence, a line isn't degraded prose -- it's OCR
# hallucinating text rows out of non-text content (decorative banner art,
# logos, photos). Validated against 7 real pages covering the hard real
# cases (faded scans, heavy Sanskrit vocabulary the dictionary doesn't
# cover, degraded photocopies): every genuine body-text line across all of
# them still cleared 55 mean confidence. Only an actual cover/poster page's
# graphic regions dropped below it, down into single digits. 40 leaves a
# comfortable margin below every observed real-prose case while catching
# that non-text one. Deliberately confidence-only, not combined with a
# dictionary valid-word-ratio check: ratio alone misfires on exactly the
# Sanskrit-vocabulary case above (real text, ~0% dictionary hits, but
# completely normal confidence), so it can't be used as a safe secondary
# gate here. Never applies to PDF-digital-extraction lines (line.get('conf')
# is None for those -- they were never OCR'd from pixels at all).
NON_TEXT_LINE_CONFIDENCE = 40.0


# Second, independent non-text signal: a line that is mostly Latin letters on a
# monolingual Kannada page is not text this pipeline can use. It is not merely
# wrong, it is unrecoverable -- the tokenizer hands non-Kannada tokens through
# untouched by design, so nothing downstream will ever repair or even flag it.
#
# For the Tesseract path this is a NO-OP and was measured to be one: with
# --lang kan the engine has no Latin characters to emit, and adding this filter
# left all nine real pages bit-identical (CER 0.0558 either way). It exists for
# Surya, which is a general multilingual model with no language restriction and
# hallucinates fluent-looking Latin out of graphic content:
#
#   conf 56.4  'terrare and the second the second'
#   conf 40.6  'Carl State State'
#
# On page 08 that alone accounted for the entire difference between Surya
# looking far worse than Tesseract (CER 0.1909) and far better (0.0065).
# Confidence could not catch it -- those lines score 40-60, overlapping the
# range real degraded prose occupies -- but script does, unambiguously.
#
# Counting characters rather than tokens on purpose: a genuine Kannada line
# quoting an English title ("(Skylark)", which really does occur in this
# corpus) stays majority-Kannada by character count and survives.
_LATIN_CHAR_RE = re.compile(r'[A-Za-z]')
_KANNADA_CHAR_RE = re.compile(r'[\u0C80-\u0CFF]')


def is_latin_majority(text: str) -> bool:
    """True if a line has more Latin letters than Kannada ones."""
    latin = len(_LATIN_CHAR_RE.findall(text))
    kannada = len(_KANNADA_CHAR_RE.findall(text))
    return latin > kannada


def correct_layout_lines(layout_lines: List[Dict[str, Any]], allowed_types: Optional[Set[str]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Correct structured layout lines preserving bounding box / alignment tags.
    See correct_text for what allowed_types does. Automatically forwards
    each line's own 'word_confidences' (populated by ocr_image_with_layout)
    into correct_text if present -- callers don't need to opt in separately.

    Each output line also carries 'is_likely_non_text' (see
    NON_TEXT_LINE_CONFIDENCE) so callers building a clean training corpus
    can exclude OCR hallucination on graphic content without discarding it
    outright -- the line is still returned here for audit/visual fidelity,
    just flagged.
    """
    corrected_lines = []
    all_corrections = []

    for line in layout_lines:
        res = correct_text(line.get('text', ''), allowed_types=allowed_types, word_confidences=line.get('word_confidences'))
        conf = line.get('conf')
        corrected_lines.append({
            'text': res['corrected'],
            'original_text': line.get('text', ''),
            'alignment': line.get('alignment', 'L'),
            'top': line.get('top', 0),
            'left': line.get('left', 0),
            'width': line.get('width', 0),
            'height': line.get('height', 0),
            'page_num': line.get('page_num', 1),
            'ocr_confidence': conf,
            'is_likely_non_text': (
                (conf is not None and conf < NON_TEXT_LINE_CONFIDENCE)
                or is_latin_majority(line.get('text', ''))
            )
        })
        all_corrections.extend(res['corrections'])

    return corrected_lines, all_corrections

