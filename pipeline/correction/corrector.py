"""
Main Kannada Autocorrect & Post-OCR Correction Engine
Combines: Universal Script Normalization -> Space Healing -> Dynamic Indic Candidate Generation -> N-Gram Context Ranking
"""

import re
from typing import Dict, List, Tuple, Any, Optional, Set
from .dictionary import get_dictionary, get_word_list, load_dictionary
from .tokenizer import tokenize, reconstruct
from .edit_distance import weighted_edit_distance, GLYPH_CONFUSIONS
from .morphology import (
    decompose_word,
    join_root_suffix,
    is_compound_word,
    SUFFIXES,
    SUFFIX_NORMALIZATIONS,
    BROKEN_SUFFIXES
)
from .ocr_repairs import normalize_script, clean_unicode_glitches
from .ngram import train_model, score_candidate

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

            if w1 not in PROTECTED_TOKENS and w2 not in PROTECTED_TOKENS:
                joined = w1 + w2
                should_join = False

                # Case A: Second word is a known suffix fragment
                if w2 in SUFFIXES or w2.startswith(('ವಾಗಿ', 'ವಾಗಲು', 'ವಾಗುವುದು', 'ವಾಗಬೇಕಿದೆ', 'ವಾಗಿದೆ', 'ವಾಗುತ್ತದೆ')):
                    should_join = True
                # Case B: First word is not a valid word, but joined form IS a valid dictionary word
                elif w1 not in dictionary and (joined in dictionary or decompose_word(joined, dictionary)[0] is not None):
                    should_join = True
                # Case C: Joint compound word is in dictionary
                elif joined in dictionary:
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
    """Check if the surface word token is a completely valid, uncorrupted Kannada word."""
    if w in dictionary:
        return True
    for bs in BROKEN_SUFFIXES:
        if bs in w:
            return False
    root, suf = decompose_word(w, dictionary)
    if root is not None and suf not in BROKEN_SUFFIXES:
        # Guard against greedy 1-letter suffix stripping on short words (e.g. ಆಳೆಯ != ಆಳೆ + ಯ)
        if len(w) <= 4 and suf in ('ಯ', 'ದ', 'ನ', 'ಅ', 'ಉ', 'ವು'):
            return False
        return True
    if is_compound_word(w, dictionary):
        return True
    return False


def resolve_valid_surface_form(cand: str, dictionary: Set[str]) -> Optional[str]:
    """
    Validate a candidate transformation against the dictionary and morphology engine,
    reconstructing the proper inflected surface word if needed.
    """
    if cand in dictionary:
        return cand
    root, suf = decompose_word(cand, dictionary)
    if root is not None:
        return join_root_suffix(root, suf)
    if is_compound_word(cand, dictionary):
        return cand
    return None


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
    # 1. Exact / Valid surface form check
    if is_valid_surface_word(word, dictionary):
        return [(word, 0.0, 'none')]

    candidates: List[Tuple[str, float, str]] = []

    # 2. Universal Script Normalization (Repha regex + zero-anusvara)
    norm = normalize_script(word)
    if norm != word:
        res = resolve_valid_surface_form(norm, dictionary)
        if res:
            candidates.append((res, 0.20, 'ocr_repair'))

    w = norm

    # 3. Subscript & Multi-character Optical Ligature Substitutions (High precision)
    for k_len in (5, 4, 3, 2):
        for i in range(len(w) - k_len + 1):
            sub = w[i:i+k_len]
            if sub in GLYPH_CONFUSIONS:
                for rep, cost in GLYPH_CONFUSIONS[sub]:
                    cand = w[:i] + rep + w[i+k_len:]
                    res = resolve_valid_surface_form(cand, dictionary)
                    if res:
                        candidates.append((res, cost, 'ocr_repair'))

    # 4. Suffix normalizations (e.g. ಟ್ಟತ್ತು -> ಟ್ಟಿತ್ತು, ವಳ್ಳು -> ವಳು, ಸಂಯುವ -> ಸಾಯುವ, ುತದೆ -> ುತ್ತದೆ)
    for bad_suf, good_suf in SUFFIX_NORMALIZATIONS.items():
        if bad_suf in w:
            cand = w.replace(bad_suf, good_suf)
            res = resolve_valid_surface_form(cand, dictionary)
            if not res and w.endswith(bad_suf) and len(w) > len(bad_suf):
                stem = w[:-len(bad_suf)]
                r_test, s_test = decompose_word(stem + 'ು', dictionary)
                if r_test or stem + 'ು' in dictionary:
                    res = join_root_suffix(stem + 'ು', good_suf)
            if res:
                candidates.append((res, 0.20, 'word_correction'))

    # 5. Single-char Optical Glyph Substitutions
    for i in range(len(w)):
        sub = w[i]
        if sub in GLYPH_CONFUSIONS:
            for rep, cost in GLYPH_CONFUSIONS[sub]:
                cand = w[:i] + rep + w[i+1:]
                res = resolve_valid_surface_form(cand, dictionary)
                if res:
                    candidates.append((res, cost, 'ocr_repair'))

    # 6. Systematic Vowel Length Transformations (ಹ್ರಸ್ವ <-> ದೀರ್ಘ)
    for i, ch in enumerate(w):
        if ch in VOWEL_MATRA_MAP:
            cand = w[:i] + VOWEL_MATRA_MAP[ch] + w[i+1:]
            res = resolve_valid_surface_form(cand, dictionary)
            if res:
                candidates.append((res, 0.25, 'word_correction'))

    # 7. Stem-level substitutions for inflected words (e.g., ದೂಹಿಸಿದರು -> ದೂಷಿಸಿದರು)
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) > len(suf) + 1:
            stem = w[:-len(suf)]
            for k_len in (3, 2, 1):
                for j in range(len(stem) - k_len + 1):
                    sub = stem[j:j+k_len]
                    if sub in GLYPH_CONFUSIONS:
                        for rep, cost in GLYPH_CONFUSIONS[sub]:
                            rep_stem = stem[:j] + rep + stem[j+k_len:]
                            res = resolve_valid_surface_form(join_root_suffix(rep_stem, suf), dictionary)
                            if not res:
                                res = resolve_valid_surface_form(join_root_suffix(rep_stem + 'ು', suf), dictionary)
                            if res:
                                candidates.append((res, cost, 'ocr_repair'))
            break

    # 8. Virama (Halant) Insertion between Adjacent Consonants
    for i in range(len(w) - 1):
        if w[i] in KANNADA_CONSONANTS and w[i+1] in KANNADA_CONSONANTS:
            cand = w[:i+1] + '್' + w[i+1:]
            res = resolve_valid_surface_form(cand, dictionary)
            if res:
                candidates.append((res, 0.30, 'ocr_repair'))
            for ext in ['ೆ', 'ಾ', 'ು', 'ಣ', 'ಮಿ', 'ಿ']:
                scand = cand + ext
                sres = resolve_valid_surface_form(scand, dictionary)
                if sres:
                    candidates.append((sres, 0.40, 'ocr_repair'))

    # 9. Anusvara Noise Speckle Removal
    if 'ಂ' in w:
        cand = w.replace('ಂ', '')
        res = resolve_valid_surface_form(cand, dictionary)
        if res:
            candidates.append((res, 0.35, 'ocr_repair'))

    # 10. Deduplicate candidates and score with weighted Levenshtein & N-gram Language Model
    candidate_dict: Dict[str, Tuple[float, str]] = {}
    for cand, base_cost, ctype in candidates:
        if cand not in candidate_dict or base_cost < candidate_dict[cand][0]:
            candidate_dict[cand] = (base_cost, ctype)

    ranked: List[Tuple[str, float, str]] = []
    for cand, (base_cost, ctype) in candidate_dict.items():
        lm_bonus = 0.0
        if prev_word or next_word:
            lm_score = score_candidate(cand, prev_word, next_word)
            # Normalize LM score into an additive bonus [0.0, 0.3]
            lm_bonus = max(0.0, min(0.3, (lm_score + 10.0) / 20.0))

        final_score = round(base_cost - lm_bonus, 3)
        ranked.append((cand, final_score, ctype))

    ranked.sort(key=lambda x: x[1])
    return ranked

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
            return top_cand, dist, corr_type

    # 3. Default Safe Fallback: Preserve original OCR text without corrupting
    return word, 0.0, 'none'


def correct_text(text: str) -> Dict[str, Any]:
    """
    Correct a full block of mixed Kannada text.
    Combines Unicode normalization, space healing, dynamic Indic transforms, and N-gram ranking.
    """
    dictionary = get_dictionary()
    if not dictionary:
        load_dictionary()
        train_model(get_word_list())

    # Pre-pass: clean Unicode zero-digit / anusvara scanning artifacts & universal Repha
    pre_cleaned = normalize_script(text)

    tokens = tokenize(pre_cleaned)

    # Pre-pass 2: Heal split word gaps (Treated as OCR repairs)
    tokens, space_corrections = heal_split_tokens(tokens, dictionary)
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


def correct_layout_lines(layout_lines: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Correct structured layout lines preserving bounding box / alignment tags.
    """
    corrected_lines = []
    all_corrections = []

    for line in layout_lines:
        res = correct_text(line.get('text', ''))
        corrected_lines.append({
            'text': res['corrected'],
            'original_text': line.get('text', ''),
            'alignment': line.get('alignment', 'L'),
            'top': line.get('top', 0),
            'left': line.get('left', 0),
            'width': line.get('width', 0),
            'height': line.get('height', 0),
            'page_num': line.get('page_num', 1)
        })
        all_corrections.extend(res['corrections'])

    return corrected_lines, all_corrections

