"""
Main Kannada Autocorrect & Post-OCR Correction Engine
Combines: OCR Rule Repairs -> Morphological Decomposition -> Compound Word (Samasa) Verification -> Safe Conservative Edit Distance
"""

from typing import Dict, List, Tuple, Any, Optional
from .dictionary import get_dictionary, get_word_list, load_dictionary
from .tokenizer import tokenize, reconstruct
from .edit_distance import weighted_edit_distance
from .morphology import decompose_word, join_root_suffix, is_compound_word, SUFFIXES
from .ocr_repairs import apply_ocr_repairs
from .ngram import train_model, score_candidate

# Strict maximum edit distance for conservative substitution
MAX_EDIT_DIST = 1.0


def suggest_kannada_word(word: str, prev_word: Optional[str] = None, next_word: Optional[str] = None) -> Tuple[str, float]:
    """
    Find best correction suggestion for a single Kannada word.
    Guaranteed not to corrupt valid proper nouns, compounds, or initials.
    """
    dictionary = get_dictionary()
    word_list = get_word_list()

    if not dictionary:
        return word, 0.0

    # 1. Single character or short initials (e.g. ಶ್ರೀ, ಡಾ, ಎ, ವಿ, ಎಸ್) - never corrupt
    if len(word) <= 1:
        return word, 0.0

    # 2. Exact match in dictionary
    if word in dictionary:
        return word, 0.0

    # 3. Direct morphological decomposition (root + valid Kannada suffix)
    root, suf = decompose_word(word, dictionary)
    if root is not None:
        joined = join_root_suffix(root, suf)
        if joined != word:
            return joined, 0.5
        return word, 0.0

    # 4. Compound Word (Samasa) validation (e.g., ಸಂತಕವಿ, ಸಹಕಾರ, ಕನಕದಾಸರು, ಮರುಓದಿಗೆ)
    if is_compound_word(word, dictionary):
        return word, 0.0

    # 5. Rule-based OCR Repairs (Missing Halant/Virama, vowel sign optical errors)
    repaired = apply_ocr_repairs(word, dictionary)
    if repaired != word:
        if repaired in dictionary or is_compound_word(repaired, dictionary):
            return repaired, 0.5
        rep_root, rep_suf = decompose_word(repaired, dictionary)
        if rep_root is not None:
            return join_root_suffix(rep_root, rep_suf), 0.5


    # 6. Conservative Candidate Search for High-Confidence Typos Only (dist <= 1.0)
    best_candidate = None
    best_dist = 999.0
    best_score = -999.0

    for suf_try in [''] + SUFFIXES:
        if repaired.endswith(suf_try) and len(repaired) >= len(suf_try):
            stem = repaired[:-len(suf_try)] if suf_try else repaired
            if len(stem) < 2:
                continue

            for dict_word in word_list:
                # Disallow matching very short words to long words or vice-versa
                if abs(len(stem) - len(dict_word)) > 1:
                    continue

                d = weighted_edit_distance(stem, dict_word, max_dist=MAX_EDIT_DIST)
                if d <= MAX_EDIT_DIST:
                    cand = join_root_suffix(dict_word, suf_try)
                    lm_score = score_candidate(cand, prev_word, next_word)

                    if d < best_dist or (abs(d - best_dist) < 0.01 and lm_score > best_score):
                        best_dist = d
                        best_score = lm_score
                        best_candidate = cand

    # Only accept candidate if it's within strict distance
    if best_candidate and best_dist <= MAX_EDIT_DIST:
        return best_candidate, best_dist

    # If no confident match found, preserve the original OCR word (Safe Fail)
    return word, 0.0


def correct_text(text: str) -> Dict[str, Any]:
    """
    Correct a full block of mixed Kannada text.
    Returns structured results with list of detailed corrections.
    """
    dictionary = get_dictionary()
    if not dictionary:
        load_dictionary()
        train_model(get_word_list())

    tokens = tokenize(text)
    corrections = []

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

        corrected_word, dist = suggest_kannada_word(orig_word, prev_kannada, next_kannada)

        if corrected_word != orig_word:
            corrections.append({
                'original': orig_word,
                'correction': corrected_word,
                'edit_distance': round(dist, 2),
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
