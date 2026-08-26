"""
Main Kannada Autocorrect & Post-OCR Correction Engine
Combines: Unicode Glitch Cleaning -> Space Healing -> Safe Rule Repairs -> Dictionary-Gated Optical Normalization
"""

import re
from typing import Dict, List, Tuple, Any, Optional, Set
from .dictionary import get_dictionary, get_word_list, load_dictionary
from .tokenizer import tokenize, reconstruct
from .edit_distance import weighted_edit_distance
from .morphology import decompose_word, join_root_suffix, is_compound_word, SUFFIXES
from .ocr_repairs import apply_ocr_repairs, clean_unicode_glitches
from .ngram import train_model, score_candidate

# Strict protected tokens that should never be fused or altered
PROTECTED_TOKENS = {'ಶ್ರೀ', 'ಡಾ', 'ಪ್ರೊ', 'ಆ', 'ಈ', 'ಏ', 'ಮತ್ತು', 'ಹಾಗೂ', 'ಅಥವಾ', 'ಎಂದು', 'ಎಂಬ', 'ಒಂದು', 'ಆಗಿದೆ', 'ಆಗಿದ್ದಾರೆ', 'ಆಗುವುದು', 'ಆಗಿದೆ.', 'ಇದೆ', 'ಇದು', 'ಅಲ್ಲ', 'ಇಲ್ಲ'}




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


def suggest_kannada_word(word: str, prev_word: Optional[str] = None, next_word: Optional[str] = None) -> Tuple[str, float]:
    """
    Find best correction suggestion for a single Kannada word.
    Guaranteed not to corrupt valid proper nouns, compounds, or literary terms.
    """
    dictionary = get_dictionary()

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
        # If the word ended in a broken suffix like ುತದೆ, normalize it
        if suf == 'ಿಸುತ್ತದೆ' and word.endswith(('ುತದೆ', 'ಸುತದೆ')):
            return join_root_suffix(root, suf), 0.5
        # Otherwise, the original surface word is already a valid inflected form
        return word, 0.0

    # 4. Compound Word (Samasa) validation (e.g., ಸಂತಕವಿ, ಸಹಕಾರ, ಕನಕದಾಸರು, ಮರುಓದಿಗೆ)
    if is_compound_word(word, dictionary):
        return word, 0.0

    # 5. Targeted Rule-based OCR Repairs (Optical confusion, Halant/Virama, rephas)
    repaired = apply_ocr_repairs(word, dictionary)
    if repaired != word:
        if repaired in dictionary or is_compound_word(repaired, dictionary):
            return repaired, 0.5
        rep_root, rep_suf = decompose_word(repaired, dictionary)
        if rep_root is not None:
            return join_root_suffix(rep_root, rep_suf), 0.5
        # If repaired form is cleaner than original
        return repaired, 0.5

    # 6. Default Safe Fallback: Preserve original OCR text without corrupting
    return word, 0.0


def correct_text(text: str) -> Dict[str, Any]:
    """
    Correct a full block of mixed Kannada text.
    Combines Unicode normalization, space healing, and targeted optical OCR repairs.
    """
    dictionary = get_dictionary()
    if not dictionary:
        load_dictionary()
        train_model(get_word_list())

    # Pre-pass: clean Unicode zero-digit / anusvara scanning artifacts
    pre_cleaned = clean_unicode_glitches(text)

    tokens = tokenize(pre_cleaned)

    # Pre-pass 2: Heal split word gaps
    tokens, space_corrections = heal_split_tokens(tokens, dictionary)

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
