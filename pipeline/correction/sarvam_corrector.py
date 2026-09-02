"""
Sarvam-backed post-OCR correction for Kannada.

Three ways of using a language model as a corrector, in decreasing order of how
much the LM is trusted to invent text:

  'sarvam-rerank'   Candidate generation stays with the existing rule engine
                    (generate_kannada_candidates: script normalisation, glyph
                    confusion, morphology, 1-edit search). The LM replaces only
                    the *ranking* step -- it scores the whole sentence once per
                    candidate and picks the reading it finds most probable.
                    The LM can never produce a word the dictionary/morphology
                    layer did not already propose.

  'hybrid'          The rule engine runs unmodified, then the LM acts purely as
                    a veto: a correction the rule engine wanted to make is
                    applied only if the LM also scores the corrected sentence
                    above the original. Strictly fewer changes than 'rule',
                    aimed at precision.

  'sarvam-generate' Few-shot completion -- the LM rewrites a whole line. The
                    only mode where the model can emit text nothing else
                    proposed, and therefore the only one that can hallucinate
                    fluent-but-wrong Kannada into the corpus. Guarded by the
                    drift checks in `_generation_is_plausible`, but treat its
                    output as unverified.

Given this project's stated priority -- a wrong "fix" corrupts training data
more expensively than an uncorrected OCR error -- 'hybrid' and 'sarvam-rerank'
are the modes worth measuring first. 'sarvam-generate' is included because a
base model's few-shot behaviour is the thing most people actually want to test,
and because it is the natural baseline that a future fine-tune has to beat.

Every entry point returns the same dict/tuple shapes as corrector.correct_text
and corrector.correct_layout_lines, so the pipeline can swap engines without
knowing which one it is talking to.

Which checkpoint runs is a separate axis from which mode: sarvam-1 (2B base,
research licence) and sarvam-30b (30B MoE instruct, Apache 2.0) both drive all
three. See sarvam_lm.KNOWN_MODELS for the licence and hardware differences, and
sarvam_vllm.py for why the 30B needs a vLLM server rather than transformers.
"""

import difflib
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from . import sarvam_lm, sarvam_vllm
from .corrector import (
    HIGH_OCR_CONFIDENCE_TRUST,
    MAX_CORRECTION_EDIT_DISTANCE,
    NON_TEXT_LINE_CONFIDENCE,
    correct_text as rule_correct_text,
    generate_kannada_candidates,
    is_valid_surface_word,
)
from .dictionary import get_dictionary, load_dictionary
from .edit_distance import weighted_edit_distance
from .ocr_repairs import normalize_script
from .tokenizer import tokenize, reconstruct

ENGINE_RERANK = 'sarvam-rerank'
ENGINE_GENERATE = 'sarvam-generate'
ENGINE_HYBRID = 'hybrid'

# Which backend actually runs the model. Both expose the same functions
# (score_sequences / log_probs / complete / chat / has_chat_template), so
# nothing below this line changes with the choice.
#
#   'transformers'  in-process HF model. Fine for sarvam-1 (2B, bf16); cannot
#                   run either quantized sarvam-30b build -- see sarvam_vllm's
#                   module docstring for why.
#   'vllm'          an already-running vLLM OpenAI-compatible server. Required
#                   for sarvam-30b, and keeps the weights resident between CLI
#                   invocations.
#   'auto'          vllm if a server answers, transformers otherwise.
#
# 'auto' is the default so that starting a server is all it takes to switch,
# and so a machine with no server keeps working exactly as before.
BACKEND = os.environ.get('SARVAM_BACKEND', 'auto')

_active_backend = None


def _lm():
    """The active LM backend module, resolved once."""
    global _active_backend
    if _active_backend is not None:
        return _active_backend

    if BACKEND == 'vllm':
        _active_backend = sarvam_vllm
    elif BACKEND == 'transformers':
        _active_backend = sarvam_lm
    elif BACKEND == 'auto':
        _active_backend = sarvam_vllm if sarvam_vllm.is_available() else sarvam_lm
    else:
        raise ValueError(
            f"Unknown SARVAM_BACKEND '{BACKEND}'. Expected 'auto', 'vllm' or 'transformers'."
        )
    return _active_backend


def active_backend_name() -> str:
    return 'vllm' if _lm() is sarvam_vllm else 'transformers'


# How many Kannada words of context on each side of the target word are fed to
# the LM when scoring a candidate. The whole line would be more informative but
# costs a forward pass over the full line per candidate per word; a window
# keeps the sequence short and the local agreement/sandhi evidence -- which is
# what actually discriminates these candidates -- is local.
LM_CONTEXT_WORDS = int(os.environ.get('SARVAM_CONTEXT_WORDS', '8'))

# Candidates from the rule engine scored per word, best-first by the rule
# engine's own ranking. Past the top handful the candidates are increasingly
# unrelated words that merely happen to be one edit away.
MAX_LM_CANDIDATES = int(os.environ.get('SARVAM_MAX_CANDIDATES', '8'))

# How much more probable (in nats, summed over the scored window) the LM must
# find a candidate before it is allowed to overwrite what OCR actually read.
# Asymmetric on purpose: leaving an OCR error is recoverable, writing a wrong
# word into the corpus is not.
#
# Swept with tools/sarvam_bench.py over 100 synthetic-noise lines (sarvam-1,
# margins 1/2/4/8/16). Harmful changes fall monotonically with the threshold
# and both LM modes plateau at 8:
#
#   margin   rerank CER / broke     hybrid CER / broke
#     1      0.0255 / 73            0.0201 / 14
#     2      0.0239 / 58            0.0199 / 12
#     4      0.0218 / 35            0.0195 /  8
#     8      0.0194 / 11            0.0194 /  6
#    16      0.0194 /  7            0.0194 /  5
#
# (rule engine, same input: 0.0205 / 20.) Past 8 the CER gain is exhausted and
# only the change count keeps shrinking, so 8 is the point where both modes
# beat the rule engine on error rate while making a fraction of its harmful
# changes. Raise it toward 16 to buy a little more precision at the cost of
# recall.
#
# Measured on *synthetic* noise, which is generated by inverting the
# corrector's own glyph-confusion matrix and so favours the rule engine's
# error model -- re-sweep against real ground-truth pages (--pages) before
# treating this as final.
LM_MIN_MARGIN = float(os.environ.get('SARVAM_MIN_MARGIN', '8.0'))

# Generate mode only: reject the LM's rewrite of a line if it drifts further
# than this from the input, as a fraction of input length. A real post-OCR fix
# touches a few glyphs; a rewrite this far from the source is the model
# continuing the text or paraphrasing it, not correcting it.
MAX_GENERATION_DRIFT = float(os.environ.get('SARVAM_MAX_DRIFT', '0.35'))

# Generate mode only: acceptable output/input length ratio. Catches the two
# classic base-model failures -- truncating the line, and running on past it.
MIN_GENERATION_LENGTH_RATIO = 0.6
MAX_GENERATION_LENGTH_RATIO = 1.5

KANNADA_RE = re.compile(r'[ಀ-೥]')


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _ensure_dictionary() -> Set[str]:
    dictionary = get_dictionary()
    if not dictionary:
        load_dictionary()
        dictionary = get_dictionary()
    return dictionary


def _context_window(tokens: List[Dict[str, Any]], index: int) -> Tuple[int, int]:
    """
    Token-index slice [lo, hi) spanning LM_CONTEXT_WORDS Kannada words either
    side of tokens[index], including the punctuation/whitespace tokens between
    them so the scored string reads as real text.
    """
    lo = index
    seen = 0
    for i in range(index - 1, -1, -1):
        lo = i
        if tokens[i]['type'] == 'kannada':
            seen += 1
            if seen >= LM_CONTEXT_WORDS:
                break

    hi = index + 1
    seen = 0
    for i in range(index + 1, len(tokens)):
        if tokens[i]['type'] == 'kannada':
            seen += 1
            if seen > LM_CONTEXT_WORDS:
                break
        hi = i + 1

    return lo, hi


def _window_variants(
    tokens: List[Dict[str, Any]],
    index: int,
    candidates: List[str]
) -> List[str]:
    """Render the context window once per candidate substituted at `index`."""
    lo, hi = _context_window(tokens, index)
    prefix = ''.join(t['value'] for t in tokens[lo:index])
    suffix = ''.join(t['value'] for t in tokens[index + 1:hi])
    return [f"{prefix}{cand}{suffix}" for cand in candidates]


def _neighbour_words(tokens: List[Dict[str, Any]], index: int) -> Tuple[Optional[str], Optional[str]]:
    prev_word = next((t['value'] for t in reversed(tokens[:index]) if t['type'] == 'kannada'), None)
    next_word = next((t['value'] for t in tokens[index + 1:] if t['type'] == 'kannada'), None)
    return prev_word, next_word


def _result(text: str, corrected: str, corrections: List[Dict[str, Any]], kannada_count: int) -> Dict[str, Any]:
    """Assemble correct_text's return shape."""
    return {
        'original': text,
        'corrected': corrected,
        'has_errors': len(corrections) > 0,
        'total_words': kannada_count,
        'total_corrections': len(corrections),
        'accuracy_rate': round((1.0 - (len(corrections) / (kannada_count or 1))) * 100, 1),
        'corrections': corrections,
    }


# ─────────────────────────────────────────────────────────────
# Mode 1: LM reranking over rule-engine candidates
# ─────────────────────────────────────────────────────────────

def correct_text_rerank(
    text: str,
    word_confidences: Optional[List[Tuple[str, float]]] = None
) -> Dict[str, Any]:
    """
    Apply the rule engine's deterministic script-level repairs, then let the LM
    choose among the rule engine's candidates for every word that is still not
    a valid surface form.

    The script-normalisation pre-pass is kept because it is deterministic and
    lossless (Repha, zero-digit/anusvara, illegal vowel+matra) -- there is
    nothing for a language model to arbitrate there. What the LM replaces is
    the n-gram scoring plus the corpus-frequency gates in
    suggest_kannada_word, which are the parts that are genuinely making a
    judgement call about which reading is right.
    """
    dictionary = _ensure_dictionary()

    # ocr_repair-only pass: applies script normalisation and split-word
    # healing, reports (but does not apply) the dictionary-level corrections
    # that this function is here to decide for itself.
    pre = rule_correct_text(text, allowed_types={'ocr_repair'}, word_confidences=word_confidences)
    corrections: List[Dict[str, Any]] = [c for c in pre['corrections'] if c['type'] == 'ocr_repair']

    tokens = tokenize(pre['corrected'])
    kannada_count = sum(1 for t in tokens if t['type'] == 'kannada')

    conf_by_word: Dict[str, float] = dict(word_confidences or [])

    for i, token in enumerate(tokens):
        if token['type'] != 'kannada':
            continue

        word = token['value']
        if len(word) <= 1:
            continue
        if conf_by_word.get(word, -1) >= HIGH_OCR_CONFIDENCE_TRUST:
            continue
        if is_valid_surface_word(word, dictionary):
            continue

        prev_word, next_word = _neighbour_words(tokens, i)
        ranked = generate_kannada_candidates(word, dictionary, prev_word, next_word)

        # Same distance ceiling the rule engine enforces: past 2 weighted
        # edits the "candidate" is an unrelated word reachable by a cheap
        # transform, and no amount of LM fluency makes substituting it safe.
        normalized = normalize_script(word)
        pool: List[Tuple[str, str]] = []
        for cand, _score, ctype in ranked:
            if cand == word or ctype == 'none':
                continue
            if weighted_edit_distance(normalized, cand) > MAX_CORRECTION_EDIT_DISTANCE:
                continue
            pool.append((cand, 'word_correction' if ctype.endswith('_unconstrained') else ctype))
            if len(pool) >= MAX_LM_CANDIDATES:
                break

        if not pool:
            continue

        variants = [word] + [c for c, _ in pool]
        scores = _lm().log_probs(_window_variants(tokens, i, variants))

        original_score = scores[0]
        best_idx = max(range(1, len(scores)), key=lambda k: scores[k])
        margin = scores[best_idx] - original_score

        if margin < LM_MIN_MARGIN:
            continue

        best_word, best_type = pool[best_idx - 1]
        corrections.append({
            'original': word,
            'correction': best_word,
            'edit_distance': round(weighted_edit_distance(word, best_word), 2),
            'type': best_type,
            'start': token['start'],
            'end': token['end'],
            'lm_margin': round(margin, 3),
            'engine': ENGINE_RERANK,
        })
        token['value'] = best_word

    return _result(text, reconstruct(tokens), corrections, kannada_count)


# ─────────────────────────────────────────────────────────────
# Mode 2: LM as a veto over the rule engine
# ─────────────────────────────────────────────────────────────

def correct_text_hybrid(
    text: str,
    word_confidences: Optional[List[Tuple[str, float]]] = None
) -> Dict[str, Any]:
    """
    Run the rule engine, then drop any word-level correction the LM disagrees
    with -- i.e. keep a change only if the corrected sentence scores at least
    LM_MIN_MARGIN above the original sentence.

    Deterministic 'ocr_repair' fixes are never vetoed: they are script-level
    normalisations with a single correct answer, and a base LM's opinion about
    them is noise. Only 'word_correction'/'hybrid' fixes -- the ones that
    involve choosing a different word -- go to the LM.

    The result is always a subset of what 'rule' would have produced, so this
    mode can only trade recall for precision, never the reverse.
    """
    full = rule_correct_text(text, word_confidences=word_confidences)
    if not full['corrections']:
        return full

    # Re-derive the post-script-repair text so token offsets line up with what
    # the rule engine was actually looking at when it proposed each fix, then
    # apply only the word-level proposals the LM endorses.
    pre = rule_correct_text(text, allowed_types={'ocr_repair'}, word_confidences=word_confidences)
    tokens = tokenize(pre['corrected'])
    kannada_count = sum(1 for t in tokens if t['type'] == 'kannada')

    # Keyed by the original word rather than by character offset: after
    # heal_split_tokens merges a split word, the rule engine's reported offsets
    # refer to the pre-heal text while the tokens here come from the post-heal
    # text, so the two offset spaces don't line up. A word occurring twice in
    # one line can attract more than one proposal (context differs), so each
    # occurrence is scored against all of them and takes the best.
    proposals: Dict[str, List[Dict[str, Any]]] = {}
    kept: List[Dict[str, Any]] = [c for c in full['corrections'] if c['type'] == 'ocr_repair']
    for c in full['corrections']:
        if c['type'] == 'ocr_repair':
            continue
        seen = proposals.setdefault(c['original'], [])
        if all(c['correction'] != s['correction'] for s in seen):
            seen.append(c)

    for i, token in enumerate(tokens):
        if token['type'] != 'kannada':
            continue
        options = proposals.get(token['value'])
        if not options:
            continue

        scores = _lm().log_probs(
            _window_variants(tokens, i, [token['value']] + [o['correction'] for o in options])
        )
        best = max(range(1, len(scores)), key=lambda k: scores[k])
        margin = scores[best] - scores[0]
        if margin < LM_MIN_MARGIN:
            continue

        proposal = options[best - 1]
        token['value'] = proposal['correction']
        kept.append({**proposal, 'lm_margin': round(margin, 3), 'engine': ENGINE_HYBRID})

    return _result(text, reconstruct(tokens), kept, kannada_count)


# ─────────────────────────────────────────────────────────────
# Mode 3: few-shot generative rewriting
# ─────────────────────────────────────────────────────────────

# Few-shot conditioning for a base completion model. These are format
# exemplars, not correction rules: they establish "line in, repaired line out,
# stop at the newline" for a model that has no instruction tuning. They are
# deliberately generic error *shapes* (dropped matra, split word, wrong
# consonant) rather than the specific vocabulary of any document, and nothing
# in the pipeline looks a word up in them -- so the zero-hardcoding rule in
# CLAUDE.md, which bans word-specific mappings inside the correction engine, is
# not circumvented here. Override with SARVAM_FEWSHOT_FILE (a TSV of
# noisy<TAB>clean lines) to test different conditioning.
DEFAULT_FEWSHOT: List[Tuple[str, str]] = [
    ("ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತದೆ",
     "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜೀವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತ್ತದೆ"),
    ("ಈ ಕಾರ್ಯವು ಸಾಧ್ಯ ವಾಗಬೇಕಿದೆ ಎಂದು ಅವರು ಹೇಳಿದರು",
     "ಈ ಕಾರ್ಯವು ಸಾಧ್ಯವಾಗಬೇಕಿದೆ ಎಂದು ಅವರು ಹೇಳಿದರು"),
    ("ರಾಜ್ಯ ಸರ್ಕಾರವು ಹೊಸ ಯೋಜನಯನ್ನು ಜಾರಿಗೆ ತಂದಿದೆ",
     "ರಾಜ್ಯ ಸರ್ಕಾರವು ಹೊಸ ಯೋಜನೆಯನ್ನು ಜಾರಿಗೆ ತಂದಿದೆ"),
    ("ಅವನು ಪುಸ್ತಕವನ್ನು ಓದುತಿದ್ದನು ಮತ್ತು ಬರಯುತ್ತಿದ್ದನು",
     "ಅವನು ಪುಸ್ತಕವನ್ನು ಓದುತ್ತಿದ್ದನು ಮತ್ತು ಬರೆಯುತ್ತಿದ್ದನು"),
]

NOISY_LABEL = 'ತಪ್ಪು'
CLEAN_LABEL = 'ಸರಿ'


def load_fewshot() -> List[Tuple[str, str]]:
    path = os.environ.get('SARVAM_FEWSHOT_FILE')
    if not path or not os.path.exists(path):
        return DEFAULT_FEWSHOT
    pairs = []
    with open(path, encoding='utf-8') as fh:
        for raw in fh:
            if '\t' not in raw:
                continue
            noisy, clean = raw.rstrip('\n').split('\t', 1)
            if noisy and clean:
                pairs.append((noisy, clean))
    return pairs or DEFAULT_FEWSHOT


def build_prompt(line: str, examples: Optional[List[Tuple[str, str]]] = None) -> str:
    shots = examples if examples is not None else load_fewshot()
    blocks = [f"{NOISY_LABEL}: {noisy}\n{CLEAN_LABEL}: {clean}" for noisy, clean in shots]
    blocks.append(f"{NOISY_LABEL}: {line}\n{CLEAN_LABEL}:")
    return "\n\n".join(blocks)


# Instruction used when the loaded checkpoint is instruction-tuned (sarvam-30b
# and anything else with a chat template). The constraints are the same ones
# the few-shot exemplars establish by demonstration, stated directly because an
# instruct model responds to them far better than to four examples -- and
# because the failure modes being ruled out (paraphrasing, completing the
# passage, explaining itself) are exactly what an unconstrained chat model does
# when handed a broken sentence.
INSTRUCT_SYSTEM = (
    "You repair OCR errors in Kannada text. You never rewrite, translate, "
    "summarise, modernise, or continue the text, and you never explain your "
    "output."
)

INSTRUCT_TEMPLATE = (
    "The following line of Kannada was produced by OCR and may contain "
    "scanning errors: wrong consonants, missing or wrong vowel signs "
    "(matras), broken conjuncts, and words incorrectly split by a space.\n\n"
    "Repair only those errors. Keep every word the author actually wrote, in "
    "the same order. If the line is already correct, return it unchanged. "
    "Reply with the corrected line and nothing else.\n\n"
    "{line}"
)


def build_instruction(line: str) -> str:
    return INSTRUCT_TEMPLATE.format(line=line)


def _rewrite_line(line: str, budget: int) -> str:
    """Ask the LM for a repaired version of `line`, however it prefers to be asked."""
    if _lm().has_chat_template():
        raw = _lm().chat(
            build_instruction(line), max_new_tokens=budget, system=INSTRUCT_SYSTEM
        )
        # An instruct model may still wrap the answer in a line of preamble;
        # take the first line that actually contains Kannada.
        for candidate in (l.strip() for l in raw.split('\n')):
            if candidate and KANNADA_RE.search(candidate):
                return candidate
        return raw.strip()

    return _lm().complete(build_prompt(line), max_new_tokens=budget)


def _generation_is_plausible(source: str, generated: str) -> Tuple[bool, str]:
    """
    Reject a rewrite that is not recognisably a repair of `source`.

    A base model asked to complete a repair pattern will sometimes continue the
    passage, paraphrase it, or answer in a different language. None of those
    are detectable downstream once the text is in a corpus, so they are
    filtered here rather than trusted.
    """
    if not generated:
        return False, 'empty'
    if not KANNADA_RE.search(generated):
        return False, 'no_kannada'

    ratio = len(generated) / max(1, len(source))
    if not (MIN_GENERATION_LENGTH_RATIO <= ratio <= MAX_GENERATION_LENGTH_RATIO):
        return False, f'length_ratio={ratio:.2f}'

    drift = _char_distance(source, generated) / max(1, len(source))
    if drift > MAX_GENERATION_DRIFT:
        return False, f'drift={drift:.2f}'

    return True, ''


def _char_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[len(b)]


def diff_corrections(source: str, corrected: str) -> List[Dict[str, Any]]:
    """
    Word-level diff between two strings, in the pipeline's corrections schema.

    Generate mode rewrites whole lines, so unlike the token-substituting modes
    there is no per-token record of what changed -- it has to be recovered
    after the fact. Character offsets are into `source`.
    """
    src_words = source.split()
    dst_words = corrected.split()
    offsets: List[int] = []
    cursor = 0
    for w in src_words:
        idx = source.index(w, cursor)
        offsets.append(idx)
        cursor = idx + len(w)

    out: List[Dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src_words, dst_words).get_opcodes():
        if tag == 'equal':
            continue
        original = ' '.join(src_words[i1:i2])
        replacement = ' '.join(dst_words[j1:j2])
        if not original or not replacement or original == replacement:
            continue
        start = offsets[i1] if i1 < len(offsets) else len(source)
        out.append({
            'original': original,
            'correction': replacement,
            'edit_distance': round(weighted_edit_distance(original, replacement), 2),
            'type': 'word_correction',
            'start': start,
            'end': start + len(original),
            'engine': ENGINE_GENERATE,
        })
    return out


def correct_text_generate(
    text: str,
    word_confidences: Optional[List[Tuple[str, float]]] = None
) -> Dict[str, Any]:
    """
    Few-shot rewrite of each line by the LM, with the drift guards above.

    Lines are handled one at a time: a base model's output quality degrades
    over long completions, and a per-line guard can reject one bad rewrite
    without discarding the page.
    """
    lines = text.split('\n')
    kannada_count = sum(1 for t in tokenize(text) if t['type'] == 'kannada')

    out_lines: List[str] = []
    corrections: List[Dict[str, Any]] = []
    offset = 0

    for line in lines:
        if not KANNADA_RE.search(line):
            out_lines.append(line)
            offset += len(line) + 1
            continue

        try:
            generated = _rewrite_line(line, budget=max(32, len(line) // 2 + 32))
        except Exception:
            out_lines.append(line)
            offset += len(line) + 1
            continue

        ok, _reason = _generation_is_plausible(line, generated)
        if not ok or generated == line:
            out_lines.append(line)
            offset += len(line) + 1
            continue

        for c in diff_corrections(line, generated):
            c['start'] += offset
            c['end'] += offset
            corrections.append(c)
        out_lines.append(generated)
        offset += len(line) + 1

    return _result(text, '\n'.join(out_lines), corrections, kannada_count)


# ─────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────

_MODES = {
    ENGINE_RERANK: correct_text_rerank,
    ENGINE_HYBRID: correct_text_hybrid,
    ENGINE_GENERATE: correct_text_generate,
}


def correct_text_sarvam(
    text: str,
    mode: str = ENGINE_RERANK,
    word_confidences: Optional[List[Tuple[str, float]]] = None
) -> Dict[str, Any]:
    if mode not in _MODES:
        raise ValueError(f"Unknown Sarvam mode '{mode}'. Expected one of: {', '.join(_MODES)}")
    return _MODES[mode](text, word_confidences=word_confidences)


def correct_layout_lines_sarvam(
    layout_lines: List[Dict[str, Any]],
    mode: str = ENGINE_RERANK
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Layout-preserving counterpart of correct_layout_lines, with the same output
    shape (including 'is_likely_non_text').

    Lines already flagged non-text are passed through untouched rather than
    corrected: below NON_TEXT_LINE_CONFIDENCE the input is OCR hallucinating
    rows out of graphics, and asking a language model to "repair" that just
    spends a GPU pass turning noise into fluent noise.
    """
    corrected_lines: List[Dict[str, Any]] = []
    all_corrections: List[Dict[str, Any]] = []

    for line in layout_lines:
        conf = line.get('conf')
        is_non_text = conf is not None and conf < NON_TEXT_LINE_CONFIDENCE
        source = line.get('text', '')

        if is_non_text or not source.strip():
            text_out = source
        else:
            res = correct_text_sarvam(source, mode=mode, word_confidences=line.get('word_confidences'))
            text_out = res['corrected']
            all_corrections.extend(res['corrections'])

        corrected_lines.append({
            'text': text_out,
            'original_text': source,
            'alignment': line.get('alignment', 'L'),
            'top': line.get('top', 0),
            'left': line.get('left', 0),
            'width': line.get('width', 0),
            'height': line.get('height', 0),
            'page_num': line.get('page_num', 1),
            'ocr_confidence': conf,
            'is_likely_non_text': is_non_text,
        })

    return corrected_lines, all_corrections
