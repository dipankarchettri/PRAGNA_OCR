"""
Correction-engine A/B benchmark: rule vs Sarvam-1.

Answers the only question that matters for swapping in an LM corrector -- does
it lower CER/WER on real pages, and does it do so without making *wrong*
changes? The second half is reported separately on purpose: for a corpus meant
to train an LLM, a confident wrong "fix" is worse than an untouched OCR error,
so an engine that improves CER while raising the harmful-change count is not an
improvement.

Every engine is fed byte-identical input (OCR runs once, or is read from disk),
so differences are correction quality and nothing else.

Three ways to supply data:

  # Real pages: page image + same-basename .txt ground truth
  python tools/sarvam_bench.py --pages scans/*.jpg

  # Pre-OCR'd text: a noisy file and its clean reference
  python tools/sarvam_bench.py --text-pair raw.txt truth.txt

  # No ground truth on hand: corrupt clean corpus lines and measure recovery
  python tools/sarvam_bench.py --synthetic 200

  # Compare checkpoints (sarvam-1 base vs sarvam-30b instruct)
  python tools/sarvam_bench.py --synthetic 200 --model sarvam-30b-fp8

Reported per engine:
  CER / WER   character and word error rate against the reference
  d-CER       change from the uncorrected OCR text (negative = improvement)
  fixed       changes that moved a word to what the reference actually says
  broke       changes that took an already-correct word away from it
  other       changes to words that were wrong before and still are
  secs        wall clock, model load excluded
"""

import argparse
import difflib
import glob
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import init_pipeline
from pipeline.correction import ENGINES, SARVAM_ENGINES, correct_layout_lines_with, preload_engine
from pipeline.correction.edit_distance import GLYPH_CONFUSIONS
from tools.ocr_bench import char_error_rate, word_error_rate

KANNADA_RE = re.compile(r'[ಀ-೥]')
MATRAS = 'ಾಿೀುೂೃೆೇೈೊೋೌಂ್'


class Doc:
    """One benchmark unit: OCR/noisy lines plus the reference text for them."""

    def __init__(self, name: str, lines: List[Dict[str, Any]], reference: str):
        self.name = name
        self.lines = lines
        self.reference = reference

    @property
    def raw_text(self) -> str:
        return '\n'.join(l['text'] for l in self.lines)


# ─────────────────────────────────────────────────────────────
# Input loaders
# ─────────────────────────────────────────────────────────────

def load_page_docs(paths: List[str], lang: str, psm: int, oem: int) -> List[Doc]:
    """OCR each image once; ground truth is the same-basename .txt file."""
    from pipeline.ingestion import load_and_preprocess_image, normalize_resolution, preprocess_for_ocr
    from pipeline.ocr import ocr_image_with_layout

    docs = []
    for path in paths:
        gt_path = os.path.splitext(path)[0] + '.txt'
        if not os.path.exists(gt_path):
            print(f"  skipping {os.path.basename(path)}: no {os.path.basename(gt_path)} alongside it")
            continue

        img = load_and_preprocess_image(path, enhance_contrast=False)
        img = preprocess_for_ocr(normalize_resolution(img))
        lines = ocr_image_with_layout(img, lang=lang, psm=psm, oem=oem)
        reference = open(gt_path, encoding='utf-8').read()
        docs.append(Doc(os.path.basename(path), lines, reference))
    return docs


def load_text_pair(noisy_path: str, clean_path: str) -> List[Doc]:
    noisy = open(noisy_path, encoding='utf-8').read()
    clean = open(clean_path, encoding='utf-8').read()
    lines = [{'text': l} for l in noisy.split('\n')]
    return [Doc(os.path.basename(noisy_path), lines, clean)]


# Reverse of the optical confusion matrix: for each character the corrector
# knows can be *misread as* something else, the things it gets misread as. Used
# only to manufacture test noise -- the corrector never sees this table.
def _confusion_sources() -> Dict[str, List[str]]:
    sources: Dict[str, List[str]] = {}
    for wrong, options in GLYPH_CONFUSIONS.items():
        if len(wrong) != 1:
            continue
        for right, _cost in options:
            if len(right) == 1:
                sources.setdefault(right, []).append(wrong)
    return sources


def corrupt_line(line: str, rate: float, rng: random.Random, sources: Dict[str, List[str]]) -> str:
    """
    Inject OCR-shaped damage: optically confusable glyph swaps, dropped matras,
    and spurious intra-word spaces -- the three failure modes the correction
    engine is built around.
    """
    chars = list(line)
    n_edits = max(1, int(len(chars) * rate))
    for _ in range(n_edits):
        if not chars:
            break
        i = rng.randrange(len(chars))
        ch = chars[i]
        roll = rng.random()
        if roll < 0.5 and ch in sources:
            chars[i] = rng.choice(sources[ch])
        elif roll < 0.8 and ch in MATRAS:
            chars[i] = ''
        elif KANNADA_RE.match(ch) and 0 < i < len(chars) - 1:
            chars[i] = ch + ' '
    return ''.join(chars)


def load_synthetic_docs(
    count: int,
    corpus_dir: str,
    rate: float,
    seed: int,
    min_words: int = 6
) -> List[Doc]:
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(corpus_dir, '*.txt')))
    if not files:
        raise SystemExit(f"No .txt corpus files under {corpus_dir}")

    pool: List[str] = []
    rng.shuffle(files)
    for path in files:
        with open(path, encoding='utf-8', errors='ignore') as fh:
            for raw in fh:
                line = ' '.join(raw.split())
                if len(line.split()) >= min_words and KANNADA_RE.search(line):
                    pool.append(line)
        if len(pool) >= count * 20:
            break

    if not pool:
        raise SystemExit(f"No usable Kannada lines found under {corpus_dir}")

    sources = _confusion_sources()
    picked = rng.sample(pool, min(count, len(pool)))
    return [
        Doc(f"syn{i:04d}", [{'text': corrupt_line(clean, rate, rng, sources)}], clean)
        for i, clean in enumerate(picked)
    ]


# ─────────────────────────────────────────────────────────────
# Change classification
# ─────────────────────────────────────────────────────────────

def _word_spans(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S+', text)]


def _locate(
    correction: Dict[str, Any],
    raw_spans: List[Tuple[int, int, str]],
    used: set
) -> Optional[int]:
    """
    Word index in the raw text that a correction refers to.

    Offset first, then a string fallback, because the two are not always in the
    same coordinate space: when heal_split_tokens merges a space-split word the
    corrector reports offsets into the pre-heal text, while later corrections
    on the same line are measured against the post-heal text. The fallback
    takes the first occurrence of the original word not already claimed by an
    earlier correction.
    """
    start = correction.get('start')
    if start is not None:
        idx = next((n for n, (s, e, _) in enumerate(raw_spans) if s <= start < e), None)
        if idx is not None and correction['original'] in raw_spans[idx][2]:
            return idx

    original = correction['original']
    return next(
        (n for n, (_s, _e, w) in enumerate(raw_spans) if n not in used and original in w),
        None
    )


def classify_changes(
    raw_text: str,
    reference: str,
    corrections: List[Dict[str, Any]]
) -> Tuple[int, int, int]:
    """
    Split the engine's changes into (fixed, broke, other) by aligning the
    uncorrected OCR text to the reference word by word.

    A change is 'fixed' if the word the engine wrote is what the reference has
    at that position, 'broke' if the reference already agreed with the OCR text
    the engine overwrote, and 'other' if the word was wrong before and is still
    not right -- which is neither a win nor a regression, just noise moved
    around.

    Alignment is difflib over whitespace-split words, so this is only
    meaningful where the OCR text and reference are broadly parallel; badly
    misaligned pages inflate 'other' rather than the other two counts.
    """
    raw_spans = _word_spans(raw_text)
    raw_words = [w for _, _, w in raw_spans]
    ref_words = reference.split()

    aligned: Dict[int, str] = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, raw_words, ref_words).get_opcodes():
        if tag in ('equal', 'replace') and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                aligned[i1 + k] = ref_words[j1 + k]

    fixed = broke = other = 0
    used: set = set()
    for corr in corrections:
        idx = _locate(corr, raw_spans, used)
        if idx is None or idx not in aligned:
            other += 1
            continue
        used.add(idx)

        ref_word = aligned[idx]
        original, replacement = corr['original'], corr['correction']
        # The aligned reference word can carry punctuation the token doesn't.
        if replacement in ref_word:
            fixed += 1
        elif original in ref_word:
            broke += 1
        else:
            other += 1
    return fixed, broke, other


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(engine: str, docs: List[Doc]) -> Dict[str, Any]:
    total = {'cer': 0.0, 'wer': 0.0, 'fixed': 0, 'broke': 0, 'other': 0, 'changes': 0}
    start = time.time()

    for doc in docs:
        corrected_lines, corrections = correct_layout_lines_with(doc.lines, engine=engine)
        hyp = '\n'.join(l['text'] for l in corrected_lines if not l.get('is_likely_non_text'))
        total['cer'] += char_error_rate(hyp, doc.reference)
        total['wer'] += word_error_rate(hyp, doc.reference)

        fixed, broke, other = classify_changes(doc.raw_text, doc.reference, corrections)
        total['fixed'] += fixed
        total['broke'] += broke
        total['other'] += other
        total['changes'] += len(corrections)

    n = len(docs)
    return {
        'engine': engine,
        'cer': total['cer'] / n,
        'wer': total['wer'] / n,
        'fixed': total['fixed'],
        'broke': total['broke'],
        'other': total['other'],
        'changes': total['changes'],
        'secs': time.time() - start,
    }


def baseline_metrics(docs: List[Doc]) -> Dict[str, float]:
    cer = sum(char_error_rate(d.raw_text, d.reference) for d in docs) / len(docs)
    wer = sum(word_error_rate(d.raw_text, d.reference) for d in docs) / len(docs)
    return {'cer': cer, 'wer': wer}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pages', nargs='+', metavar='IMG',
                    help='page images with same-basename .txt ground truth '
                         '(one image per reference file, so PDFs must be rasterized first)')
    ap.add_argument('--text-pair', nargs=2, metavar=('NOISY', 'CLEAN'),
                    help='an already-OCRd text file and its reference')
    ap.add_argument('--synthetic', type=int, metavar='N',
                    help='corrupt N clean corpus lines and measure recovery')
    ap.add_argument('--corpus', default='data/corpus_literary',
                    help='corpus directory for --synthetic (default: data/corpus_literary)')
    ap.add_argument('--noise-rate', type=float, default=0.04,
                    help='per-character corruption rate for --synthetic (default: 0.04)')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--model', help="Sarvam checkpoint for the LM engines: a KNOWN_MODELS key "
                                    "('sarvam-1', 'sarvam-30b-fp8'), a HF repo id, or a local path")
    ap.add_argument('--engines', default=','.join(ENGINES),
                    help=f"comma-separated engines to compare (default: all -- {', '.join(ENGINES)})")
    ap.add_argument('--lang', default='kan')
    ap.add_argument('--psm', type=int, default=6)
    ap.add_argument('--oem', type=int, default=1)
    ap.add_argument('--dump', metavar='DIR', help='write each engine\'s output text to DIR')
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(',') if e.strip()]
    unknown = [e for e in engines if e not in ENGINES]
    if unknown:
        raise SystemExit(f"Unknown engine(s): {', '.join(unknown)}. Choose from: {', '.join(ENGINES)}")

    if args.model:
        from pipeline.correction import sarvam_lm
        sarvam_lm.set_model(args.model)

    init_pipeline()

    if args.pages:
        paths = [p for pattern in args.pages for p in sorted(glob.glob(pattern))] or args.pages
        docs = load_page_docs(paths, args.lang, args.psm, args.oem)
    elif args.text_pair:
        docs = load_text_pair(*args.text_pair)
    elif args.synthetic:
        docs = load_synthetic_docs(args.synthetic, args.corpus, args.noise_rate, args.seed)
    else:
        raise SystemExit("Give one of --pages, --text-pair or --synthetic. See --help.")

    if not docs:
        raise SystemExit("No usable documents to benchmark.")

    base = baseline_metrics(docs)
    print(f"\n{len(docs)} document(s); uncorrected OCR baseline: "
          f"CER {base['cer']:.4f}  WER {base['wer']:.4f}\n")

    header = (f"  {'engine':16} {'CER':>7} {'d-CER':>8} {'WER':>7} "
              f"{'changes':>8} {'fixed':>6} {'broke':>6} {'other':>6} {'secs':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for engine in engines:
        if engine in SARVAM_ENGINES:
            # Excluded from the timing below; it is a one-off, not per-page cost.
            preload_engine(engine)
        row = evaluate(engine, docs)
        rows.append(row)
        print(f"  {row['engine']:16} {row['cer']:>7.4f} {row['cer'] - base['cer']:>+8.4f} "
              f"{row['wer']:>7.4f} {row['changes']:>8} {row['fixed']:>6} {row['broke']:>6} "
              f"{row['other']:>6} {row['secs']:>7.1f}")

        if args.dump:
            os.makedirs(args.dump, exist_ok=True)
            with open(os.path.join(args.dump, f"{engine}.txt"), 'w', encoding='utf-8') as fh:
                for doc in docs:
                    lines, _ = correct_layout_lines_with(doc.lines, engine=engine)
                    fh.write(f"### {doc.name}\n")
                    fh.write('\n'.join(l['text'] for l in lines) + '\n\n')

    best = min(rows, key=lambda r: r['cer'])
    print(f"\n  lowest CER: {best['engine']} ({best['cer']:.4f}), "
          f"{best['fixed']} fixed / {best['broke']} broke\n")


if __name__ == '__main__':
    main()
