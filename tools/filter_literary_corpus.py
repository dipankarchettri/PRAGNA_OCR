"""
Filter raw, uncurated text-extraction dumps down to a clean Kannada literary
corpus, ready to hand to tools/build_ngram_model.py via --corpus-dir.

Why this exists: source folders like a bulk "extract every book/scan/epub we
have" dump mix genuinely clean Kannada text with things that must not reach
the n-gram trainer -- unrelated English documents, and (more subtly) badly
OCR'd scans where a scanner-watermark line repeats hundreds of times or a
page dissolves into single stray glyphs per line. iter_corpus_sentences()
already drops non-Kannada tokens per line, so pure-English files are mostly
harmless noise (just wasted I/O). What it can't catch is Kannada-*charset*
garbage: fragmented OCR lines that still parse as "kannada" tokens under
tokenize() without being real words, which would quietly inflate unigram/
bigram counts. This script filters at the line and file level before that
data ever reaches the trainer.

Handles two input kinds:
  - .txt files, one physical line per unit (as extracted upstream).
  - .docx files, one paragraph per unit (python-docx) -- Kanaja-style
    digitized-book Word files land here.

Filtering, per file:
  0. OCR-origin rejection -- any file whose *whole-file* average
     characters-per-unit (line for .txt, paragraph for .docx) falls below a
     threshold is treated as OCR/scan output and dropped entirely, no
     partial credit. Rationale: OCR engines (this project's own Tesseract
     stage included) emit one line per detected line box on the scanned
     page, so OCR text has short, fairly uniform line lengths mirroring the
     print layout -- typically under ~55 chars/line for Kannada book pages.
     Natively-authored/extracted digital text (epub paragraphs, docx, typed
     documents) has no such constraint and reliably comes out far longer per
     unit. Validated against this corpus: every .txt file with "_ocred"
     literally in its filename measured 9.8-16.2 chars/line; every
     clean_epub_text_extracted file measured 58.5+ *except* three that
     turned out to say "Scanned by CamScanner" in the body text and
     measured 15.5-33.7 -- caught correctly despite the folder name; a
     plain-typed file with no scan behind it (a travel essay) measured
     84-85. For .docx, a much lower floor (15 chars/paragraph) is used as a
     tripwire rather than a tuned cutoff, since paragraph granularity is
     coarser than OCR line granularity and legitimate short paragraphs
     (dialogue, verse, list items) are common; the observed range across a
     sampled Kanaja docx batch was 26-586, so 15 only fires on something
     genuinely degenerate.
  1. Repeated-line removal -- any line (after stripping) that recurs more
     than a frequency threshold is treated as a scanner watermark / running
     header and dropped everywhere in the file. Threshold-based, not tied to
     any specific string, so it generalizes across sources.
  2. Structural noise removal -- blank lines, pure page-number markers,
     lines with no letters at all.
  3. Per-line Kannada-content gate -- a line is kept only if it has enough
     Kannada letters, and Kannada letters are the majority of its letters.
     This drops both English-heavy lines in mixed files and single/double
     stray-glyph OCR fragments.
  4. File-level accept/reject -- if too little survives steps 1-3, the file
     is rejected outright (its signal-to-noise is too low to be worth
     partial inclusion) rather than written as a near-empty stub.

Usage:
    # Report only -- see what would happen, write nothing
    python tools/filter_literary_corpus.py --source /path/to/dump --report-only

    # Filter for real (.txt and .docx sources can be mixed freely)
    python tools/filter_literary_corpus.py \
        --source /mnt/siet_llm_data/new_data_text_extracted \
        --source /mnt/siet_llm_data/clean_epub_text_extracted \
        --source /mnt/siet_llm_data/kanaja_docx \
        --output data/corpus_literary
"""

import argparse
import csv
import os
import re
from collections import Counter

KANNADA_RE = re.compile(r'[ಀ-೿]')
LETTER_RE = re.compile(r'[^\W\d_]', re.UNICODE)
PAGE_MARKER_RE = re.compile(r'^\s*(page|pg)\.?\s*[:#]?\s*\d+\s*$', re.IGNORECASE)

MIN_LINE_KANNADA_CHARS = 3
MIN_LINE_KANNADA_RATIO = 0.5
MIN_FILE_KEPT_CHARS = 200
# A line recurring at least this often, or covering at least this fraction
# of the file's lines, is boilerplate (running header/footer, watermark).
REPEATED_LINE_ABS_THRESHOLD = 5
REPEATED_LINE_FRAC_THRESHOLD = 0.02
# Whole-file average chars/unit below this -> treated as OCR/scan output,
# rejected outright. See module docstring step 0 for how these were picked.
OCR_AVG_CHARS_PER_UNIT_THRESHOLD = {'.txt': 55, '.docx': 15}


def find_source_files(source_dirs):
    for d in source_dirs:
        for root, _, files in os.walk(d):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in ('.txt', '.docx'):
                    yield os.path.join(root, fname)


def read_units(path: str):
    """Returns a list of raw text units: physical lines for .txt, paragraphs for .docx."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.docx':
        import docx
        d = docx.Document(path)
        return [p.text for p in d.paragraphs]
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read().split('\n')


def is_structural_noise(line: str) -> bool:
    if not line:
        return True
    if PAGE_MARKER_RE.match(line):
        return True
    if not LETTER_RE.search(line):
        return True
    return False


def kannada_ratio(text: str):
    letters = LETTER_RE.findall(text)
    if not letters:
        return 0.0, 0, 0
    kn = sum(1 for c in letters if KANNADA_RE.match(c))
    return kn / len(letters), kn, len(letters)


def filter_file(path: str):
    """Returns (kept_lines, stats dict). Raises on unreadable files (e.g. corrupt docx)."""
    ext = os.path.splitext(path)[1].lower()
    lines = read_units(path)
    orig_chars = sum(len(ln) for ln in lines)

    avg_chars_per_line = orig_chars / max(1, len(lines))
    threshold = OCR_AVG_CHARS_PER_UNIT_THRESHOLD.get(ext, 55)
    if avg_chars_per_line < threshold:
        return [], {
            'orig_chars': orig_chars,
            'orig_lines': len(lines),
            'kept_chars': 0,
            'kept_lines': 0,
            'boilerplate_lines_dropped': 0,
            'avg_chars_per_line': avg_chars_per_line,
            'is_ocr': True,
        }

    stripped = [ln.strip() for ln in lines]
    counts = Counter(l for l in stripped if l)
    repeat_threshold = max(REPEATED_LINE_ABS_THRESHOLD, int(len(lines) * REPEATED_LINE_FRAC_THRESHOLD))
    boilerplate = {l for l, c in counts.items() if c >= repeat_threshold}

    kept_lines = []
    for ln in stripped:
        if ln in boilerplate:
            continue
        if is_structural_noise(ln):
            continue
        ratio, kn_count, letter_count = kannada_ratio(ln)
        if kn_count < MIN_LINE_KANNADA_CHARS or ratio < MIN_LINE_KANNADA_RATIO:
            continue
        kept_lines.append(ln)

    kept_text = '\n'.join(kept_lines)
    stats = {
        'orig_chars': orig_chars,
        'orig_lines': len(lines),
        'kept_chars': len(kept_text),
        'kept_lines': len(kept_lines),
        'boilerplate_lines_dropped': sum(counts[l] for l in boilerplate),
        'avg_chars_per_line': avg_chars_per_line,
        'is_ocr': False,
    }
    return kept_lines, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', action='append', required=True, help='Source directory to scan (repeatable)')
    parser.add_argument('--output', default=os.path.join('data', 'corpus_literary'), help='Where to write filtered .txt files')
    parser.add_argument('--report', default=os.path.join('data', 'corpus_literary_report.csv'), help='Where to write the per-file CSV report')
    parser.add_argument('--report-only', action='store_true', help="Compute and report, but don't write filtered output")
    args = parser.parse_args()

    files = list(find_source_files(args.source))
    print(f"[*] Found {len(files)} .txt/.docx files across {len(args.source)} source dir(s)")

    if not args.report_only:
        os.makedirs(args.output, exist_ok=True)

    rows = []
    kept_count = 0
    rejected_ocr_count = 0
    rejected_sparse_count = 0
    rejected_error_count = 0
    kept_bytes = 0
    rejected_ocr_bytes = 0
    rejected_sparse_bytes = 0
    used_names = set()

    for i, path in enumerate(files, 1):
        if i % 20 == 0:
            print(f"    ... {i}/{len(files)}")

        src_dir = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path)
        orig_bytes = os.path.getsize(path)

        try:
            kept_lines, stats = filter_file(path)
        except Exception as e:
            print(f"    [!] {fname}: {e}")
            rejected_error_count += 1
            rows.append({
                'source_dir': src_dir, 'filename': fname, 'orig_bytes': orig_bytes,
                'orig_chars': 0, 'avg_chars_per_line': '0.0', 'kept_chars': 0,
                'kept_ratio': '0.000', 'orig_lines': 0, 'kept_lines': 0,
                'boilerplate_lines_dropped': 0, 'verdict': 'rejected_error',
            })
            continue

        if stats.get('is_ocr'):
            verdict = 'rejected_ocr'
        elif stats['kept_chars'] < MIN_FILE_KEPT_CHARS:
            verdict = 'rejected_sparse'
        else:
            verdict = 'kept'

        if verdict == 'kept':
            kept_count += 1
            kept_bytes += orig_bytes
            if not args.report_only:
                out_name = os.path.splitext(fname)[0] + '.txt'
                if out_name in used_names:
                    out_name = f"{src_dir}__{out_name}"
                used_names.add(out_name)
                out_path = os.path.join(args.output, out_name)
                with open(out_path, 'w', encoding='utf-8') as out_f:
                    out_f.write('\n'.join(kept_lines) + '\n')
        elif verdict == 'rejected_ocr':
            rejected_ocr_count += 1
            rejected_ocr_bytes += orig_bytes
        else:
            rejected_sparse_count += 1
            rejected_sparse_bytes += orig_bytes

        rows.append({
            'source_dir': src_dir,
            'filename': fname,
            'orig_bytes': orig_bytes,
            'orig_chars': stats['orig_chars'],
            'avg_chars_per_line': f"{stats['avg_chars_per_line']:.1f}",
            'kept_chars': stats['kept_chars'],
            'kept_ratio': f"{(stats['kept_chars'] / stats['orig_chars']):.3f}" if stats['orig_chars'] else '0.000',
            'orig_lines': stats['orig_lines'],
            'kept_lines': stats['kept_lines'],
            'boilerplate_lines_dropped': stats['boilerplate_lines_dropped'],
            'verdict': verdict,
        })

    os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[✓] {kept_count} files kept ({kept_bytes / 1e6:.1f} MB raw source)")
    print(f"[✓] {rejected_ocr_count} files rejected as OCR-derived ({rejected_ocr_bytes / 1e6:.1f} MB raw source)")
    print(f"[✓] {rejected_sparse_count} files rejected as too sparse/irrelevant ({rejected_sparse_bytes / 1e6:.1f} MB raw source)")
    if rejected_error_count:
        print(f"[!] {rejected_error_count} files could not be read (see [!] lines above)")
    print(f"[✓] Report written to {args.report}")
    if not args.report_only:
        out_size = sum(os.path.getsize(os.path.join(args.output, f)) for f in os.listdir(args.output))
        print(f"[✓] Filtered corpus written to {args.output} ({out_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
