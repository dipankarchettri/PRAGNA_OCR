#!/usr/bin/env python3
"""
Build a page-aligned OCR evaluation set by typesetting clean Kannada book text.

Why this exists
---------------
Until now nothing in this repo could be measured. The CER figures quoted in
CLAUDE.md were produced by hand against pages that were never committed, so no
change to the OCR or correction stages could be shown to help or hurt. This
tool manufactures a fixed, reproducible eval set so tools/correction_bench.py
and tools/ocr_bench.py have something to score against.

Why we typeset rather than convert
----------------------------------
The obvious route -- LibreOffice .docx -> PDF, then render each page and read
its text layer -- was tried first and does not work on this corpus. On these
files LibreOffice emits justified Kannada as positioned glyph runs with **no
space characters at all** in the content stream, and its shaper duplicates
matras. Measured on data/kanaja_docx_raw/115.docx, page 3, every extraction
mode PyMuPDF offers (`get_text()`, `"words"`, `"blocks"`) returned the same
damage:

    reference:  'ಒದಗಿಸುವಹಂಬಲನಮ್ಮದು.'      <- words concatenated, no spaces
    reference:  'ಪ್ರಕಟಿಿಸಿ' / 'ದಿಿನಾಂಕ'      <- matra doubled by the shaper
    docx source: 'ಒದಗಿಸುವ ಹಂಬಲ ನಮ್ಮದು.'     <- correct, via python-docx

That reference is unusable: 49 "words" against 88 real ones drove WER to 1.19
and made every correction unclassifiable. A reference that is itself wrong
measures nothing.

So instead we read the paragraph text straight from the .docx with python-docx
(which is clean, verified above), typeset it ourselves with Pillow + RAQM using
the Noto Sans Kannada already in this repo, and write the exact text we laid
out as the reference. Alignment is then correct by construction rather than
recovered, and the ground truth is known-good.

Outputs, per page
-----------------
    <doc>_pNNN.png        the page image, what OCR sees
    <doc>_pNNN.txt        reference for OCR/correction scoring -- the lines as
                          typeset, including the running header and page number,
                          because OCR does see those
    <doc>_pNNN.para.txt   reference for *reflow* scoring -- the paragraphs as
                          authored, header/footer excluded and line wrapping
                          undone. This is what a correct corpus exporter should
                          produce from the page.

The .png/.txt pair is the same-basename convention tools/ocr_bench.py and
tools/correction_bench.py already expect.

HONESTY NOTE -- read before trusting a number produced with it
--------------------------------------------------------------
These pages are typeset and then optionally degraded synthetically. They are
not scans. A rendered page has perfect glyph shapes, uniform illumination, no
paper texture, no bleed-through, no binding curvature and no dust. Even at
--degrade 3 the noise model is a plausible guess at what a scanner does, not a
sample of one. Nor does it exercise the one font this repo happens to ship;
real books use many.

So this set is a fast, deterministic *regression gate*: it catches a change
that makes recognition worse and lets a preprocessing change be swept across a
difficulty ladder. It is not evidence about real book scans. Real scanned pages
with hand-checked transcripts, dropped into the same directory as
`page.jpg` + `page.txt`, are the honest test, and any conclusion that matters
should be confirmed against those.

Usage
-----
    python tools/build_eval_set.py --docs 12 --degrade 0,1,2,3
    python tools/build_eval_set.py --list
"""

import argparse
import glob
import io
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DEFAULT_DOCX_DIR = 'data/kanaja_docx_raw'
DEFAULT_OUT_DIR = 'tests/fixtures/eval'
FONT_PATH = 'web/static/fonts/NotoSansKannada-Regular.ttf'

# US Letter at 300 DPI, the resolution a book scanner is usually set to.
PAGE_W, PAGE_H = 2550, 3300
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 260, 260, 260
FONT_SIZE = 46
LINE_SPACING = 1.62
PARA_INDENT = 90

MIN_PARA_CHARS = 60
MIN_KANNADA_RATIO = 0.80


def _kannada_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if 'ಀ' <= c <= '೿') / len(letters)


# ─────────────────────────────────────────────────────────────
# Clean text from the .docx
# ─────────────────────────────────────────────────────────────

def clean_paragraphs(docx_path: str):
    """Body paragraphs worth typesetting: long enough, and actually Kannada."""
    import docx

    doc = docx.Document(docx_path)
    out = []
    for para in doc.paragraphs:
        text = ' '.join(para.text.split())
        if len(text) < MIN_PARA_CHARS:
            continue
        if _kannada_ratio(text) < MIN_KANNADA_RATIO:
            continue
        out.append(text)
    return out


# ─────────────────────────────────────────────────────────────
# Typesetting
# ─────────────────────────────────────────────────────────────

def wrap_paragraph(text: str, font, draw, max_width: int, first_indent: int):
    """Greedy word wrap using real shaped text metrics."""
    words = text.split()
    lines, current, indent = [], [], first_indent
    for word in words:
        trial = ' '.join(current + [word])
        if current and draw.textlength(trial, font=font) + indent > max_width:
            lines.append(' '.join(current))
            current, indent = [word], 0
        else:
            current.append(word)
    if current:
        lines.append(' '.join(current))
    return lines


def paginate(paragraphs, font, draw, max_width: int, lines_per_page: int):
    """
    Lay paragraphs out into pages.

    Yields (page_lines, page_paragraphs) where page_lines are the wrapped lines
    as they will be drawn and page_paragraphs are the same content unwrapped --
    the reflow reference. A paragraph split across a page boundary contributes
    its own fragment to each page, which is what a per-page reflow can
    legitimately recover.
    """
    page_lines, page_paras, current_para = [], [], []

    for para in paragraphs:
        wrapped = wrap_paragraph(para, font, draw, max_width, PARA_INDENT)
        for i, line in enumerate(wrapped):
            page_lines.append((line, i == 0))
            current_para.append(line)
            if len(page_lines) >= lines_per_page:
                if current_para:
                    page_paras.append(' '.join(current_para))
                yield page_lines, page_paras
                page_lines, page_paras, current_para = [], [], []
        if current_para:
            page_paras.append(' '.join(current_para))
            current_para = []

    if page_lines:
        if current_para:
            page_paras.append(' '.join(current_para))
        yield page_lines, page_paras


def render_page(page_lines, header: str, page_no: int, font, header_font):
    """Draw one page. Returns (image, reference_text)."""
    img = Image.new('L', (PAGE_W, PAGE_H), 255)
    draw = ImageDraw.Draw(img)
    ref_lines = []

    # Running header, centered -- the kind of repeated furniture a corpus
    # exporter has to learn to drop (see pipeline/exporter/reflow.py).
    if header:
        w = draw.textlength(header, font=header_font)
        draw.text(((PAGE_W - w) / 2, MARGIN_TOP - 130), header, font=header_font, fill=0)
        ref_lines.append(header)

    y = MARGIN_TOP
    step = int(FONT_SIZE * LINE_SPACING)
    for line, is_para_start in page_lines:
        x = MARGIN_X + (PARA_INDENT if is_para_start else 0)
        draw.text((x, y), line, font=font, fill=0)
        ref_lines.append(line)
        y += step

    footer = str(page_no)
    w = draw.textlength(footer, font=header_font)
    draw.text(((PAGE_W - w) / 2, PAGE_H - MARGIN_BOTTOM + 60), footer, font=header_font, fill=0)
    ref_lines.append(footer)

    return img, '\n'.join(ref_lines)


# ─────────────────────────────────────────────────────────────
# Synthetic scan degradation
# ─────────────────────────────────────────────────────────────
# Level 0 is the clean render. Levels 1-3 stack blur, sensor noise, JPEG
# requantization, skew and an illumination gradient -- the artifacts a flatbed
# or phone capture actually introduces. The parameters are a plausible ladder,
# not a fit to measured scanner output; see the honesty note above.

DEGRADE_LEVELS = {
    0: {},
    1: {'blur': 0.4, 'noise': 3.0, 'jpeg': 88, 'skew': 0.4, 'illum': 0.10},
    2: {'blur': 0.8, 'noise': 7.0, 'jpeg': 72, 'skew': 1.2, 'illum': 0.22},
    3: {'blur': 1.3, 'noise': 13.0, 'jpeg': 55, 'skew': 2.5, 'illum': 0.35, 'downscale': 0.72},
}


def _apply_illumination(arr: np.ndarray, strength: float, rng: random.Random) -> np.ndarray:
    """Darken one edge, the way a bound book falls away from the platen."""
    h, w = arr.shape[:2]
    axis = rng.choice(['left', 'right', 'top', 'bottom'])
    ramp = np.linspace(1.0 - strength, 1.0, w if axis in ('left', 'right') else h)
    if axis in ('right', 'bottom'):
        ramp = ramp[::-1]
    gradient = np.tile(ramp, (h, 1)) if axis in ('left', 'right') else np.tile(ramp[:, None], (1, w))
    return arr * gradient


def degrade(img: Image.Image, level: int, seed: int) -> Image.Image:
    """Apply the level-`level` degradation ladder. Level 0 returns a copy."""
    params = DEGRADE_LEVELS[level]
    if not params:
        return img.copy()

    rng = random.Random(seed)
    out = img.convert('L')

    if params.get('skew'):
        angle = rng.uniform(-params['skew'], params['skew'])
        out = out.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

    if params.get('blur'):
        out = out.filter(ImageFilter.GaussianBlur(params['blur']))

    arr = np.asarray(out, dtype=np.float32)

    if params.get('illum'):
        arr = _apply_illumination(arr, params['illum'], rng)

    if params.get('noise'):
        arr = arr + np.random.default_rng(seed).normal(0.0, params['noise'], arr.shape)

    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode='L')

    if params.get('downscale'):
        f = params['downscale']
        small = out.resize((max(1, int(out.width * f)), max(1, int(out.height * f))), Image.BILINEAR)
        out = small.resize(out.size, Image.BICUBIC)

    if params.get('jpeg'):
        buf = io.BytesIO()
        out.save(buf, format='JPEG', quality=params['jpeg'])
        buf.seek(0)
        out = Image.open(buf).convert('L')

    return out


# ─────────────────────────────────────────────────────────────

def build(args) -> int:
    if not os.path.exists(args.font):
        raise SystemExit(f"Font not found: {args.font} (run python setup.py first)")

    docx_files = sorted(glob.glob(os.path.join(args.docx_dir, '*.docx')))
    if not docx_files:
        raise SystemExit(f"No .docx files under {args.docx_dir}")

    levels = [int(v) for v in args.degrade.split(',')]
    for lv in levels:
        if lv not in DEGRADE_LEVELS:
            raise SystemExit(f"Unknown degrade level {lv}; valid: {sorted(DEGRADE_LEVELS)}")

    font = ImageFont.truetype(args.font, FONT_SIZE)
    header_font = ImageFont.truetype(args.font, int(FONT_SIZE * 0.62))
    probe = ImageDraw.Draw(Image.new('L', (10, 10)))

    max_width = PAGE_W - 2 * MARGIN_X
    lines_per_page = int((PAGE_H - MARGIN_TOP - MARGIN_BOTTOM) / (FONT_SIZE * LINE_SPACING))

    rng = random.Random(args.seed)
    rng.shuffle(docx_files)

    os.makedirs(args.out, exist_ok=True)
    written = docs_used = 0

    for docx_path in docx_files:
        if docs_used >= args.docs:
            break
        name = os.path.splitext(os.path.basename(docx_path))[0]

        try:
            paragraphs = clean_paragraphs(docx_path)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue

        if len(paragraphs) < 3:
            print(f"  [skip] {name}: only {len(paragraphs)} usable paragraph(s)")
            continue

        # A running header modelled on real book furniture: the title, repeated.
        header = ' '.join(paragraphs[0].split()[:3])

        pages = list(paginate(paragraphs, font, probe, max_width, lines_per_page))
        pages = pages[:args.pages_per_doc]
        if not pages:
            continue

        for n, (page_lines, page_paras) in enumerate(pages, 1):
            img, reference = render_page(page_lines, header, n, font, header_font)
            for level in levels:
                stem = f"{name}_p{n:03d}" + (f"_d{level}" if level else "")
                degrade(img, level, seed=args.seed + n + level).save(
                    os.path.join(args.out, stem + '.png'))
                with open(os.path.join(args.out, stem + '.txt'), 'w', encoding='utf-8') as fh:
                    fh.write(reference)
                with open(os.path.join(args.out, stem + '.para.txt'), 'w', encoding='utf-8') as fh:
                    fh.write('\n\n'.join(page_paras))
                written += 1
            print(f"  {name} p{n:03d}: {len(page_lines):2d} lines, {len(reference):5d} chars, "
                  f"{len(page_paras)} paragraph(s) -> {len(levels)} variant(s)")

        docs_used += 1

    print(f"\nWrote {written} page/reference pairs from {docs_used} document(s) to {args.out}")
    if written:
        print("These are typeset pages, not scans -- a regression gate, not evidence about "
              "real book scans. Drop real page.jpg + page.txt pairs into the same directory "
              "for the honest test.")
    return 0 if written else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--docx-dir', default=DEFAULT_DOCX_DIR,
                    help=f'source .docx corpus (default: {DEFAULT_DOCX_DIR})')
    ap.add_argument('--out', default=DEFAULT_OUT_DIR,
                    help=f'output directory (default: {DEFAULT_OUT_DIR})')
    ap.add_argument('--font', default=FONT_PATH, help=f'TTF to typeset with (default: {FONT_PATH})')
    ap.add_argument('--docs', type=int, default=12, help='documents to sample (default: 12)')
    ap.add_argument('--pages-per-doc', type=int, default=2, help='pages per document (default: 2)')
    ap.add_argument('--degrade', default='0',
                    help='comma-separated degradation levels 0-3 (default: 0 = clean render)')
    ap.add_argument('--seed', type=int, default=20260901, help='sampling/noise seed')
    ap.add_argument('--list', action='store_true', help='count the source corpus and exit')
    args = ap.parse_args()

    if args.list:
        files = sorted(glob.glob(os.path.join(args.docx_dir, '*.docx')))
        total = sum(os.path.getsize(f) for f in files)
        print(f"{len(files)} .docx files in {args.docx_dir} ({total / 1e9:.2f} GB)")
        return 0

    return build(args)


if __name__ == '__main__':
    sys.exit(main())
