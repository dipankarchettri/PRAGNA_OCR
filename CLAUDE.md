# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PRAGNA_OCR is a Kannada (`kn_IN`) OCR and morphological autocorrection pipeline. It ingests scanned/searchable PDFs and images, runs Tesseract LSTM OCR where needed, repairs Indic-script OCR artifacts, corrects spelling via Sandhi-aware morphology + weighted edit distance + n-gram scoring, and exports layout-preserving Unicode PDFs, text, and JSON diff reports. There is both a Flask web dashboard and a CLI.

## Setup

```bash
# System deps (Tesseract + Poppler)
sudo apt install tesseract-ocr tesseract-ocr-kan tesseract-ocr-eng poppler-utils   # Debian/Ubuntu
brew install tesseract tesseract-lang poppler                                     # macOS

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Downloads Hunspell kn_IN dictionary, Noto Sans Kannada font, and tessdata models
python setup.py
```

## Common commands

```bash
# Run the full test suite (plain unittest, no pytest)
./venv/bin/python tests/test_pipeline.py

# Run a single test case/method
./venv/bin/python -m unittest tests.test_pipeline.TestKannadaPipeline.test_dynamic_ocr_repairs -v

# Web dashboard
./venv/bin/python web/app.py          # http://127.0.0.1:5000

# CLI: single file, direct text, batch folder
python cli.py scan.pdf --lang kan --dpi 400 --output-dir ./results
python cli.py --text "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತದೆ"
python cli.py --batch ./scans/ --output-dir ./batch_results/

# Sweep Tesseract PSM/upscale settings against sample pages to measure OCR quality
# (ranks by % Kannada tokens the corrector recognizes + Latin-script "unrecoverable" runs;
# drop a same-basename .txt transcript next to the image to get real CER instead)
./venv/bin/python tools/ocr_bench.py page.jpg --psm 3,4,6 --upscale 0,1
./venv/bin/python tools/ocr_bench.py page.jpg --dump ./bench_out
```

Always run `tests/test_pipeline.py` after touching anything under `pipeline/correction/` or `pipeline/exporter/`.

## Architecture

Everything routes through `pipeline/__init__.py`'s two entry points, both of which call `init_pipeline()` first (loads the Hunspell dictionary and trains the n-gram model, once, lazily):

- `process_text_input(text)` — direct string in, corrected string out. Used by CLI `--text` and the web app's live autocorrect endpoint.
- `process_document(input_path, ...)` — full file pipeline: ingest → OCR/extract → correct → export. Used by CLI file/batch mode and the web app's upload flow.

### `process_document` per-page decision logic (`pipeline/__init__.py`)

For PDFs, `inspect_pdf()` classifies each page as `searchable`, `mixed`, or `scanned` — the decision is made **per page**, and for `mixed` pages, **per block within the page** (`_page_text_is_valid` / block-level `is_valid`):

- `searchable` pages → `extract_searchable_pdf_layout` (PyMuPDF), no OCR.
- `scanned`/`mixed` pages → `rasterize_page_masking_valid_text` paints over any already-trustworthy digital blocks before rasterizing, so Tesseract only OCRs what the text layer can't be trusted for; OCR line coordinates (pixels) are rescaled to PDF points (`effective_dpi = dpi * (ocr_img.width / img.width)`, `scale = effective_dpi / 72.0`) so they can be merged with digital-extraction lines by vertical position into one reading order.

Images always go through `load_and_preprocess_image` → `ocr_image_with_layout`.

Both paths converge on `correct_layout_lines()` before export.

### `pipeline/correction/` — the correction engine

Stages run in this fixed order (see `corrector.py`):
1. `tokenizer.py` — splits text into Kannada (`ಀ-೿` + ZWJ/ZWNJ) vs. non-Kannada tokens by exact byte offset; non-Kannada tokens (English, numbers, punctuation) are never touched, and `reconstruct(tokens)` must losslessly rebuild the original around them.
2. `ocr_repairs.py` — universal script-level normalization (Repha `...೯` → `ರ್...`, zero-digit→anusvara `೦`→`ಂ`, illegal vowel+matra cleanup), run before dictionary lookup.
3. `morphology.py` — Sandhi/agglutinative suffix stripping via **longest-match-first** against the Hunspell affix rules; `join_root_suffix()` must stay in sync with any suffix list changes here.
4. `edit_distance.py` — weighted Levenshtein with a Kannada optical-glyph confusion matrix (visually similar glyph pairs get lower substitution cost so they outrank generic dictionary stems) plus universal single-character insertion/deletion candidate generation.
5. `ngram.py` — unigram/bigram frequency scoring to rank among surviving candidates. `init_pipeline()` loads a cached real-corpus model from `data/ngram_model.pkl.gz` (gzip+pickle — ~2x faster to load than gzip+JSON at this scale) if present (built via `tools/build_ngram_model.py` from AI4Bharat IndicCorpV2), falling back to unigram-only counts from the dictionary word list if that cache doesn't exist. `score_candidate()` always returns a bounded `[0.0, 0.3]` bonus regardless of which one is loaded, so `corrector.py` never needs corpus-size-dependent tuning.
6. `dictionary.py` — loads `data/kn_IN.dic` + `data/kn_IN.aff` (Hunspell `kn_IN`, ~46k words) as the vocabulary source of truth.

`corrector.py` (`suggest_kannada_word`, `correct_text`, `correct_layout_lines`) combines all of the above and classifies each fix as `ocr_repair` (blue), `word_correction` (green), or `hybrid` (yellow) — this classification drives the diff coloring in the web UI.

**Zero-hardcoding rule (strict, enforced in review):** never add word-specific or stem-specific mappings to `CONFUSION_PAIRS`, `GLYPH_CONFUSIONS`, `ocr_repairs.py`, or anywhere else in the correction engine — no `'ಧ್ಯಯ' -> 'ಧ್ಯಯನ'`-style special cases. Every fix must fall out of the universal mechanisms above (script normalization, optical confusion matrices, longest-match morphology, 1-edit search, n-gram ranking). New vocabulary belongs in `data/kn_IN.dic`, not as a conditional branch in code. This history is visible in git log (search for "hardcoding" / "stem-specific").

### Other pipeline modules

- `pipeline/ingestion/` — `pdf_processor.py` (PyMuPDF digital-text + page classification + `pdf2image` rasterization at configurable DPI) and `image_processor.py` (EXIF auto-rotation, contrast boost, `normalize_resolution` auto-upscales low-DPI input toward ~300 DPI since Kannada ottakshara/subscript conjuncts alias badly below that).
- `pipeline/ocr/tesseract_engine.py` — multi-language Tesseract wrapper (`ocr_image`, `ocr_image_with_layout`), baseline-aware reading-order sort, alignment classification.
- `pipeline/exporter/` — `pdf_generator.py` (FPDF2 + embedded `NotoSansKannada-Regular.ttf`, must use `pdf.epw` and `new_x="LMARGIN", new_y="NEXT"` for correct line wrapping) and `text_exporter.py` (per-page/combined TXT, JSON report).

### Web app (`web/app.py`)

Flask + SSE: uploads go through `_SESSIONS` for progress streaming back to the browser during long OCR jobs; `init_pipeline()` runs via `@app.before_request`. Frontend is `web/static/js/app.js` (live autocorrect + diff viewer) and `web/templates/index.html`.

### OCR defaults and why

| Setting | Default | Rationale |
|---|---|---|
| `--lang` | `kan` | Adding `eng` on a monolingual Kannada page makes Tesseract emit Latin glyphs for ambiguous shapes; the corrector skips non-Kannada tokens, so that output is unrecoverable. |
| `--psm` | `6` | Best measured on single-column book pages; use `3` for mixed/multi-column layouts. |
| `--oem` | `1` | LSTM only — the bundled `tessdata_best` models ship no legacy engine. |
| `--dpi` | `400` | Subscript conjuncts alias badly at 300. |

`tessdata/kan.traineddata` is the `tessdata_best` LSTM model; `tessdata_standard/` holds the smaller stock model kept for comparison via `tools/ocr_bench.py`.
