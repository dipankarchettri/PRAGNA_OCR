# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PRAGNA_OCR is a Kannada (`kn_IN`) OCR and morphological autocorrection pipeline. It ingests scanned/searchable PDFs and images, runs Tesseract LSTM OCR where needed, repairs Indic-script OCR artifacts, corrects spelling via Sandhi-aware morphology + weighted edit distance + n-gram scoring, and exports layout-preserving Unicode PDFs, text, and JSON diff reports. There is both a Flask web dashboard and a CLI.

## Goal

The end purpose of this pipeline is to scan Kannada books at volume and turn them into a clean text dataset for training a Kannada LLM. This means CER/WER on real book pages (not just short benchmark snippets) is the metric that actually matters, and correctness/precision of automated corrections matters more than coverage — a wrong "fix" silently corrupts training data in a way that's much harder to catch downstream than leaving an OCR error untouched.

**Excluded: Krutrim (Chitrapathak / Krutrim-1 / Krutrim-2 / Chitrarth) models.** Chitrapathak-2 (Krutrim AI Labs) is currently one of the strongest OCR systems measured on Kannada specifically (word-level ANLS ~18.8, beating Surya OCR's ~24.4, close behind Gemini-2.5-Flash's ~17.2 — lower is better), but it and all other Krutrim-branded models ship under the **Krutrim Community License Agreement v1.0**, whose §5 ("Restriction on Use to Compete with Krutrim") bans using the software to "develop, market, sell, or support competing products or services" — with no research/non-commercial carve-out and no scale/revenue threshold, unlike the license's own MAU-gated definition of "Commercial Use" elsewhere. Since Krutrim itself ships multilingual Indic LLMs (Krutrim-1, Krutrim-2) with Kannada support, using their OCR model to build a Kannada LLM training corpus sits squarely inside that non-compete clause regardless of project scale. Do not suggest or reintroduce Krutrim models for this pipeline.

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
./venv/bin/python tests/test_correction_precision.py   # words that must NOT change

# Run a single test case/method
./venv/bin/python -m unittest tests.test_pipeline.TestKannadaPipeline.test_dynamic_ocr_repairs -v

# Web dashboard
./venv/bin/python web/app.py          # http://127.0.0.1:5010 (PORT env overrides)

# CLI: single file, direct text, batch folder
python cli.py scan.pdf --lang kan --dpi 400 --output-dir ./results
python cli.py --text "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತದೆ"
python cli.py --batch ./scans/ --output-dir ./batch_results/

# Sweep Tesseract PSM/upscale settings against sample pages to measure OCR quality
# (ranks by % Kannada tokens the corrector recognizes + Latin-script "unrecoverable" runs;
# drop a same-basename .txt transcript next to the image to get real CER instead)
./venv/bin/python tools/ocr_bench.py page.jpg --psm 3,4,6 --upscale 0,1
./venv/bin/python tools/ocr_bench.py page.jpg --dump ./bench_out

# Build the evaluation fixture set (deterministic; images are gitignored)
./venv/bin/python tools/build_eval_set.py --docs 12 --degrade 0,1,2,3

# Measure the correction engine: CER/WER before vs after + fixed/broke counts
./venv/bin/python tools/correction_bench.py --pages 'tests/fixtures/eval/*_p0??.png'
./venv/bin/python tools/correction_bench.py --synthetic 200   # smoke test, no ground truth
```

Always run `tests/test_pipeline.py` and `tests/test_correction_precision.py` after touching
anything under `pipeline/correction/` or `pipeline/exporter/`.

## Measuring changes

`tools/build_eval_set.py` typesets clean Kannada book text from
`data/kanaja_docx_raw/` into page images at four degradation levels, writing the exact text
it laid out as the reference. `tools/correction_bench.py` then scores the engine on them.

**Read `tools/build_eval_set.py`'s docstring before trusting a number from it.** These are
typeset pages, not scans — a deterministic regression gate, not evidence about real book
scans. Real page/transcript pairs dropped into `tests/fixtures/eval/` as `page.jpg` +
`page.txt` are the honest test. (The obvious shortcut — LibreOffice `.docx`→PDF, then read
the PDF's text layer — was tried and does not work here: on this corpus LibreOffice emits
justified Kannada with **no space characters at all** and duplicates matras, so the
"reference" was itself wrong. Hence typesetting.)

**CER alone is not the metric.** The bench reports `fixed` / `broke` separately on purpose:
a wrong correction silently corrupts training data, where an untouched OCR error stays
visible downstream. An engine that lowers CER while raising `broke` is a regression.

Measured on 24 typeset pages (Tesseract `kan`, psm 6), before → after the space-merge
precision fix:

| | CER | WER | fixed | broke | precision |
|---|---|---|---|---|---|
| uncorrected baseline | 0.0047 | 0.0310 | — | — | — |
| corrector, before fix | 0.0056 | 0.0450 | 2 | 50 | 0.038 |
| corrector, after fix | **0.0050** | **0.0335** | 2 | **4** | **0.333** |

Note what the baseline says: Tesseract reads these clean pages at 0.47% CER, so there is
almost nothing for the corrector to fix, and it is still slightly net-negative here. Its
value has to show up on degraded and genuinely scanned pages — which is what the `--degrade`
ladder and real fixtures are for.

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
6. `dictionary.py` — loads `data/kn_IN.dic` + `data/kn_IN.aff` as the vocabulary source of truth. The `.dic` is the extended 141,115-entry list (the stock Hunspell `kn_IN` was 19,645; `data/kn_IN.dic.bak` keeps it); suffix-rule expansion brings the in-memory set to 182,294 forms.

`corrector.py` (`suggest_kannada_word`, `correct_text`, `correct_layout_lines`) combines all of the above and classifies each fix as `ocr_repair` (blue), `word_correction` (green), or `hybrid` (yellow) — this classification drives the diff coloring in the web UI.

**Non-text line filtering:** `correct_layout_lines()` flags each OCR'd line as `is_likely_non_text` when its mean Tesseract confidence falls below `NON_TEXT_LINE_CONFIDENCE` (40) -- below that, Tesseract isn't misreading degraded prose, it's hallucinating text rows out of non-text content (embedded photos, decorative banner art, logos). Validated against 7 real ground-truth pages spanning the hard real cases (faded scans, heavy Sanskrit vocabulary the dictionary doesn't cover, degraded photocopies): every genuine body-text line across all of them still cleared 55 confidence; only actual graphic content ever dropped below 40, into single digits. Deliberately confidence-only, not combined with a dictionary valid-word-ratio check -- ratio misfires on real Sanskrit-vocabulary text (0% dictionary hits at completely normal confidence), so it can't safely gate this. `pipeline/__init__.py` excludes flagged lines from the training-corpus text (`corrected_text` / exported `.txt`) but keeps them in `raw_text` and the generated PDF (`all_layout_lines`, unfiltered) for audit/visual fidelity -- nothing is silently discarded, only excluded from the clean-corpus output. Never applies to PDF-digital-extraction lines (no `conf` field -- never OCR'd from pixels).

**Zero-hardcoding rule (strict, enforced in review):** never add word-specific or stem-specific mappings to `CONFUSION_PAIRS`, `GLYPH_CONFUSIONS`, `ocr_repairs.py`, or anywhere else in the correction engine — no `'ಧ್ಯಯ' -> 'ಧ್ಯಯನ'`-style special cases. Every fix must fall out of the universal mechanisms above (script normalization, optical confusion matrices, longest-match morphology, 1-edit search, n-gram ranking). New vocabulary belongs in `data/kn_IN.dic`, not as a conditional branch in code. This history is visible in git log (search for "hardcoding" / "stem-specific").

### Other pipeline modules

- `pipeline/ingestion/` — `pdf_processor.py` (PyMuPDF digital-text + page classification + `pdf2image` rasterization at configurable DPI) and `image_processor.py` (EXIF auto-rotation, `normalize_resolution` auto-upscales low-DPI input toward ~300 DPI since Kannada ottakshara/subscript conjuncts alias badly below that). `preprocess_for_ocr()` runs deskew (via `detect_skew_angle`'s projection-profile search) on every page right before OCR, for both the image-file path (inside `load_and_preprocess_image`) and the PDF-rasterization path in `pipeline/__init__.py` -- validated empirically: near-zero effect on already-straight scans (skew below `MIN_SKEW_DEGREES` is left untouched), meaningful CER/WER improvement on skewed ones (e.g. a synthetic 3° skew: CER 2.65%→1.28%, WER 20%→10%). `enhance_contrast` on `preprocess_for_ocr`/`load_and_preprocess_image` itself defaults to `False` and should stay that way: a blanket +20% contrast boost applied unconditionally can take a clean scan from near-perfect OCR to complete word-salad, confirmed on a real document, and grayscale std-dev/percentile-spread showed no clean signal separating the pages it helps from the one it catastrophically breaks. Instead, `pipeline/__init__.py`'s `_ocr_with_adaptive_contrast()` (used by `process_document` when `adaptive_contrast=True`, the default; disable via CLI `--no-adaptive-contrast`) OCRs each page twice -- once untouched, once with the boost -- and keeps whichever run **Tesseract's own mean word confidence** scores higher, ties going to the unboosted run. Validated against 7 real ground-truth pages: catches genuine wins on faded scans (CER 16.68%→8.80% on one), stays neutral on near-ties, and correctly discards the boosted run on the catastrophic-corruption case (confidence drops 25-35 points there, vs. a ≤1-point wobble on genuine improvements) -- confidence *compared between two runs* separates the cases that a static image-statistics threshold could not. Costs a second Tesseract pass per page.
- `pipeline/ocr/tesseract_engine.py` — multi-language Tesseract wrapper (`ocr_image`, `ocr_image_with_layout`), baseline-aware reading-order sort, alignment classification.
- `pipeline/exporter/` — `pdf_generator.py` (FPDF2 + embedded `NotoSansKannada-Regular.ttf`, must use `pdf.epw` and `new_x="LMARGIN", new_y="NEXT"` for correct line wrapping) and `text_exporter.py` (per-page/combined TXT, JSON report).

### Web app (`web/app.py`)

Flask + SSE: uploads go through `_SESSIONS` for progress streaming back to the browser during long OCR jobs; `init_pipeline()` runs via `@app.before_request`. Frontend is `web/static/js/app.js` (live autocorrect + diff viewer) and `web/templates/index.html`.

**Startup warmup.** `start_warmup()` loads the dictionary and n-gram model on a background
thread at boot, so the server binds its port and serves the page immediately instead of
stalling on the ~33 s model load. `/api/system-status` exposes `warmup`
(`pending`/`loading`/`ready`/`failed`) and the frontend reads it.

### OCR defaults and why

| Setting | Default | Rationale |
|---|---|---|
| `--lang` | `kan` | Adding `eng` on a monolingual Kannada page makes Tesseract emit Latin glyphs for ambiguous shapes; the corrector skips non-Kannada tokens, so that output is unrecoverable. |
| `--psm` | `6` | Best measured on single-column book pages; use `3` for mixed/multi-column layouts. |
| `--oem` | `1` | LSTM only — the bundled `tessdata_best` models ship no legacy engine. |
| `--dpi` | `400` | Subscript conjuncts alias badly at 300. |

`tessdata/kan.traineddata` is the `tessdata_best` LSTM model; `tessdata_standard/` holds the smaller stock model kept for comparison via `tools/ocr_bench.py`.
