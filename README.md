# PRAGNA_OCR: ಕನ್ನಡ OCR & ಸ್ವಯಂ ತಿದ್ದುಪಡಿ

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract%20LSTM-orange.svg)](https://github.com/tesseract-ocr/tesseract)
[![Indic NLP](https://img.shields.io/badge/Indic%20NLP-Kannada%20Sandhi-cyan.svg)](#indic-morphological-engine)

A high-accuracy, production-grade Indic OCR and morphological correction pipeline designed for Kannada documents. **PRAGNA_OCR** combines optical character recognition for scanned/searchable PDFs and high-resolution images with dynamic Sandhi morphology, subscript ligature repair matrices, weighted Levenshtein distance, N-gram language models, and layout-preserving Unicode PDF export.

---

## 🌟 Key Capabilities

### 1. 📑 Universal Document Ingestion & Rasterization
- **Searchable PDFs**: Fast digital text extraction with PyMuPDF layout analysis.
- **Scanned Multi-Page PDFs**: High-resolution rasterization (200, 300, 400 DPI) via `pdf2image` and Poppler.
- **Image Documents**: High-contrast preprocessing for PNG, JPG, JPEG, TIFF, BMP, and WEBP.
- **Real-Time Text Stream**: Direct text autocorrection with sub-second latency.

### 2. 🔠 Multi-Language Indic OCR Subsystem
- Tesseract LSTM neural model configured with Indic language packs (`kan`, `eng`, `san`, `hin`, `tam`, `tel`).
- Baseline-aware reading order sorting to prevent cross-column text interleaving.
- Text block segmentation and alignment classification (`Left`, `Center`, `Right`).

### 3. 🎯 Dynamic Indic Morphological & Repair Engine
- **Universal Script Normalization**: Repha transforms (`...೯` $\rightarrow$ `ರ್...`), zero-to-anusvara conversion (`೦` $\rightarrow$ `ಂ`), and noise speckle cleaning.
- **Subscript & Optical Ligature Repairs**: Confusion penalty matrices for visual OCR ambiguities (`ಂಜ ↔ ಂದ`, `ವ್ಮ ↔ ಮ್ಮ`, `ಸ್ಮ ↔ ಷ್ಮೆ`, `ಶ ↔ ತ`, `ಹ ↔ ಯ`, `ಳ ↔ ಕ`).
- **Sandhi & Agglutinative Morphology**: Longest-match suffix stripping against LibreOffice Hunspell `kn_IN` dictionary (46,000+ words) and affix rules (`kn_IN.aff`).
- **No-Regression Protection**: Suffix recognition for honorifics, plurals, auxiliary past verbs, and participles (`-ಅಂತಹವಳು`, `-ವಿತ್ತು`, `-ವರು`, `-ವರೆವಿಗೂ`).
- **N-Gram Context Ranking**: Unigram and bigram frequency scoring to resolve homophones and contextual tokens.

### 4. 📄 Layout-Preserving Export & Rich Visual Diffs
- **PDF Generation**: Layout-preserved digital PDF output via FPDF2 with embedded Google Noto Sans Kannada Unicode TTF font.
- **Diff Classification System**:
  - 🔵 **Blue (`ocr_repair`)**: Subscript ottu, consonant cluster, and optical glyph repairs.
  - 🟢 **Green (`word_correction`)**: Suffix normalization, Sandhi reconstruction, and dictionary corrections.
  - 🟡 **Yellow (`hybrid`)**: Combined optical and morphological corrections.
- **Export Formats**: PDF, per-page TXT, unified combined TXT, and machine-readable JSON reports.

### 5. 🎨 Dual Interfaces
- **Interactive Web App**: Modern Apple-grade dark-theme dashboard with real-time text autocorrect, drag-and-drop document pipeline, and live Server-Sent Events (SSE) progress tracking.
- **Command-Line Interface (CLI)**: High-speed batch processing for folders and automated pipelines.

---

## 🚀 Quickstart & Installation

### 1. System Dependencies

#### macOS (Homebrew)
```bash
brew install tesseract tesseract-lang poppler
```

#### Ubuntu / Debian
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-kan tesseract-ocr-eng poppler-utils
```

---

### 2. Clone & Setup Python Environment

```bash
git clone https://github.com/dipankarchettri/PRAGNA_OCR.git
cd PRAGNA_OCR

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify assets (dictionary, fonts, tessdata all ship with the repo)
python setup.py
```

### 3. Drop in the n-gram language model ⚠️

**The pipeline runs without this file, but not well — and it will not tell you it is
missing.** Everything else the project needs is in the repo; this one file is 358 MB,
past GitHub's 100 MB per-file limit, so it is distributed separately. Ask the team for
`ngram_model.pkl.gz` and put it here:

```
PRAGNA_OCR/
└── data/
    └── ngram_model.pkl.gz     ← here, exactly this name
```

No config or environment variable — the path is resolved relative to the repo. Then
confirm it actually loaded:

```bash
./venv/bin/python -c "
from pipeline import init_pipeline; init_pipeline()
from pipeline.correction.ngram import has_corpus_counts, corpus_frequency
print('corpus loaded:', has_corpus_counts())
print('ಕಾಫಿ frequency:', corpus_frequency('ಕಾಫಿ'))"
```

You want `corpus loaded: True`. If it prints `False` the file was not found, and the
failure is silent: the engine falls back to unigram counts padded from the dictionary
word list, which disables the corpus-frequency gate that decides whether a proposed
correction is allowed at all. It still runs and still "corrects" — as a materially less
precise engine than the one every benchmark number in this repo describes. A clean
startup is *not* evidence the model loaded; check the line.

First load takes ~30 s and about 4.7 GB of RAM. The web dashboard warms it on a
background thread at boot, so the page serves immediately and `/api/system-status`
reports `warmup` as `pending` / `loading` / `ready` / `failed`.

### What ships in the repo, and what does not

| Asset | Location | In git? | Needed for |
|---|---|---|---|
| Kannada dictionary, 589,521 entries | `data/kn_IN.dic.gz` | ✅ 2.5 MB, gzipped | everything — read in place, no unpacking |
| Hunspell affix rules | `data/kn_IN.aff` | ✅ | suffix expansion |
| Tesseract LSTM models | `tessdata/` | ✅ | OCR |
| Noto Sans Kannada + Latin fallback | `web/static/fonts/` | ✅ | PDF export |
| Real page/transcript fixtures | `tests/fixtures/real/` | ✅ | the honest accuracy measurement |
| **N-gram language model** | `data/ngram_model.pkl.gz` | ❌ **358 MB — get from the team** | correction quality |
| Filtered literary corpus | `data/corpus_literary/` | ❌ | only `correction_bench.py --synthetic` |

A note on the dictionary, since it is easy to break: an uncompressed `data/kn_IN.dic`
takes precedence over the `.gz` if present. That is deliberate — drop a different `.dic`
in to experiment — but it means `setup.py`'s download fallback, which fetches the *stock*
LibreOffice `kn_IN` (19,645 entries, 3% of this vocabulary), would silently downgrade the
engine. `setup.py` skips it whenever the `.gz` is present, and says so loudly if it ever
does run.

---

## 💻 Usage

### 1. Web Application

Launch the local web dashboard:
```bash
python web/app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

* Features:
  * **Live Text Editor**: Instant token-by-token autocorrect as you type.
  * **Document OCR & Pipeline**: Drag-and-drop PDFs/images with real-time progress, side-by-side diff viewers, and PDF downloads.
  * **Engine Architecture**: Live Tesseract and Hunspell lexicon diagnostics.

---

### 2. Command Line Interface (CLI)

```bash
# Process a single scanned PDF or Image document
python cli.py scan.pdf --lang kan --dpi 400 --output-dir ./results

# Process an image file and save preprocessed sheets
python cli.py book_page.jpg --lang kan --save-images

# Pages with mixed layout (title pages, tables of contents, multi-column)
python cli.py frontmatter.pdf --psm 3

# Drop low-confidence words instead of letting the corrector guess at them
python cli.py faint_scan.jpg --min-confidence 40

# Direct text correction from terminal
python cli.py --text "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತದೆ"

# Batch process an entire folder of scans
python cli.py --batch ./scans/ --output-dir ./batch_results/

# Output machine-readable JSON report
python cli.py scan.pdf --json
```

---

## 🧪 Testing & Verification

Run the automated test suite:
```bash
./venv/bin/python tests/test_pipeline.py             # 26 tests
./venv/bin/python tests/test_correction_precision.py # 15 — words that must NOT change
./venv/bin/python tests/test_reflow.py               # 14 — paragraph reconstruction
```

### Measuring the correction engine

`tools/correction_bench.py` is the gate every change to `pipeline/correction/` has to
clear. It reports CER and WER before/after, and — more importantly — splits changes into
`fixed` / `broke` / `other`:

```bash
./venv/bin/python tools/correction_bench.py --pages 'tests/fixtures/real/*.png'
```

**CER alone is not the metric.** This corpus feeds LLM training, where a confidently
wrong "fix" silently corrupts the data in a way that is far harder to catch downstream
than an untouched OCR error. A change that lowers CER while raising `broke` is a
regression here, whatever it does to the headline number.

Current state on the nine real page/transcript pairs — the honest test, since they are
genuine book scans rather than typeset pages:

| | CER | WER | fixed | broke | precision |
|---|---|---|---|---|---|
| uncorrected Tesseract | 0.0558 | 0.2889 | — | — | — |
| after correction | **0.0520** | **0.2697** | 10 | **0** | **1.000** |

Requires the n-gram model to be in place — without it these numbers do not reproduce.

### Tuning OCR accuracy

`tools/ocr_bench.py` sweeps Tesseract settings over sample pages and reports which
combination produces the cleanest Kannada, so configuration changes can be measured
rather than guessed at:

```bash
# Compare page segmentation modes and resolution handling
./venv/bin/python tools/ocr_bench.py page.jpg --psm 3,4,6 --upscale 0,1

# Write each configuration's output for side-by-side reading
./venv/bin/python tools/ocr_bench.py page.jpg --dump ./bench_out
```

It ranks by two ground-truth-free proxies — the share of Kannada tokens the correction
engine recognizes, and the count of Latin-script runs (misrecognized glyphs, which are
unrecoverable downstream). Drop a `.txt` transcript next to an image with the same
basename and it reports true character error rate and ranks by that instead.

### OCR defaults

| Setting | Default | Rationale |
|---|---|---|
| `--lang` | `kan` | Adding `eng` to a monolingual Kannada page makes Tesseract emit Latin for ambiguous glyphs; the correction engine skips non-Kannada tokens, so that output is unrecoverable. |
| `--psm` | `3` | Automatic segmentation. Wins or ties on 8 of 9 real pages (mean raw CER 0.0549 vs 0.0585 for `6`), and improves the corrected result too. Was `6` until real transcripts existed — that choice came from proxy metrics that reward the confident output `6` produces *while segmenting a page wrongly*. Try `6` on a page that genuinely is one uniform block. |
| `--oem` | `1` | LSTM only. The `tessdata_best` models ship no legacy engine. |
| `--dpi` | `400` | Kannada ottakshara (subscript conjuncts) alias badly at 300. |

Low-resolution images are automatically upscaled to roughly 300 DPI before OCR
(`pipeline/ingestion/image_processor.py`). Phone photos and web-sized scans are often
near 100 DPI, at which conjuncts collapse into a smudge that no dictionary can repair.

The bundled `tessdata/kan.traineddata` is the `tessdata_best` LSTM model; the smaller
stock model is kept at `tessdata_standard/` for comparison.

---

## 📂 Repository Structure

```
PRAGNA_OCR/
├── pipeline/                   # Core OCR & Morphological pipeline
│   ├── ingestion/              # PDF & image rasterization and preprocessing
│   │   ├── pdf_processor.py
│   │   └── image_processor.py
│   ├── ocr/                    # Tesseract multi-language engine & layout extraction
│   │   └── tesseract_engine.py
│   ├── correction/             # Indic Morphological Correction Engine
│   │   ├── tokenizer.py        # Tokenizer & punctuation handling
│   │   ├── graphemes.py        # Akshara (grapheme-cluster) segmentation
│   │   ├── ocr_repairs.py      # Script-level Repha, joiner & Unicode normalization
│   │   ├── morphology.py       # Kannada Sandhi, suffix stripping & decomposition
│   │   ├── dictionary.py       # Hunspell kn_IN loader & core vocabulary
│   │   ├── edit_distance.py    # Weighted Levenshtein & optical confusion matrices
│   │   ├── ngram.py            # N-Gram language model scoring
│   │   └── corrector.py        # Dynamic candidate generation & ranking
│   └── exporter/               # PDF, TXT & JSON exporters
│       ├── pdf_generator.py    # Layout-preserved Unicode PDF builder
│       ├── reflow.py           # Line boxes → paragraphs for the training corpus
│       └── text_exporter.py    # Structured text and JSON report builder
├── web/                        # Flask Web Application & Dashboard
│   ├── app.py                  # REST API & SSE streaming endpoints
│   ├── static/
│   │   ├── css/style.css       # Apple-grade glassmorphic styles
│   │   ├── js/app.js           # Real-time event streaming & interactive diffs
│   │   └── fonts/              # Noto Sans Kannada Unicode TTF
│   └── templates/
│       └── index.html          # Web dashboard template
├── tools/                      # Measurement & maintenance scripts
│   ├── correction_bench.py     # CER/WER + fixed/broke — the gate for engine changes
│   ├── ocr_bench.py            # Tesseract PSM/upscale sweeps
│   ├── build_eval_set.py       # Typeset synthetic eval pages from clean text
│   └── build_ngram_model.py    # Rebuild the n-gram cache (maintainer-only)
├── tessdata/                   # Tesseract LSTM models (tessdata_best) — in git
├── data/
│   ├── kn_IN.dic.gz            # 589,521-entry dictionary — in git, read in place
│   ├── kn_IN.aff               # Hunspell affix rules — in git
│   └── ngram_model.pkl.gz      # ⚠️ NOT in git (358 MB) — get from the team
├── tests/                      # Automated test suite
│   ├── test_pipeline.py
│   ├── test_correction_precision.py
│   ├── test_reflow.py
│   └── fixtures/real/          # Real page + transcript pairs — the honest test
├── cli.py                      # Command-line interface
├── setup.py                    # Asset verifier & environment validator
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
