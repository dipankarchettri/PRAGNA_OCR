# ಕನ್ನಡ OCR & ಸ್ವಯಂ ತಿದ್ದುಪಡಿ — Kannada OCR & Autocorrect Pipeline

A unified, high-accuracy Indic OCR and morphological correction pipeline that combines optical character recognition for scanned/searchable PDFs and images with deep Kannada morphological decomposition, Indic OCR repair rules, weighted Levenshtein distance, and layout-preserved PDF export.

---

## Key Features

- 📑 **Universal Document Ingestion**:
  - Direct digital text extraction for searchable PDFs (via PyMuPDF).
  - High-resolution multi-page rasterization (at 200, 300, or 400 DPI) for scanned PDFs (via `pdf2image` + Poppler).
  - Image document OCR (PNG, JPG, TIFF, BMP, WEBP).
  - Direct text input correction with instant response.

- 🔠 **Multi-Language Indic OCR**:
  - Tesseract OCR engine integration.
  - Multi-language combinations (e.g., `kan+eng`, `san+kan`, `hin+kan`, `tam+kan`, `tel+kan`).
  - Layout-aware bounding box and text alignment extraction (`Left`, `Center`, `Right`).

- 🎯 **Indic OCR Normalization & Morphological Engine**:
  - **OCR Rule Repairs**: Restores missing Virama/Halant (`U+0CCD`), repairs consonant and ottu collapsing (`ಸಸ` $\rightarrow$ `ಸ್ಸ`, `ತತ` $\rightarrow$ `ತ್ತ`, `ಶಿಕಷ` $\rightarrow$ `ಶಿಕ್ಷ`), normalizes short/long vowels (`ಜಿವನ` $\rightarrow$ `ಜೀವನ`), and repairs prefix corruption (`ಪ` $\rightarrow$ `ಪ್ರ`).
  - **Morphological Decomposition**: Strips agglutinative Kannada case markers, verbal inflections, and plural suffixes against LibreOffice Hunspell `kn_IN.dic` (45,000+ words) and affix rules (`kn_IN.aff`).
  - **Sandhi Rule Engine**: Euphonic reconstruction for Kannada verb and noun sandhi.
  - **Weighted Levenshtein Edit Distance**: Lower penalty for visually similar Indic glyph pairs (`ಕ`/`ಖ`, `ಗ`/`ಘ`, `ತ`/`ಥ`, `ಪ`/`ಫ`, `ಣ`/`ನ`, `ಶ`/`ಷ`).
  - **N-Gram Context Ranking**: Unigram and bigram candidate scoring.

- 📄 **Layout-Preserving Export & Diffs**:
  - Layout-preserved PDF generation using FPDF2 and Google's Noto Sans Kannada Unicode TTF.
  - Per-page `.txt` exports and unified text file.
  - Machine-readable JSON analysis report with word-by-word correction diffs and edit costs.

- 🎨 **Dual Interfaces**:
  - **Interactive Web App**: Modern glassmorphic dark-theme dashboard with live text correction, file drag-and-drop, side-by-side diff viewers, and engine diagnostic status.
  - **Command-Line Interface (CLI)**: Fast CLI for batch processing folders and single documents.

---

## Installation & Setup

### 1. Prerequisites (macOS / Linux)

```bash
# macOS (Homebrew)
brew install tesseract tesseract-lang poppler

# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-kan tesseract-ocr-eng poppler-utils
```

### 2. Python Environment Setup

```bash
cd kannada_ocr_pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download / Initialize dictionaries & Noto Sans Kannada font
python setup.py
```

---

## Usage

### 1. Command Line Interface (CLI)

```bash
source venv/bin/activate

# Process a single PDF or Image document
python cli.py sample.pdf --lang kan+eng --dpi 300 --output-dir ./results

# Process an image file and save intermediate images
python cli.py scan.png --lang kan+eng --save-images

# Direct text correction
python cli.py --text "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತದೆ"

# Batch process an entire directory of PDFs and images
python cli.py --batch ./scans/ --output-dir ./batch_results/

# Print machine-readable JSON output
python cli.py sample.pdf --json
```

### 2. Web Application

```bash
source venv/bin/activate
python web/app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## Project Structure

```
kannada_ocr_pipeline/
├── pipeline/                   # Core pipeline modules
│   ├── ingestion/              # PDF & image loading, layout analysis
│   │   ├── pdf_processor.py
│   │   └── image_processor.py
│   ├── ocr/                    # Tesseract multi-language engine
│   │   └── tesseract_engine.py
│   ├── correction/             # Morphological & Indic OCR correction
│   │   ├── tokenizer.py
│   │   ├── ocr_repairs.py
│   │   ├── morphology.py
│   │   ├── dictionary.py
│   │   ├── edit_distance.py
│   │   ├── ngram.py
│   │   └── corrector.py
│   └── exporter/               # PDF, TXT & JSON exporters
│       ├── pdf_generator.py
│       └── text_exporter.py
├── web/                        # Flask Web Dashboard
│   ├── app.py
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/app.js
│   │   └── fonts/NotoSansKannada-Regular.ttf
│   └── templates/index.html
├── data/                       # Hunspell kn_IN.dic & kn_IN.aff
├── tests/                      # Automated test suite
│   └── test_pipeline.py
├── cli.py                      # CLI tool
├── setup.py                    # Asset downloader & validator
├── requirements.txt
└── README.md
```
