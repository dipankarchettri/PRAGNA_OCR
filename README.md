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

# Initialize Hunspell lexicon and download Noto Sans Kannada TTF font
python setup.py
```

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
python cli.py scan.pdf --lang kan+eng --dpi 300 --output-dir ./results

# Process an image file and save preprocessed sheets
python cli.py book_page.jpg --lang kan+eng --save-images

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
./venv/bin/python tests/test_pipeline.py
```

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
│   │   ├── ocr_repairs.py      # Script-level Repha & Unicode normalization
│   │   ├── morphology.py       # Kannada Sandhi, suffix stripping & decomposition
│   │   ├── dictionary.py       # Hunspell kn_IN loader & core vocabulary
│   │   ├── edit_distance.py    # Weighted Levenshtein & optical confusion matrices
│   │   ├── ngram.py            # N-Gram language model scoring
│   │   └── corrector.py        # Dynamic candidate generation & ranking
│   └── exporter/               # PDF, TXT & JSON exporters
│       ├── pdf_generator.py    # Layout-preserved Unicode PDF builder
│       └── text_exporter.py    # Structured text and JSON report builder
├── web/                        # Flask Web Application & Dashboard
│   ├── app.py                  # REST API & SSE streaming endpoints
│   ├── static/
│   │   ├── css/style.css       # Apple-grade glassmorphic styles
│   │   ├── js/app.js           # Real-time event streaming & interactive diffs
│   │   └── fonts/              # Noto Sans Kannada Unicode TTF
│   └── templates/
│       └── index.html          # Web dashboard template
├── data/                       # Hunspell kn_IN.dic & kn_IN.aff dictionary files
├── tests/                      # Automated test suite
│   └── test_pipeline.py
├── cli.py                      # Command-line interface
├── setup.py                    # Asset downloader & environment validator
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
