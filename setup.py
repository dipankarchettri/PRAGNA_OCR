#!/usr/bin/env python3
"""
Setup and Asset Initialization for Kannada OCR Pipeline
Downloads Hunspell kn_IN dictionary files and Noto Sans Kannada font.
"""

import os
import shutil
import urllib.request
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FONTS_DIR = os.path.join(BASE_DIR, 'web', 'static', 'fonts')

DIC_PATH = os.path.join(DATA_DIR, 'kn_IN.dic')
# The 589k-entry dictionary ships gzipped (2.5MB vs 18.4MB) and
# pipeline.correction.dictionary reads it in place. If it is present there is
# nothing to fetch -- and nothing MUST be fetched, because a plain .dic takes
# precedence over the .gz, so downloading the 19,645-entry stock LibreOffice
# dictionary here would silently downgrade the pipeline to 3% of its vocabulary.
DIC_GZ_PATH = DIC_PATH + '.gz'
AFF_PATH = os.path.join(DATA_DIR, 'kn_IN.aff')
FONT_PATH = os.path.join(FONTS_DIR, 'NotoSansKannada-Regular.ttf')
# NotoSansKannada-Regular.ttf has no Latin glyph coverage at all (Google
# ships each Noto Sans <Script> font as a script-only subset); this pairs
# with it as an fpdf2 fallback font so English/digit/punctuation runs in a
# source document still render in the exported PDF instead of going blank.
FALLBACK_FONT_PATH = os.path.join(FONTS_DIR, 'NotoSans-Regular.ttf')

DIC_URL = 'https://raw.githubusercontent.com/LibreOffice/dictionaries/master/kn_IN/kn_IN.dic'
AFF_URL = 'https://raw.githubusercontent.com/LibreOffice/dictionaries/master/kn_IN/kn_IN.aff'
FONT_URL = 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansKannada/NotoSansKannada-Regular.ttf'
FALLBACK_FONT_URL = 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf'

# Local fallback paths from neighbor repos
SOURCE_DIC = os.path.join(BASE_DIR, '..', 'Kannada-Autocorrect', 'data', 'kn_IN.dic')
SOURCE_AFF = os.path.join(BASE_DIR, '..', 'Kannada-Autocorrect', 'data', 'kn_IN.aff')


def init_assets():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FONTS_DIR, exist_ok=True)

    print("Initializing Kannada OCR Pipeline Assets...")

    # 1. Dictionary kn_IN.dic
    if os.path.exists(DIC_GZ_PATH):
        print(f"  [✓] kn_IN.dic.gz shipped with the repo "
              f"({os.path.getsize(DIC_GZ_PATH)} bytes, read directly -- nothing to do)")
    elif os.path.exists(DIC_PATH):
        print(f"  [✓] kn_IN.dic already exists ({os.path.getsize(DIC_PATH)} bytes)")
    elif os.path.exists(SOURCE_DIC):
        shutil.copy2(SOURCE_DIC, DIC_PATH)
        print(f"  [✓] Copied kn_IN.dic from local repository")
    else:
        print("  [↓] Downloading kn_IN.dic...")
        try:
            urllib.request.urlretrieve(DIC_URL, DIC_PATH)
            print("  [✓] kn_IN.dic downloaded successfully")
            print("  [!] NOTE: this is the stock LibreOffice kn_IN, 19,645 entries.")
            print("      The pipeline is tuned for the 589,521-entry build normally")
            print("      shipped as data/kn_IN.dic.gz -- expect much worse correction")
            print("      coverage until that file is restored.")
        except Exception as e:
            print(f"  [!] Failed to download kn_IN.dic: {e}")

    # 2. Affix kn_IN.aff
    if os.path.exists(AFF_PATH):
        print(f"  [✓] kn_IN.aff already exists ({os.path.getsize(AFF_PATH)} bytes)")
    elif os.path.exists(SOURCE_AFF):
        shutil.copy2(SOURCE_AFF, AFF_PATH)
        print(f"  [✓] Copied kn_IN.aff from local repository")
    else:
        print("  [↓] Downloading kn_IN.aff...")
        try:
            urllib.request.urlretrieve(AFF_URL, AFF_PATH)
            print("  [✓] kn_IN.aff downloaded successfully")
        except Exception as e:
            print(f"  [!] Failed to download kn_IN.aff: {e}")

    # 3. Noto Sans Kannada Font
    if os.path.exists(FONT_PATH):
        print(f"  [✓] NotoSansKannada-Regular.ttf already exists ({os.path.getsize(FONT_PATH)} bytes)")
    else:
        print("  [↓] Downloading NotoSansKannada-Regular.ttf...")
        try:
            req = urllib.request.Request(
                FONT_URL,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as resp, open(FONT_PATH, 'wb') as out_f:
                shutil.copyfileobj(resp, out_f)
            print("  [✓] NotoSansKannada-Regular.ttf downloaded successfully")
        except Exception as e:
            print(f"  [!] Failed to download NotoSansKannada-Regular.ttf: {e}")

    # 3b. Noto Sans (Latin fallback, paired with NotoSansKannada for PDF export)
    if os.path.exists(FALLBACK_FONT_PATH):
        print(f"  [✓] NotoSans-Regular.ttf already exists ({os.path.getsize(FALLBACK_FONT_PATH)} bytes)")
    else:
        print("  [↓] Downloading NotoSans-Regular.ttf...")
        try:
            req = urllib.request.Request(
                FALLBACK_FONT_URL,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as resp, open(FALLBACK_FONT_PATH, 'wb') as out_f:
                shutil.copyfileobj(resp, out_f)
            print("  [✓] NotoSans-Regular.ttf downloaded successfully")
        except Exception as e:
            print(f"  [!] Failed to download NotoSans-Regular.ttf: {e}")

    # 4. Tesseract Language Models
    TESSDATA_DIR = os.path.join(BASE_DIR, 'tessdata')
    os.makedirs(TESSDATA_DIR, exist_ok=True)
    tess_models = {
        'kan.traineddata': 'https://github.com/tesseract-ocr/tessdata_fast/raw/main/kan.traineddata',
        'hin.traineddata': 'https://github.com/tesseract-ocr/tessdata_fast/raw/main/hin.traineddata',
        'san.traineddata': 'https://github.com/tesseract-ocr/tessdata_fast/raw/main/san.traineddata',
        'eng.traineddata': 'https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata',
        'osd.traineddata': 'https://github.com/tesseract-ocr/tessdata_fast/raw/main/osd.traineddata',
    }
    for model_name, url in tess_models.items():
        m_path = os.path.join(TESSDATA_DIR, model_name)
        if os.path.exists(m_path):
            print(f"  [✓] {model_name} already exists ({os.path.getsize(m_path)} bytes)")
        else:
            print(f"  [↓] Downloading {model_name}...")
            try:
                urllib.request.urlretrieve(url, m_path)
                print(f"  [✓] {model_name} downloaded successfully")
            except Exception as e:
                print(f"  [!] Failed to download {model_name}: {e}")

    # 5. N-gram language model (real-corpus word/bigram frequencies for context
    # ranking). Not auto-downloaded here -- it's built from a large external
    # corpus offline; see tools/build_ngram_model.py. Without it the pipeline
    # falls back to unigram-only counts from the dictionary (still correct,
    # just unable to use surrounding words to disambiguate candidates).
    NGRAM_MODEL_PATH = os.path.join(DATA_DIR, 'ngram_model.pkl.gz')
    if os.path.exists(NGRAM_MODEL_PATH):
        print(f"  [✓] ngram_model.pkl.gz already exists ({os.path.getsize(NGRAM_MODEL_PATH)} bytes)")
    else:
        print("  [i] ngram_model.pkl.gz not found -- context-aware ranking will fall back to "
              "unigram-only dictionary counts. Run tools/build_ngram_model.py to build it from a "
              "real Kannada corpus.")

    print("\nAsset initialization complete!")



if __name__ == '__main__':
    init_assets()
