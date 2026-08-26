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
AFF_PATH = os.path.join(DATA_DIR, 'kn_IN.aff')
FONT_PATH = os.path.join(FONTS_DIR, 'NotoSansKannada-Regular.ttf')

DIC_URL = 'https://raw.githubusercontent.com/LibreOffice/dictionaries/master/kn_IN/kn_IN.dic'
AFF_URL = 'https://raw.githubusercontent.com/LibreOffice/dictionaries/master/kn_IN/kn_IN.aff'
FONT_URL = 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansKannada/NotoSansKannada-Regular.ttf'

# Local fallback paths from neighbor repos
SOURCE_DIC = os.path.join(BASE_DIR, '..', 'Kannada-Autocorrect', 'data', 'kn_IN.dic')
SOURCE_AFF = os.path.join(BASE_DIR, '..', 'Kannada-Autocorrect', 'data', 'kn_IN.aff')


def init_assets():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FONTS_DIR, exist_ok=True)

    print("Initializing Kannada OCR Pipeline Assets...")

    # 1. Dictionary kn_IN.dic
    if os.path.exists(DIC_PATH):
        print(f"  [✓] kn_IN.dic already exists ({os.path.getsize(DIC_PATH)} bytes)")
    elif os.path.exists(SOURCE_DIC):
        shutil.copy2(SOURCE_DIC, DIC_PATH)
        print(f"  [✓] Copied kn_IN.dic from local repository")
    else:
        print("  [↓] Downloading kn_IN.dic...")
        try:
            urllib.request.urlretrieve(DIC_URL, DIC_PATH)
            print("  [✓] kn_IN.dic downloaded successfully")
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

    print("\nAsset initialization complete!")



if __name__ == '__main__':
    init_assets()
