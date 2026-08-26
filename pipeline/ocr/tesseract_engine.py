"""
Tesseract OCR Engine Wrapper
Handles multi-language Indic OCR extraction, page data analysis, and layout detection.
"""

import os
import shutil
from typing import List, Dict, Any, Optional
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_TESSDATA_DIR = os.path.join(BASE_DIR, 'tessdata')

if os.path.exists(LOCAL_TESSDATA_DIR):
    os.environ['TESSDATA_PREFIX'] = LOCAL_TESSDATA_DIR

# Comprehensive Indic language mappings supported by Tesseract
SUPPORTED_LANGUAGES = {
    'kan': 'Kannada',
    'eng': 'English',
    'hin': 'Hindi',
    'san': 'Sanskrit',
    'tam': 'Tamil',
    'tel': 'Telugu',
    'mal': 'Malayalam',
    'mar': 'Marathi',
    'ben': 'Bengali',
    'guj': 'Gujarati',
    'ori': 'Odia',
    'pan': 'Punjabi',
    'asm': 'Assamese',
    'urd': 'Urdu',
    'sin': 'Sinhala'
}


def is_tesseract_available() -> bool:
    """Check if Tesseract is installed on the system and in PATH."""
    if pytesseract is None:
        return False
    if shutil.which('tesseract') is not None:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    return False


def get_available_languages() -> List[str]:
    """Retrieve list of installed language packs in Tesseract."""
    if not is_tesseract_available():
        return []
    
    langs = set()
    if os.path.exists(LOCAL_TESSDATA_DIR):
        for f in os.listdir(LOCAL_TESSDATA_DIR):
            if f.endswith('.traineddata'):
                langs.add(f.replace('.traineddata', ''))

    try:
        sys_langs = pytesseract.get_languages(config=f'--tessdata-dir "{LOCAL_TESSDATA_DIR}"' if os.path.exists(LOCAL_TESSDATA_DIR) else '')
        langs.update(sys_langs)
    except Exception:
        pass

    return sorted(list(langs)) if langs else ['kan', 'eng']


def _get_tesseract_config(psm: int = 3, oem: int = 3) -> str:
    parts = [f'--oem {oem}', f'--psm {psm}']
    if os.path.exists(LOCAL_TESSDATA_DIR):
        parts.append(f'--tessdata-dir "{LOCAL_TESSDATA_DIR}"')
    return ' '.join(parts)


def ocr_image(
    image: Image.Image,
    lang: str = 'kan+eng',
    psm: int = 3,
    oem: int = 3
) -> str:
    """
    Extract raw text from a PIL Image using Tesseract OCR.
    """
    if not is_tesseract_available():
        raise RuntimeError("Tesseract OCR is not installed or not available in PATH.")

    config = _get_tesseract_config(psm=psm, oem=oem)
    return pytesseract.image_to_string(image, lang=lang, config=config)



def ocr_image_with_layout(
    image: Image.Image,
    lang: str = 'kan+eng',
    page_num: int = 1
) -> List[Dict[str, Any]]:
    """
    Perform OCR and extract structured layout lines with estimated horizontal alignment.
    """
    if not is_tesseract_available():
        raise RuntimeError("Tesseract OCR is not installed or not available in PATH.")

    img_w, img_h = image.size
    config = _get_tesseract_config()
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)

    # Group extracted word boxes by block and line
    lines_map: Dict[tuple, List[Dict[str, Any]]] = {}
    n_boxes = len(data['text'])

    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i]) if str(data['conf'][i]).isdigit() or isinstance(data['conf'][i], (int, float)) else -1

        if not text or conf < 0:
            continue

        block_num = data['block_num'][i]
        line_num = data['line_num'][i]
        key = (block_num, line_num)

        box_info = {
            'text': text,
            'left': data['left'][i],
            'top': data['top'][i],
            'width': data['width'][i],
            'height': data['height'][i],
            'conf': conf
        }

        if key not in lines_map:
            lines_map[key] = []
        lines_map[key].append(box_info)

    layout_lines = []
    for key in sorted(lines_map.keys()):
        boxes = lines_map[key]
        line_text = ' '.join(b['text'] for b in boxes)
        
        min_left = min(b['left'] for b in boxes)
        max_right = max(b['left'] + b['width'] for b in boxes)
        min_top = min(b['top'] for b in boxes)
        max_bottom = max(b['top'] + b['height'] for b in boxes)
        
        line_width = max_right - min_left
        line_center = min_left + (line_width / 2.0)
        page_center = img_w / 2.0

        # Alignment heuristic: Center vs Left vs Right
        if line_width < img_w * 0.7 and abs(line_center - page_center) < (img_w * 0.1):
            alignment = 'C'
        elif min_left > (img_w * 0.55):
            alignment = 'R'
        else:
            alignment = 'L'

        layout_lines.append({
            'text': line_text,
            'alignment': alignment,
            'left': min_left,
            'top': min_top,
            'width': line_width,
            'height': max_bottom - min_top,
            'page_num': page_num
        })

    return layout_lines
