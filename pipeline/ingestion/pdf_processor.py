"""
PDF Processor & Ingestion
Handles multi-page PDF ingestion:
- Fast text extraction via PyMuPDF (fitz) for searchable PDFs
- High-resolution page rasterization via pdf2image for scanned PDFs
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:
    fitz = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

MAX_PDF_PAGES = 500


class PDFPageLimitExceeded(Exception):
    pass


class PopplerMissingError(Exception):
    pass


def is_pdf_file(filepath: str) -> bool:
    return filepath.lower().endswith('.pdf')


def inspect_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Inspect PDF to determine page count, metadata, and whether it contains searchable digital text.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed. Install with: pip install pymupdf")

    doc = fitz.open(pdf_path)
    page_count = len(doc)

    if page_count > MAX_PDF_PAGES:
        raise PDFPageLimitExceeded(f"PDF exceeds the maximum supported page limit ({MAX_PDF_PAGES} pages). Found: {page_count}")

    total_chars = 0
    kannada_chars = 0

    for page in doc:
        txt = page.get_text() or ''
        total_chars += len(txt.strip())
        # Check for Kannada unicode chars (U+0C80 - U+0CFF)
        kannada_chars += sum(1 for c in txt if '\u0C80' <= c <= '\u0CFF')

    is_searchable = (total_chars > 50 and (kannada_chars / (total_chars or 1)) > 0.05) or (total_chars > 200)

    return {
        'page_count': page_count,
        'total_chars': total_chars,
        'kannada_chars': kannada_chars,
        'is_searchable': is_searchable,
        'metadata': doc.metadata
    }


def extract_searchable_pdf_layout(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract lines and layout alignments from a searchable digital PDF using PyMuPDF.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for digital PDF extraction.")

    doc = fitz.open(pdf_path)
    layout_lines = []

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_rect = page.rect
        page_width = page_rect.width

        # Extract structured text blocks
        blocks = page.get_text("blocks")
        for b in blocks:
            # b: (x0, y0, x1, y1, text, block_no, block_type)
            if len(b) >= 5 and b[6] == 0:  # text block
                text_content = b[4].strip()
                if not text_content:
                    continue

                for line in text_content.splitlines():
                    line_clean = line.strip()
                    if not line_clean:
                        continue

                    # Determine alignment
                    x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
                    block_width = x1 - x0
                    block_center = x0 + (block_width / 2.0)
                    page_center = page_width / 2.0

                    if block_width < page_width * 0.6 and abs(block_center - page_center) < (page_width * 0.1):
                        alignment = 'C'
                    elif x0 > (page_width * 0.55):
                        alignment = 'R'
                    else:
                        alignment = 'L'

                    layout_lines.append({
                        'text': line_clean,
                        'alignment': alignment,
                        'top': y0,
                        'left': x0,
                        'width': block_width,
                        'height': y1 - y0,
                        'page_num': page_num
                    })

    return layout_lines


def rasterize_pdf_to_images(
    pdf_path: str,
    dpi: int = 300,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None
) -> List[Image.Image]:
    """
    Rasterize PDF pages into PIL Images using pdf2image (backed by poppler).
    """
    if convert_from_path is None:
        raise RuntimeError("pdf2image is not installed. Install with: pip install pdf2image")

    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first_page,
            last_page=last_page
        )
        return images
    except Exception as e:
        if 'poppler' in str(e).lower():
            raise PopplerMissingError(
                "Poppler utilities not found. On macOS, install with: brew install poppler"
            ) from e
        raise
