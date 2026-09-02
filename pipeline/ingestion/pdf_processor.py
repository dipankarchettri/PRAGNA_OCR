"""
PDF Processor & Ingestion
Handles multi-page PDF ingestion:
- Fast text extraction via PyMuPDF (fitz) for searchable PDFs
- High-resolution page rasterization via pdf2image for scanned PDFs
"""

import os
import re
import string
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw

try:
    import pymupdf as fitz
except ImportError:
    fitz = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

MAX_PDF_PAGES = 500

# Unicode blocks for the Indic + Urdu scripts this pipeline claims to support.
# Used to tell real script text apart from legacy-font mojibake, which decodes
# to plain Latin/ASCII codepoints (glyphs remapped onto a-z) even though it
# renders as Kannada/Devanagari/etc. on screen.
_INDIC_SCRIPT_RANGES = [
    (0x0600, 0x06FF),  # Arabic (Urdu)
    (0x0900, 0x097F),  # Devanagari (Hindi, Sanskrit, Marathi)
    (0x0980, 0x09FF),  # Bengali (also Assamese)
    (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0D80, 0x0DFF),  # Sinhala
]

# Common English function words. Legacy-font mojibake ends up as Latin
# lookalike gibberish (e.g. "doart\"gd", "eOodn$") that essentially never
# reproduces these specific short words, so their presence is a cheap,
# reliable signal that Latin-script extraction is real text, not garbage.
_COMMON_ENGLISH_WORDS = {
    'the', 'of', 'and', 'to', 'in', 'is', 'for', 'on', 'that', 'this',
    'with', 'as', 'by', 'at', 'from', 'be', 'are', 'was', 'or', 'an',
}


def _is_indic_script_char(c: str) -> bool:
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in _INDIC_SCRIPT_RANGES)


def _looks_like_real_english(txt: str) -> bool:
    # Match whole whitespace-delimited tokens, not bare [A-Za-z']+ runs --
    # legacy-font mojibake mixes plain ASCII letters (e.g. vowel-sign
    # fragments) in between Latin-1-supplement glyphs *within* what was one
    # source word (e.g. "ªÀiÁ£À«Ã"), and scanning for ASCII substrings alone
    # pulls out short fragments like "i" or "s" that coincidentally collide
    # with 2-letter stopwords ("is", "an", "or"...) across a long garbled
    # page. Requiring the entire token to be ASCII letters filters that out:
    # a mixed-script token never fully matches, so it can never be counted.
    tokens = re.split(r"\s+", txt.strip())
    words = [w for t in tokens if (w := t.strip(string.punctuation).lower()) and w.isalpha() and w.isascii()]
    if len(words) < 10:
        return False
    hits = sum(1 for w in words if w in _COMMON_ENGLISH_WORDS)
    return hits >= 5


def _has_cmap_corruption(txt: str, threshold: float = 0.01) -> bool:
    """Whether text contains enough C1 control characters (U+0080-U+009F)
    to indicate a broken PDF ToUnicode CMap rather than real content.

    Real extracted text -- Kannada, Latin, or otherwise -- never
    legitimately contains these codepoints; PDF viewers/producers use them
    only as an artifact of an embedded font whose CMap maps some glyph IDs
    (often just the matras/conjuncts absent from the font's expected set)
    to the wrong Unicode range. Unlike a fully legacy-encoded font (caught
    by the low Kannada/indic density check below), this corruption can
    coexist with mostly-correct Kannada text in the same block, so it has
    to be checked independently rather than folding it into the density
    ratios.
    """
    total = len(txt)
    if total == 0:
        return False
    control = sum(1 for c in txt if 0x80 <= ord(c) <= 0x9F)
    return (control / total) > threshold


def _page_text_is_valid(txt: str) -> bool:
    """Whether one page's extracted 'digital text' looks trustworthy rather
    than mojibake from a legacy non-Unicode font or a corrupted font CMap."""
    total = len(txt.strip())
    if total == 0:
        return False
    if _has_cmap_corruption(txt):
        return False
    kannada = sum(1 for c in txt if 'ಀ' <= c <= '೿')
    indic = sum(1 for c in txt if _is_indic_script_char(c))
    if total > 30 and (kannada / total) > 0.05:
        return True
    if total > 30 and (indic / total) > 0.05:
        return True
    if total > 80 and _looks_like_real_english(txt):
        return True
    return False


class PDFPageLimitExceeded(Exception):
    pass


class PopplerMissingError(Exception):
    pass


def is_pdf_file(filepath: str) -> bool:
    return filepath.lower().endswith('.pdf')


def _page_blocks_from_page(page) -> List[Dict[str, Any]]:
    """Text blocks for one already-open fitz page, each tagged with its own
    validity. A page can mix a block with a trustworthy digital text layer
    (e.g. an English abstract) and a block that's mojibake from a legacy
    font (e.g. the Kannada body text below it) -- validity has to be judged
    per block, not for the page's text as a whole."""
    out = []
    for b in page.get_text("blocks"):
        # b: (x0, y0, x1, y1, text, block_no, block_type)
        if len(b) >= 5 and b[6] == 0:  # text block
            text_content = b[4].strip()
            if not text_content:
                continue
            out.append({
                'text': text_content,
                'x0': b[0], 'y0': b[1], 'x1': b[2], 'y1': b[3],
                'is_valid': _page_text_is_valid(text_content),
            })
    return out


def get_page_blocks(pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
    """Text blocks (with bbox + validity) for one 1-indexed page of a PDF."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for digital PDF extraction.")
    doc = fitz.open(pdf_path)
    return _page_blocks_from_page(doc[page_num - 1])


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
    indic_chars = 0
    # Per page: 'searchable' (every block trustworthy -> fast digital
    # extraction), 'scanned' (no text, or none of it trustworthy -> full
    # page OCR), or 'mixed' (some blocks trustworthy, some are mojibake from
    # a legacy non-Unicode font -> OCR just the untrustworthy regions and
    # keep the good digital text for the rest of the page).
    page_classifications = []

    for page in doc:
        txt = page.get_text() or ''
        total_chars += len(txt.strip())
        # Check for Kannada unicode chars (U+0C80 - U+0CFF)
        kannada_chars += sum(1 for c in txt if '\u0C80' <= c <= '\u0CFF')
        indic_chars += sum(1 for c in txt if _is_indic_script_char(c))

        blocks = _page_blocks_from_page(page)
        if not blocks:
            page_classifications.append('scanned')
        elif all(b['is_valid'] for b in blocks):
            page_classifications.append('searchable')
        elif any(b['is_valid'] for b in blocks):
            page_classifications.append('mixed')
        else:
            page_classifications.append('scanned')

    # Whole-document flag, kept for backward compatibility with callers that
    # only care about a single yes/no. process_document makes the real
    # extract-vs-OCR decision per page via page_classifications below.
    is_searchable = page_count > 0 and all(c == 'searchable' for c in page_classifications)

    return {
        'page_count': page_count,
        'total_chars': total_chars,
        'kannada_chars': kannada_chars,
        'indic_chars': indic_chars,
        'is_searchable': is_searchable,
        'page_classifications': page_classifications,
        'metadata': doc.metadata
    }


def extract_searchable_pdf_layout(pdf_path: str, pages: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Extract lines and layout alignments from the trustworthy text blocks of
    a digital PDF using PyMuPDF. Blocks that look like legacy-font mojibake
    (see _page_text_is_valid) are skipped -- callers relying on OCR to fill
    those in should pair this with rasterize_page_masking_valid_text.

    `pages`, if given, restricts extraction to that list of 1-indexed page
    numbers (used when only some pages of the document have a trustworthy
    digital text layer).
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for digital PDF extraction.")

    doc = fitz.open(pdf_path)
    layout_lines = []

    try:
        # Visit only the requested pages. This used to walk every page of the
        # document and `continue` past the unwanted ones, while process_document
        # calls it once per mixed page -- quadratic in page count on exactly the
        # documents that are already the slowest.
        if pages is not None:
            page_numbers = [p for p in sorted(set(pages)) if 1 <= p <= doc.page_count]
        else:
            page_numbers = range(1, doc.page_count + 1)

        for page_num in page_numbers:
            page = doc.load_page(page_num - 1)
            page_width = page.rect.width
            page_center = page_width / 2.0

            # "dict" rather than "blocks" because it carries a bbox for every
            # individual line. The old code split a block's text on newlines and
            # stamped EVERY resulting line with the enclosing block's bbox, so
            # all lines of a paragraph reported an identical top, left, width and
            # height. That makes per-line geometry meaningless: the reading-order
            # sort had nothing to order lines within a block by, and anything
            # asking "did this line stop short of the margin?" -- which is how
            # the reflow exporter finds paragraph ends -- always got "no".
            page_dict = page.get_text("dict")

            for block in page_dict.get('blocks', []):
                if block.get('type') != 0:      # 0 = text, 1 = image
                    continue
                block_lines = block.get('lines', [])
                if not block_lines:
                    continue

                def _line_text(ln):
                    return ''.join(span.get('text', '') for span in ln.get('spans', []))

                # Validity stays a per-block judgement: a page can mix a
                # trustworthy block with a mojibake one, and a single line is
                # too little text to classify reliably.
                block_text = '\n'.join(_line_text(ln) for ln in block_lines).strip()
                if not block_text or not _page_text_is_valid(block_text):
                    continue

                # Alignment likewise stays a block property -- a centred heading
                # is a centred *block*, whereas the last line of a left-aligned
                # paragraph is short without being centred.
                bx0, _by0, bx1, _by1 = block['bbox']
                block_width = bx1 - bx0
                block_center = bx0 + (block_width / 2.0)
                if block_width < page_width * 0.6 and abs(block_center - page_center) < (page_width * 0.1):
                    alignment = 'C'
                elif bx0 > (page_width * 0.55):
                    alignment = 'R'
                else:
                    alignment = 'L'

                for ln in block_lines:
                    text = _line_text(ln).strip()
                    if not text:
                        continue
                    lx0, ly0, lx1, ly1 = ln['bbox']
                    layout_lines.append({
                        'text': text,
                        'alignment': alignment,
                        'top': ly0,
                        'left': lx0,
                        'width': lx1 - lx0,
                        'height': ly1 - ly0,
                        'page_num': page_num
                    })
    finally:
        doc.close()

    return layout_lines


def rasterize_page_masking_valid_text(
    pdf_path: str,
    page_num: int,
    dpi: int,
    blocks: List[Dict[str, Any]]
) -> Image.Image:
    """
    Rasterize one page and paint white over any block already covered by a
    trustworthy digital text layer, so OCR only has to read the regions the
    text layer can't be trusted for (used for 'mixed' pages).
    """
    img = rasterize_pdf_to_images(pdf_path, dpi=dpi, first_page=page_num, last_page=page_num)[0].convert('RGB')
    scale = dpi / 72.0
    pad = max(2, int(0.02 * dpi))  # small margin so masking doesn't clip glyphs right at a block edge
    draw = ImageDraw.Draw(img)
    for b in blocks:
        if not b['is_valid']:
            continue
        x0 = max(0, int(b['x0'] * scale) - pad)
        y0 = max(0, int(b['y0'] * scale) - pad)
        x1 = min(img.width, int(b['x1'] * scale) + pad)
        y1 = min(img.height, int(b['y1'] * scale) + pad)
        draw.rectangle([x0, y0, x1, y1], fill='white')
    return img


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
