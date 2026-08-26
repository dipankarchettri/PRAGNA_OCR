"""
Layout-Preserving Kannada PDF Generator
Generates clean, styled PDF documents with embedded Noto Sans Kannada Unicode font.
"""

import os
from typing import List, Dict, Any, Optional
from fpdf import FPDF, Align

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FONT_PATH = os.path.join(BASE_DIR, 'web', 'static', 'fonts', 'NotoSansKannada-Regular.ttf')

_ALIGN_MAP = {'L': Align.L, 'C': Align.C, 'R': Align.R}


class _KannadaPDFDocument(FPDF):
    def header(self):
        try:
            self.set_font('NotoKannada', size=9)
            self.set_text_color(130, 115, 100)
            self.set_x(self.l_margin)
            self.cell(self.epw, 7, 'ಕನ್ನಡ ಸ್ವಯಂ ತಿದ್ದುಪಡಿ ದಾಖಲೆ — ತಿದ್ದಿದ ಪ್ರತಿ', align='C', new_x="LMARGIN", new_y="NEXT")
            self.ln(1)
            self.set_draw_color(220, 210, 195)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)
            self.set_x(self.l_margin)
        except Exception:
            pass

    def footer(self):
        try:
            self.set_y(-12)
            self.set_x(self.l_margin)
            self.set_font('NotoKannada', size=8)
            self.set_text_color(160, 150, 135)
            self.cell(self.epw, 8, f'ಪುಟ {self.page_no()}', align='C')
        except Exception:
            pass


def _ensure_kannada_font(pdf: _KannadaPDFDocument, font_path: Optional[str] = None):
    font_file = font_path or DEFAULT_FONT_PATH
    if not os.path.exists(font_file):
        raise FileNotFoundError(
            f"Noto Sans Kannada font not found at '{font_file}'. Run 'python setup.py' to download it."
        )
    pdf.add_font('NotoKannada', fname=font_file)


def generate_pdf_from_text(
    text: str,
    output_path: str,
    title: str = 'Corrected Document',
    font_path: Optional[str] = None
) -> str:
    """
    Generate a simple formatted PDF from a block of corrected Kannada text.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf = _KannadaPDFDocument()
    _ensure_kannada_font(pdf, font_path)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font('NotoKannada', size=11)
    pdf.set_text_color(35, 30, 25)

    for line in text.splitlines():
        trimmed = line.strip()
        pdf.set_x(pdf.l_margin)
        if not trimmed:
            pdf.ln(4)
        else:
            pdf.multi_cell(pdf.epw, 7, trimmed, align='L', new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_path)
    return output_path


def generate_pdf_from_layout(
    layout_lines: List[Dict[str, Any]],
    output_path: str,
    font_path: Optional[str] = None
) -> str:
    """
    Generate a layout-preserving PDF respecting extracted line alignments (Left, Center, Right).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf = _KannadaPDFDocument()
    _ensure_kannada_font(pdf, font_path)
    pdf.set_auto_page_break(auto=True, margin=15)
    
    current_page = None
    for line in layout_lines:
        line_page = line.get('page_num', 1)
        if current_page is None or line_page != current_page:
            pdf.add_page()
            current_page = line_page
            pdf.set_font('NotoKannada', size=11)
            pdf.set_text_color(35, 30, 25)

        text = line.get('text', '').strip()
        align_code = line.get('alignment', 'L')
        fpdf_align = _ALIGN_MAP.get(align_code, Align.L)

        pdf.set_x(pdf.l_margin)
        if not text:
            pdf.ln(4)
            continue

        pdf.multi_cell(pdf.epw, 7, text, align=fpdf_align, new_x="LMARGIN", new_y="NEXT")

    if current_page is None:
        pdf.add_page()

    pdf.output(output_path)
    return output_path

