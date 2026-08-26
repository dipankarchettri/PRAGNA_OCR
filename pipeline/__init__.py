"""
Kannada OCR & Autocorrect Pipeline
Unified entry points for document ingestion, optical character recognition,
morphological post-processing, spelling correction, and structured export.
"""

import os
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from .ingestion import (
    is_pdf_file, is_image_file, inspect_pdf,
    extract_searchable_pdf_layout, rasterize_pdf_to_images,
    load_and_preprocess_image
)
from .ocr import is_tesseract_available, ocr_image, ocr_image_with_layout, SUPPORTED_LANGUAGES
from .correction import load_dictionary, train_model, get_word_list, correct_text, correct_layout_lines
from .exporter import (
    generate_pdf_from_text,
    generate_pdf_from_layout,
    export_pages_to_text,
    export_combined_text,
    export_json_report
)

_INITIALIZED = False


def init_pipeline(dic_path: Optional[str] = None):
    """Initialize dictionary and n-gram models once."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    load_dictionary(dic_path)
    train_model(get_word_list())
    _INITIALIZED = True


def process_text_input(text: str) -> Dict[str, Any]:
    """Process raw text input directly through the cleaning and correction engine."""
    init_pipeline()
    start_t = time.time()
    result = correct_text(text)
    result['latency_seconds'] = round(time.time() - start_t, 3)
    return result


def process_document(
    input_path: str,
    lang: str = 'kan+eng',
    dpi: int = 300,
    output_dir: Optional[str] = None,
    save_pdf: bool = True,
    save_images: bool = False,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    End-to-end processing pipeline for a PDF or image file:
    1. Ingest & Inspect
    2. Extract / OCR
    3. Clean & Correct
    4. Export Results
    """
    init_pipeline()
    start_time = time.time()
    input_path = os.path.abspath(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    filename = os.path.basename(input_path)
    stem = Path(input_path).stem

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_path), f"{stem}_processed")
    os.makedirs(output_dir, exist_ok=True)

    pages_result: List[Dict[str, Any]] = []
    all_layout_lines: List[Dict[str, Any]] = []
    all_corrections: List[Dict[str, Any]] = []
    saved_images_paths: List[str] = []

    # ─────────────────────────────────────────────────────────
    # Case 1: PDF Document
    # ─────────────────────────────────────────────────────────
    if is_pdf_file(input_path):
        if progress_callback:
            progress_callback({'stage': 'inspecting', 'message': 'Analyzing PDF document...', 'percent': 5})

        pdf_info = inspect_pdf(input_path)
        is_searchable = pdf_info['is_searchable']
        page_count = pdf_info['page_count']

        if is_searchable:
            if progress_callback:
                progress_callback({'stage': 'extracting', 'message': f'Extracting digital text from {page_count} pages...', 'percent': 30})

            # Digital Searchable PDF -> PyMuPDF Layout Extraction
            layout_lines = extract_searchable_pdf_layout(input_path)
            
            if progress_callback:
                progress_callback({'stage': 'correcting', 'message': 'Applying morphological cleaning and Sandhi rules...', 'percent': 70})

            corrected_lines, corrections = correct_layout_lines(layout_lines)
            all_layout_lines.extend(corrected_lines)
            all_corrections.extend(corrections)

            # Group by page
            pages_map = {}
            for line in layout_lines:
                p = line.get('page_num', 1)
                pages_map.setdefault(p, []).append(line['text'])

            for p in sorted(pages_map.keys()):
                raw_page_text = '\n'.join(pages_map[p])
                corr_res = correct_text(raw_page_text)
                pages_result.append({
                    'page_num': p,
                    'raw_text': raw_page_text,
                    'corrected_text': corr_res['corrected'],
                    'has_errors': corr_res['has_errors'],
                    'corrections': corr_res['corrections']
                })
        else:
            # Scanned PDF -> Rasterize + Tesseract OCR
            if not is_tesseract_available():
                raise RuntimeError("Tesseract OCR is required for scanned PDFs, but is not installed in PATH.")

            if progress_callback:
                progress_callback({'stage': 'rasterizing', 'message': f'Rasterizing {page_count} pages at {dpi} DPI...', 'percent': 15})

            images = rasterize_pdf_to_images(input_path, dpi=dpi)
            total_pages_count = len(images)
            img_dir = os.path.join(output_dir, 'images')
            if save_images:
                os.makedirs(img_dir, exist_ok=True)

            for i, img in enumerate(images):
                page_num = i + 1
                ocr_pct = int(15 + ((i + 1) / total_pages_count) * 70)
                if progress_callback:
                    progress_callback({
                        'stage': 'ocr',
                        'current_page': page_num,
                        'total_pages': total_pages_count,
                        'percent': ocr_pct,
                        'message': f'OCR Processing Page {page_num} of {total_pages_count}...'
                    })

                if save_images:
                    img_path = os.path.join(img_dir, f"page_{page_num:03d}.png")
                    img.save(img_path, 'PNG')
                    saved_images_paths.append(img_path)

                # OCR with layout
                lines = ocr_image_with_layout(img, lang=lang, page_num=page_num)
                corrected_lines, page_corrections = correct_layout_lines(lines)
                all_layout_lines.extend(corrected_lines)
                all_corrections.extend(page_corrections)

                raw_page_text = '\n'.join(l['text'] for l in lines)
                corr_page_text = '\n'.join(l['text'] for l in corrected_lines)

                pages_result.append({
                    'page_num': page_num,
                    'raw_text': raw_page_text,
                    'corrected_text': corr_page_text,
                    'has_errors': len(page_corrections) > 0,
                    'corrections': page_corrections
                })


    # ─────────────────────────────────────────────────────────
    # Case 2: Image Document (PNG, JPG, TIFF, WEBP, etc.)
    # ─────────────────────────────────────────────────────────
    elif is_image_file(input_path):
        if not is_tesseract_available():
            raise RuntimeError("Tesseract OCR is required for image files, but is not installed in PATH.")

        img = load_and_preprocess_image(input_path, enhance_contrast=True)
        if save_images:
            img_dir = os.path.join(output_dir, 'images')
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"{stem}.png")
            img.save(img_path, 'PNG')
            saved_images_paths.append(img_path)

        lines = ocr_image_with_layout(img, lang=lang, page_num=1)
        corrected_lines, page_corrections = correct_layout_lines(lines)
        all_layout_lines.extend(corrected_lines)
        all_corrections.extend(page_corrections)

        raw_page_text = '\n'.join(l['text'] for l in lines)
        corr_page_text = '\n'.join(l['text'] for l in corrected_lines)

        pages_result.append({
            'page_num': 1,
            'raw_text': raw_page_text,
            'corrected_text': corr_page_text,
            'has_errors': len(page_corrections) > 0,
            'corrections': page_corrections
        })

    else:
        raise ValueError(f"Unsupported file format: {filename}")

    # ─────────────────────────────────────────────────────────
    # Export Outputs
    # ─────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback({'stage': 'exporting', 'message': 'Generating layout-preserved PDF and reports...', 'percent': 92})

    full_raw_text = '\n\n'.join(p['raw_text'] for p in pages_result)
    full_corrected_text = '\n\n'.join(p['corrected_text'] for p in pages_result)

    # 1. Page text files
    txt_files = export_pages_to_text(pages_result, output_dir)

    # 2. Combined text
    combined_txt_path = os.path.join(output_dir, f"{stem}_corrected.txt")
    export_combined_text(full_corrected_text, combined_txt_path)

    # 3. PDF Generation (layout-preserving)
    pdf_out_path = None
    if save_pdf:
        pdf_out_path = os.path.join(output_dir, f"{stem}_corrected.pdf")
        if all_layout_lines:
            generate_pdf_from_layout(all_layout_lines, pdf_out_path)
        else:
            generate_pdf_from_text(full_corrected_text, pdf_out_path)

    # 4. JSON Report
    total_time = round(time.time() - start_time, 2)
    report_data = {
        'source_file': filename,
        'source_path': input_path,
        'processing_time_seconds': total_time,
        'total_pages': len(pages_result),
        'total_corrections': len(all_corrections),
        'language': lang,
        'output_directory': output_dir,
        'generated_files': {
            'text_files': txt_files,
            'combined_text': combined_txt_path,
            'pdf': pdf_out_path,
            'images': saved_images_paths
        },
        'corrections_summary': all_corrections,
        'pages': pages_result
    }

    json_report_path = os.path.join(output_dir, f"{stem}_report.json")
    export_json_report(report_data, json_report_path)

    if progress_callback:
        progress_callback({'stage': 'complete', 'message': 'Document processing complete!', 'percent': 100})


    return {
        'success': True,
        'report': report_data,
        'raw_text': full_raw_text,
        'corrected_text': full_corrected_text,
        'total_corrections': len(all_corrections),
        'total_pages': len(pages_result),
        'pdf_path': pdf_out_path,
        'json_path': json_report_path,
        'combined_txt_path': combined_txt_path,
        'output_dir': output_dir,
        'latency_seconds': total_time
    }
