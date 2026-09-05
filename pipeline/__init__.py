"""
Kannada OCR & Autocorrect Pipeline
Unified entry points for document ingestion, optical character recognition,
morphological post-processing, spelling correction, and structured export.
"""

import os
import threading
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from .ingestion import (
    is_pdf_file, is_image_file, inspect_pdf, get_page_blocks,
    extract_searchable_pdf_layout, rasterize_pdf_to_images,
    rasterize_page_masking_valid_text,
    load_and_preprocess_image, normalize_resolution, preprocess_for_ocr
)
from .ocr import (
    is_tesseract_available, ocr_image, ocr_image_with_layout,
    is_surya_available, surya_ocr_images_with_layout,
    SUPPORTED_LANGUAGES, DEFAULT_PSM, DEFAULT_OEM
)
from .correction import (
    load_dictionary, train_from_word_list, load_ngram_model, add_vocabulary,
    get_word_list, correct_text, correct_layout_lines, clear_correction_caches
)
from .exporter import (
    generate_pdf_from_text,
    generate_pdf_from_layout,
    export_pages_to_text,
    export_combined_text,
    export_line_provenance,
    export_json_report,
    reflow_lines,
    corpus_stats
)

_INITIALIZED = False
# init_pipeline is reentered from more than one thread: the web app warms the
# dictionary and n-gram model on a background thread while Flask still serves
# requests. The _INITIALIZED flag alone would let two threads both start
# loading the dictionary and training the n-gram model.
_INIT_LOCK = threading.Lock()


def _mean_ocr_confidence(layout_lines: List[Dict[str, Any]]) -> float:
    confs = [c for l in layout_lines for _, c in l.get('word_confidences', [])]
    return sum(confs) / len(confs) if confs else 0.0


def _ocr_with_adaptive_contrast(
    img,
    lang: str,
    page_num: int,
    psm: int,
    oem: int,
    min_confidence: int
) -> List[Dict[str, Any]]:
    """
    OCR `img` (already deskewed, contrast untouched) as-is, then again with a
    +20% contrast boost, and keep whichever run Tesseract itself reports
    higher mean word confidence for -- ties go to the unboosted run.

    Replaces the earlier blanket enhance_contrast=True default, which helped
    genuinely faded scans but could catastrophically corrupt an
    already-good-contrast page (see preprocess_for_ocr's docstring). A
    static image-statistics heuristic (grayscale std-dev/percentile spread)
    was tried first and found unable to separate the two cases. Tesseract's
    own confidence, compared directly between the two runs rather than
    thresholded in isolation, does separate them -- validated against 7 real
    ground-truth pages: it captures genuine wins (lipi CER 16.68%->8.80%,
    akaaradi 5.32%->4.96%), stays neutral on near-ties, and correctly
    discards the boosted run on the catastrophic-corruption cases (sslc
    pages: confidence drops 25-35 points when boosted, so the unboosted run
    is kept, avoiding a ~3-7% CER page turning into ~75-81% word-salad).

    Costs a second Tesseract pass on every page -- a bounded, known cost
    traded for a real per-page accuracy floor, per this project's stated
    priority of correctness over throughput.
    """
    base_lines = ocr_image_with_layout(
        img, lang=lang, page_num=page_num, psm=psm, oem=oem, min_confidence=min_confidence
    )
    boosted_img = preprocess_for_ocr(img, enhance_contrast=True, deskew=False)
    boosted_lines = ocr_image_with_layout(
        boosted_img, lang=lang, page_num=page_num, psm=psm, oem=oem, min_confidence=min_confidence
    )

    if _mean_ocr_confidence(boosted_lines) > _mean_ocr_confidence(base_lines):
        return boosted_lines
    return base_lines


def init_pipeline(dic_path: Optional[str] = None):
    """Initialize dictionary and n-gram models once, across all threads."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    with _INIT_LOCK:
        if _INITIALIZED:
            return

        load_dictionary(dic_path)

        # Prefer a real-corpus-trained model (built via tools/build_ngram_model.py)
        # if one has been cached; fall back to unigram-only dictionary counts so
        # the pipeline still works with no extra setup.
        if load_ngram_model():
            add_vocabulary(get_word_list())
        else:
            train_from_word_list(get_word_list())

        # The corrector memoizes candidate generation against whatever
        # vocabulary is loaded; drop anything cached under a previous one.
        clear_correction_caches()

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
    lang: str = 'kan',
    dpi: int = 400,
    output_dir: Optional[str] = None,
    save_pdf: bool = True,
    save_images: bool = False,
    progress_callback: Optional[callable] = None,
    psm: int = DEFAULT_PSM,
    oem: int = DEFAULT_OEM,
    min_confidence: int = 0,
    adaptive_contrast: bool = True,
    engine: str = 'tesseract'
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
        page_count = pdf_info['page_count']
        page_classifications = pdf_info['page_classifications']

        # Decide per page, not per document -- and for a page that mixes a
        # trustworthy block with a mojibake one (legacy non-Unicode font,
        # see inspect_pdf / _page_text_is_valid), decide per block within
        # that page too. Otherwise a page-level decision would either OCR
        # text that's already fine, or keep garbage text next to it.
        fully_digital_pages = [i + 1 for i, c in enumerate(page_classifications) if c == 'searchable']
        needs_ocr_pages = [i + 1 for i, c in enumerate(page_classifications) if c != 'searchable']

        pages_by_num: Dict[int, Dict[str, Any]] = {}

        if fully_digital_pages:
            if progress_callback:
                progress_callback({'stage': 'extracting', 'message': f'Extracting digital text from {len(fully_digital_pages)} page(s)...', 'percent': 20})

            # Fully searchable pages -> PyMuPDF Layout Extraction
            layout_lines = extract_searchable_pdf_layout(input_path, pages=fully_digital_pages)

            if progress_callback:
                progress_callback({'stage': 'correcting', 'message': 'Applying morphological cleaning and Sandhi rules...', 'percent': 40})

            # Correct once, per page, and derive both outputs from that.
            #
            # This used to run the corrector TWICE over identical content:
            # correct_layout_lines for the PDF, then correct_text on the same
            # text re-joined into a page, for the .txt. The two could disagree,
            # because the n-gram model sees different neighbours line-by-line
            # than it does across a joined page -- so the exported PDF and the
            # exported text could carry different corrections of the same word.
            # It also made the digital path structurally unlike the OCR path
            # below, which has always derived its page text from the corrected
            # lines.
            lines_by_page: Dict[int, List[Dict[str, Any]]] = {}
            for line in layout_lines:
                lines_by_page.setdefault(line.get('page_num', 1), []).append(line)

            for p in fully_digital_pages:
                page_lines = lines_by_page.get(p, [])
                corrected_lines, page_corrections = correct_layout_lines(page_lines)
                all_layout_lines.extend(corrected_lines)
                all_corrections.extend(page_corrections)

                pages_by_num[p] = {
                    'page_num': p,
                    'raw_text': '\n'.join(l['text'] for l in page_lines),
                    # is_likely_non_text is always False here (these lines were
                    # never OCR'd from pixels, so they carry no confidence), but
                    # filtering uniformly keeps both paths identical in shape.
                    'corrected_text': '\n'.join(
                        l['text'] for l in corrected_lines if not l.get('is_likely_non_text')
                    ),
                    'has_errors': len(page_corrections) > 0,
                    'corrections': page_corrections
                }

        if needs_ocr_pages:
            # Remaining pages -> Rasterize + Tesseract OCR. A 'scanned' page
            # has no trustworthy blocks at all, so this is a full-page OCR
            # exactly as before. A 'mixed' page has some -- those regions
            # are painted over before OCR runs (rasterize_page_masking_valid_text)
            # so Tesseract only reads what the text layer can't be trusted
            # for, and the untouched digital text is merged back in by
            # vertical position afterwards.
            if not is_tesseract_available():
                raise RuntimeError("Tesseract OCR is required for scanned pages, but is not installed in PATH.")

            img_dir = os.path.join(output_dir, 'images')
            if save_images:
                os.makedirs(img_dir, exist_ok=True)

            for i, page_num in enumerate(needs_ocr_pages):
                ocr_pct = int(45 + ((i + 1) / len(needs_ocr_pages)) * 45)
                if progress_callback:
                    progress_callback({
                        'stage': 'ocr',
                        'current_page': page_num,
                        'total_pages': page_count,
                        'percent': ocr_pct,
                        'message': f'OCR Processing Page {page_num} of {page_count}...'
                    })

                blocks = get_page_blocks(input_path, page_num)
                img = rasterize_page_masking_valid_text(input_path, page_num, dpi, blocks)

                if save_images:
                    img_path = os.path.join(img_dir, f"page_{page_num:03d}.png")
                    img.save(img_path, 'PNG')
                    saved_images_paths.append(img_path)

                # OCR with layout. normalize_resolution is a no-op for pages already
                # rasterized above the target, but protects against low --dpi settings
                # (it may upscale img further, which is why the OCR->points scale
                # below is derived from the actual image fed to OCR, not just dpi).
                ocr_img = normalize_resolution(img)
                ocr_img = preprocess_for_ocr(ocr_img)
                if adaptive_contrast:
                    ocr_lines = _ocr_with_adaptive_contrast(
                        ocr_img, lang, page_num, psm, oem, min_confidence
                    )
                else:
                    ocr_lines = ocr_image_with_layout(
                        ocr_img,
                        lang=lang,
                        page_num=page_num,
                        psm=psm,
                        oem=oem,
                        min_confidence=min_confidence
                    )
                # OCR line coords are pixels in ocr_img; digital-extraction
                # coords are PDF points (72 dpi). Convert to points so a
                # mixed page's two sources can be merged into one reading
                # order by vertical position.
                effective_dpi = dpi * (ocr_img.width / img.width)
                scale = effective_dpi / 72.0
                for l in ocr_lines:
                    l['top'] /= scale
                    l['left'] /= scale
                    l['width'] /= scale
                    l['height'] /= scale

                digital_lines = extract_searchable_pdf_layout(input_path, pages=[page_num]) if any(b['is_valid'] for b in blocks) else []
                lines = digital_lines + ocr_lines
                lines.sort(key=lambda l: (l['top'], l['left']))

                corrected_lines, page_corrections = correct_layout_lines(lines)
                all_layout_lines.extend(corrected_lines)
                all_corrections.extend(page_corrections)

                raw_page_text = '\n'.join(l['text'] for l in lines)
                # Excludes lines flagged non-text (see NON_TEXT_LINE_CONFIDENCE)
                # from the training-corpus text -- raw_page_text and the PDF
                # export (all_layout_lines, unfiltered) still keep the full
                # page for audit/visual fidelity.
                corr_page_text = '\n'.join(
                    l['text'] for l in corrected_lines if not l.get('is_likely_non_text')
                )

                pages_by_num[page_num] = {
                    'page_num': page_num,
                    'raw_text': raw_page_text,
                    'corrected_text': corr_page_text,
                    'has_errors': len(page_corrections) > 0,
                    'corrections': page_corrections
                }

        pages_result.extend(pages_by_num[p] for p in sorted(pages_by_num.keys()))
        # generate_pdf_from_layout starts a new PDF page whenever page_num
        # changes between consecutive lines, so the digital-extraction batch
        # and the OCR batch (appended separately above) must be reordered
        # into overall page order before export.
        all_layout_lines.sort(key=lambda l: l.get('page_num', 1))


    # ─────────────────────────────────────────────────────────
    # Case 2: Image Document (PNG, JPG, TIFF, WEBP, etc.)
    # ─────────────────────────────────────────────────────────
    elif is_image_file(input_path):
        if not is_tesseract_available():
            raise RuntimeError("Tesseract OCR is required for image files, but is not installed in PATH.")

        # enhance_contrast=False: see preprocess_for_ocr's docstring -- a
        # blanket contrast boost can catastrophically break OCR on an
        # already-good-contrast source, confirmed on a real document.
        img = load_and_preprocess_image(input_path, enhance_contrast=False)
        if save_images:
            img_dir = os.path.join(output_dir, 'images')
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"{stem}.png")
            img.save(img_path, 'PNG')
            saved_images_paths.append(img_path)

        if engine == 'surya':
            # Surya has no language flag, no psm/oem, and its own line
            # detection, so none of the Tesseract tuning applies. The
            # adaptive-contrast A/B is skipped too: it picks a run by comparing
            # Tesseract's mean word confidence between them, and Surya scores
            # whole lines on a different scale, so the comparison is not
            # meaningful and would double an already GPU-bound pass.
            lines = surya_ocr_images_with_layout([img], page_nums=[1])[0]
        elif adaptive_contrast:
            lines = _ocr_with_adaptive_contrast(img, lang, 1, psm, oem, min_confidence)
        else:
            lines = ocr_image_with_layout(
                img, lang=lang, page_num=1,
                psm=psm, oem=oem, min_confidence=min_confidence
            )
        corrected_lines, page_corrections = correct_layout_lines(lines)
        all_layout_lines.extend(corrected_lines)
        all_corrections.extend(page_corrections)

        raw_page_text = '\n'.join(l['text'] for l in lines)
        corr_page_text = '\n'.join(
            l['text'] for l in corrected_lines if not l.get('is_likely_non_text')
        )

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

    # 2. Corpus text.
    #
    # Two files, because they answer different questions and one cannot do
    # both. The reflowed .txt is the training-data product: paragraphs, no
    # running headers, hyphen breaks healed, NFC. The .lines.txt is the
    # verbatim line-per-box output this pipeline has always produced, kept
    # because it is what you want when auditing a page against its scan.
    combined_txt_path = os.path.join(output_dir, f"{stem}_corrected.txt")
    lines_txt_path = os.path.join(output_dir, f"{stem}_corrected.lines.txt")
    jsonl_path = os.path.join(output_dir, f"{stem}_corrected.jsonl")

    reflowed_text, line_provenance = reflow_lines(all_layout_lines)
    export_combined_text(reflowed_text, combined_txt_path)
    export_combined_text(full_corrected_text, lines_txt_path)
    export_line_provenance(line_provenance, jsonl_path)

    reflow_stats = corpus_stats(reflowed_text)
    verbatim_stats = corpus_stats(full_corrected_text)

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
        'correction_engine': 'rule',
        'output_directory': output_dir,
        'generated_files': {
            'text_files': txt_files,
            'combined_text': combined_txt_path,
            'verbatim_lines_text': lines_txt_path,
            'line_provenance': jsonl_path,
            'pdf': pdf_out_path,
            'images': saved_images_paths
        },
        # Mean chars/line before and after reflow. tools/filter_literary_corpus.py
        # rejects a file as OCR output below ~55, so this is the number that
        # decides whether this document's text is usable as corpus at all.
        'corpus_stats': {
            'reflowed': reflow_stats,
            'verbatim_lines': verbatim_stats
        },
        'corrections_summary': all_corrections,
        'pages': pages_result
    }

    json_report_path = os.path.join(output_dir, f"{stem}_report.json")
    export_json_report(report_data, json_report_path)

    if progress_callback:
        progress_callback({'stage': 'finalizing', 'message': 'Finalizing outputs and preparing report...', 'percent': 98})



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
        'lines_txt_path': lines_txt_path,
        'jsonl_path': jsonl_path,
        'corpus_stats': report_data['corpus_stats'],
        'output_dir': output_dir,
        'latency_seconds': total_time
    }
