#!/usr/bin/env python3
"""
Kannada OCR & Autocorrect CLI
Unified CLI tool for ingesting PDFs/images, performing multi-language OCR,
cleaning Indic OCR errors, and exporting corrected documents.

Examples:
    python cli.py sample.pdf
    python cli.py document.png --lang kan --dpi 400 --output-dir ./results
    python cli.py --text "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತದೆ"
    python cli.py --batch ./scans/ --output-dir ./all_results/
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import (
    process_document,
    process_text_input,
    init_pipeline
)
from pipeline.ocr import (
    is_tesseract_available, is_surya_available,
    SUPPORTED_LANGUAGES, DEFAULT_PSM, DEFAULT_OEM, OCR_ENGINES
)
from pipeline.ingestion import SUPPORTED_IMAGE_EXTENSIONS


def print_banner():
    banner = r"""
======================================================================
  ಕನ್ನಡ OCR & ಸ್ವಯಂ ತಿದ್ದುಪಡಿ — Kannada OCR & Autocorrect Pipeline
======================================================================
"""
    print(banner)


def handle_single_file(args):
    filepath = os.path.abspath(args.input_file)
    if not os.path.isfile(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print(f"[*] Processing file: {Path(filepath).name}")
    print(f"    Language      : {args.lang}")
    print(f"    DPI           : {args.dpi}")
    print(f"    Engine        : {args.engine}")
    print(f"    PSM / OEM     : {args.psm} / {args.oem}")
    print(f"    Generate PDF  : {'Yes' if not args.no_pdf else 'No'}")
    print(f"    Save Images   : {'Yes' if args.save_images else 'No'}\n")

    def progress_cb(info):
        # process_document reports progress as a single dict, not (current, total).
        msg = info.get('message', info.get('stage', ''))
        if info.get('total_pages'):
            print(f"    -> [{info.get('percent', 0):3d}%] page "
                  f"{info.get('current_page', '?')}/{info['total_pages']}: {msg}")
        else:
            print(f"    -> [{info.get('percent', 0):3d}%] {msg}")

    try:
        res = process_document(
            input_path=filepath,
            lang=args.lang,
            dpi=args.dpi,
            psm=args.psm,
            oem=args.oem,
            min_confidence=args.min_confidence,
            adaptive_contrast=not args.no_adaptive_contrast,
            engine=args.engine,
            output_dir=args.output_dir,
            save_pdf=not args.no_pdf,
            save_images=args.save_images,
            progress_callback=progress_cb
        )

        print("\n[✓] Processing Complete!")
        print(f"    Total Pages     : {res['total_pages']}")
        print(f"    Total Fixes     : {res['total_corrections']}")
        print(f"    Time Elapsed    : {res['latency_seconds']}s")
        print(f"    Output Directory: {res['output_dir']}")
        if res['pdf_path']:
            print(f"    Corrected PDF   : {res['pdf_path']}")
        print(f"    Corrected Text  : {res['combined_txt_path']}")
        print(f"    JSON Report     : {res['json_path']}")

        if args.json:
            print("\n--- JSON Result ---")
            print(json.dumps(res['report'], ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"\n[!] Pipeline Error: {e}")
        sys.exit(1)


def handle_batch(args):
    folder = os.path.abspath(args.batch)
    if not os.path.isdir(folder):
        print(f"Error: Batch directory not found: {folder}")
        sys.exit(1)

    allowed = {'.pdf'} | SUPPORTED_IMAGE_EXTENSIONS
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in allowed]

    if not files:
        print(f"No supported PDF or image files found in: {folder}")
        sys.exit(0)

    print(f"[*] Found {len(files)} file(s) to process in batch mode.")
    base_out = args.output_dir or os.path.join(folder, "batch_processed_results")
    os.makedirs(base_out, exist_ok=True)

    for idx, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        stem = Path(fpath).stem
        item_out = os.path.join(base_out, stem)
        print(f"\n[{idx}/{len(files)}] Processing: {fname}")

        try:
            res = process_document(
                input_path=fpath,
                lang=args.lang,
                dpi=args.dpi,
                psm=args.psm,
                oem=args.oem,
                min_confidence=args.min_confidence,
                adaptive_contrast=not args.no_adaptive_contrast,
                engine=args.engine,
                output_dir=item_out,
                save_pdf=not args.no_pdf,
                save_images=args.save_images
            )
            print(f"    -> Pages: {res['total_pages']} | Corrections: {res['total_corrections']} | Time: {res['latency_seconds']}s")
        except Exception as e:
            print(f"    -> [!] Failed: {e}")

    print(f"\n[✓] Batch processing complete! Results saved in: {base_out}")


def handle_text(args):
    print("[*] Correcting input text...\n")
    res = process_text_input(args.text)

    print("--- Original ---")
    print(res['original'])
    print("\n--- Corrected ---")
    print(res['corrected'])
    print(f"\nTotal Corrections: {res['total_corrections']} (Latency: {res['latency_seconds']}s)")

    if res['corrections']:
        print("\nDetailed Corrections:")
        for c in res['corrections']:
            print(f"  • '{c['original']}' -> '{c['correction']}' (dist: {c['edit_distance']})")

    if args.json:
        print("\n--- JSON Result ---")
        print(json.dumps(res, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Kannada OCR & Autocorrect Pipeline — Ingest, Extract, Clean, Correct, and Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported Indic Language Codes:
  {', '.join(f'{code} ({name})' for code, name in SUPPORTED_LANGUAGES.items())}

Combine multiple languages with '+': --lang kan+eng or --lang san+kan
Prefer a single script when the page is monolingual -- adding 'eng' to a pure
Kannada page lets Tesseract emit Latin for ambiguous glyphs.
        """
    )

    parser.add_argument('input_file', nargs='?', help='Path to PDF document or image file to process')
    parser.add_argument('--text', '-t', help='Direct text string to clean and autocorrect')
    parser.add_argument('--batch', '-b', help='Directory of PDFs/images to process in batch mode')
    parser.add_argument('--lang', '-l', default='kan', help='Tesseract OCR language code(s) (default: kan)')
    parser.add_argument('--dpi', '-d', type=int, default=400, help='DPI for PDF page rasterization (default: 400)')
    parser.add_argument('--psm', type=int, default=DEFAULT_PSM, help=f'Tesseract page segmentation mode (default: {DEFAULT_PSM} = automatic; try 6 on a page that is one uniform text block)')
    parser.add_argument('--oem', type=int, default=DEFAULT_OEM, help=f'Tesseract OCR engine mode (default: {DEFAULT_OEM} = LSTM only)')
    parser.add_argument('--engine', choices=OCR_ENGINES, default='tesseract',
                        help='OCR engine (default: tesseract). "surya" needs venv-surya '
                             '-- see pipeline/ocr/surya_engine.py. Lower CER on real scans '
                             '(0.0462 vs 0.0516 end-to-end) but ignores --lang/--psm/--oem.')
    parser.add_argument('--min-confidence', type=int, default=0, help='Drop OCR words below this confidence (0-100, default: 0 = keep all)')
    parser.add_argument('--no-adaptive-contrast', action='store_true', help='Skip the extra contrast-boosted OCR pass (faster, but misses gains on faded scans)')
    parser.add_argument('--output-dir', '-o', help='Output directory for generated files')
    parser.add_argument('--no-pdf', action='store_true', help='Skip generating corrected PDF document')
    parser.add_argument('--save-images', '-s', action='store_true', help='Save rasterized intermediate page images')
    parser.add_argument('--json', '-j', action='store_true', help='Print machine-readable JSON output')

    args = parser.parse_args()
    print_banner()

    if args.text:
        handle_text(args)
    elif args.batch:
        handle_batch(args)
    elif args.input_file:
        handle_single_file(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
