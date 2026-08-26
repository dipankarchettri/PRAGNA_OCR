"""
Text and JSON Exporter
Handles structured per-page text exports and comprehensive JSON reporting.
"""

import os
import json
from typing import List, Dict, Any


def export_pages_to_text(
    pages: List[Dict[str, Any]],
    output_dir: str,
    prefix: str = "page_"
) -> List[str]:
    """
    Save each page's extracted/corrected text as individual .txt files.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for page in pages:
        page_num = page.get('page_num', 1)
        filename = f"{prefix}{page_num:03d}.txt"
        file_path = os.path.join(output_dir, filename)

        text_content = page.get('corrected_text', page.get('raw_text', ''))
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        saved_files.append(file_path)

    return saved_files


def export_combined_text(
    text: str,
    output_path: str
) -> str:
    """Save full combined text to a single file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return output_path


def export_json_report(
    report_data: Dict[str, Any],
    output_path: str
) -> str:
    """Save detailed JSON analysis report."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    return output_path
