"""
Surya OCR engine, behind a flag, as an alternative to Tesseract.

WHY THIS EXISTS. Two independent measurements said the remaining headroom is in
the OCR pass and not in post-hoc correction: roughly half the residual
single-word substitutions have a target absent from the dictionary (no
generator can reach them), and of the errors whose truth IS reachable, the ones
that still fail need evidence neither corpus frequency nor OCR confidence can
supply. See CLAUDE.md.

Measured on the nine real page/transcript pairs, raw OCR, each engine with its
own best non-text filtering:

    Tesseract   CER 0.0539   WER 0.2847
    Surya       CER 0.0461   WER 0.2628

WHY IT RUNS OUT OF PROCESS. Surya pins pillow<11; this pipeline uses PIL
throughout ingestion and runs pillow 12.x. Installing them into one environment
downgrades pillow underneath the working pipeline. So Surya lives in its own
virtualenv (venv-surya, gitignored) and is driven through a subprocess that
speaks JSON. That also keeps torch, CUDA and a multi-GB model out of the
default install: nothing here is imported unless --engine surya is asked for.

    python3 -m venv venv-surya
    ./venv-surya/bin/pip install 'surya-ocr==0.16.7' 'transformers>=4.56.1,<5'
    # match the CUDA build to your driver; on a 12.8 driver:
    ./venv-surya/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0+cu128 torchvision

Pinned to 0.16.7 deliberately. Surya 0.22 dropped the in-process torch
predictor: it drives inference through vLLM (spawning a Docker container),
llama.cpp, or a remote OpenAI-compatible endpoint, none of which is a
reasonable dependency for a batch page-scanning pipeline. 0.16.7 is the last
line with the plain `FoundationPredictor` + `RecognitionPredictor` path.

LICENSING. The Surya code is Apache-2.0. The model weights are under a modified
AI Pubs Open RAIL-M licence: free for research, personal use, and organisations
under $5M in funding/revenue. Unlike the Krutrim models this project excludes
(see CLAUDE.md), there is no non-compete clause, and Datalab does not ship a
competing Indic LLM, so using it to build a Kannada training corpus does not sit
inside any restriction. The OpenRAIL-M use restrictions are harm-based and
propagate to derivatives -- worth a read before the corpus is redistributed.
"""

import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SURYA_PYTHON = os.path.join(BASE_DIR, 'venv-surya', 'bin', 'python')
_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_surya_runner.py')

# Surya emits confidence in [0, 1]; everything downstream (NON_TEXT_LINE_CONFIDENCE,
# HIGH_OCR_CONFIDENCE_TRUST) is calibrated on Tesseract's 0-100, so scale on the way in.
_CONF_SCALE = 100.0


def surya_python() -> str:
    """Interpreter for the Surya virtualenv. SURYA_PYTHON overrides."""
    return os.environ.get('SURYA_PYTHON', DEFAULT_SURYA_PYTHON)


def is_surya_available() -> bool:
    return os.path.exists(surya_python()) and os.path.exists(_RUNNER)


def ocr_images_with_layout(
    images: List[Image.Image],
    page_nums: Optional[List[int]] = None,
    timeout: int = 3600
) -> List[List[Dict[str, Any]]]:
    """
    OCR a batch of PIL images, returning one layout-line list per image in the
    same shape ocr_image_with_layout produces for Tesseract.

    Batched on purpose: the model load dominates a single-page call (tens of
    seconds against roughly ten seconds of actual work per page), so callers
    should hand over every page of a document at once rather than looping.
    """
    if not is_surya_available():
        raise RuntimeError(
            "Surya is not installed. Create venv-surya and install surya-ocr "
            "-- see pipeline/ocr/surya_engine.py for the exact commands."
        )

    page_nums = page_nums or list(range(1, len(images) + 1))
    tmpdir = tempfile.mkdtemp(prefix='surya_')
    try:
        paths = []
        for i, img in enumerate(images):
            p = os.path.join(tmpdir, '%04d.png' % i)
            img.convert('RGB').save(p)
            paths.append(p)

        out_path = os.path.join(tmpdir, 'out.json')
        proc = subprocess.run(
            [surya_python(), _RUNNER, out_path] + paths,
            capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            tail = (proc.stderr or proc.stdout or '')[-2000:]
            raise RuntimeError("Surya subprocess failed:\n%s" % tail)

        raw = json.load(open(out_path, encoding='utf-8'))
        return [
            _to_layout_lines(raw.get(p, []), images[i].size[0], page_nums[i])
            for i, p in enumerate(paths)
        ]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _to_layout_lines(lines: List[Dict[str, Any]], img_w: int, page_num: int) -> List[Dict[str, Any]]:
    """Map Surya text lines onto the layout-line contract the pipeline expects."""
    out = []
    for l in lines:
        left, top = int(l['left']), int(l['top'])
        width, height = int(l['width']), int(l['height'])
        center = left + width / 2.0
        page_center = img_w / 2.0
        if width < img_w * 0.7 and abs(center - page_center) < (img_w * 0.1):
            alignment = 'C'
        elif left > (img_w * 0.55):
            alignment = 'R'
        else:
            alignment = 'L'

        conf = l.get('conf')
        conf = float(conf) * _CONF_SCALE if conf is not None else None
        out.append({
            'text': l['text'],
            'alignment': alignment,
            'left': left, 'top': top, 'width': width, 'height': height,
            'page_num': page_num,
            'conf': conf,
            # Surya scores a whole line, not each word. Repeating the line's
            # score per word would fabricate per-word evidence that
            # HIGH_OCR_CONFIDENCE_TRUST would then act on, so leave it empty:
            # an absent confidence means "no signal", which those gates already
            # handle, whereas a wrong one silently changes their decisions.
            'word_confidences': []
        })
    out.sort(key=lambda l: (l['top'], l['left']))
    return out
