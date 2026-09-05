"""
Subprocess entry point for Surya OCR. RUNS UNDER venv-surya, NOT the pipeline's
interpreter -- see pipeline/ocr/surya_engine.py for why the two environments
cannot be merged (Surya pins pillow<11).

Nothing here may import from `pipeline`: that package's dependencies are not
installed in venv-surya. Contract is JSON on disk, deliberately.

    <surya-python> _surya_runner.py OUT.json IMG1 [IMG2 ...]

writes {image_path: [{text, left, top, width, height, conf}, ...]}.
"""

import json
import sys
import warnings

warnings.filterwarnings('ignore')


def main() -> int:
    out_path, image_paths = sys.argv[1], sys.argv[2:]
    if not image_paths:
        print('usage: _surya_runner.py OUT.json IMG [IMG ...]', file=sys.stderr)
        return 2

    from PIL import Image
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    # Loaded once for the whole batch -- this is why the caller batches pages.
    foundation = FoundationPredictor()
    recognition = RecognitionPredictor(foundation)
    detection = DetectionPredictor()

    result = {}
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        # sort_lines=True gives reading order; the pipeline re-sorts by
        # coordinates anyway, but this keeps the raw dump readable.
        page = recognition([image], det_predictor=detection, sort_lines=True)[0]
        lines = []
        for tl in page.text_lines:
            x0, y0, x1, y1 = tl.bbox
            lines.append({
                'text': tl.text,
                'left': int(x0), 'top': int(y0),
                'width': int(x1 - x0), 'height': int(y1 - y0),
                'conf': float(tl.confidence) if tl.confidence is not None else None,
            })
        result[path] = lines

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
