"""
OCR configuration benchmark.

Sweeps Tesseract settings over one or more page images and reports which
combination produces the cleanest Kannada.

Two proxy metrics are used so the sweep can run without hand-transcribed ground
truth. Neither is accuracy, but both move with it and they fail in different
ways, so agreement between them is meaningful:

  mean_conf  - Tesseract's own mean word confidence. Measures how sure the
               recognizer is, but a confidently-misread glyph still scores high.
  dict_rate  - share of Kannada tokens the correction engine recognizes as valid
               surface forms. Measures whether output is real Kannada, but is
               capped by vocabulary coverage, so it under-rewards rare words.
  latin      - count of Latin-script runs in what should be Kannada text. These are
               misrecognized glyphs, and they are pure loss: the correction engine
               skips non-Kannada tokens, so nothing downstream can repair them.
               This column exists because dict_rate alone is gameable -- adding
               'eng' to the language list pushes hard glyphs out of the Kannada
               token set entirely, removing them from dict_rate's denominator and
               making a strictly worse configuration score higher.

If a ground-truth .txt is supplied alongside an image (same name, .txt
extension) character error rate is reported too, and ranking switches to CER.

Usage:
    python tools/ocr_bench.py web/uploads/*.jpeg
    python tools/ocr_bench.py page.png --psm 3,4,6 --models kan,kanbest
"""

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from pipeline.ingestion import load_and_preprocess_image, normalize_resolution
from pipeline.ocr import ocr_image_with_layout
from pipeline.correction import load_dictionary, get_dictionary
from pipeline.correction.corrector import is_valid_surface_word
from pipeline.correction.dictionary import is_kannada_word
from pipeline.correction.tokenizer import tokenize


def char_error_rate(hyp: str, ref: str) -> float:
    """Levenshtein distance over characters, normalized by reference length."""
    ref = ' '.join(ref.split())
    hyp = ' '.join(hyp.split())
    if not ref:
        return 0.0 if not hyp else 1.0

    prev = list(range(len(hyp) + 1))
    for j, rc in enumerate(ref, 1):
        cur = [j]
        for i, hc in enumerate(hyp, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (hc != rc)))
        prev = cur
    return prev[len(hyp)] / len(ref)


def word_error_rate(hyp: str, ref: str) -> float:
    """Levenshtein distance over whitespace-split words, normalized by reference word count."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    prev = list(range(len(hyp_words) + 1))
    for j, rw in enumerate(ref_words, 1):
        cur = [j]
        for i, hw in enumerate(hyp_words, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (hw != rw)))
        prev = cur
    return prev[len(hyp_words)] / len(ref_words)


LATIN_RUN_RE = re.compile(r'[A-Za-z]{2,}')


def score_text(text: str, dictionary) -> dict:
    """Compute the vocabulary-validity proxy plus the Latin-contamination count."""
    latin = len(LATIN_RUN_RE.findall(text))
    tokens = [t['value'] for t in tokenize(text) if t['type'] == 'kannada']
    kn = [t for t in tokens if is_kannada_word(t)]
    if not kn:
        return {'tokens': 0, 'dict_rate': 0.0, 'latin': latin}
    valid = sum(1 for t in kn if is_valid_surface_word(t, dictionary))
    return {'tokens': len(kn), 'dict_rate': valid / len(kn), 'latin': latin}


def run_config(img, lang, psm, oem, dictionary):
    start = time.time()
    try:
        lines = ocr_image_with_layout(img, lang=lang, psm=psm, oem=oem)
    except Exception as e:
        return {'error': str(e)[:60]}

    text = '\n'.join(l['text'] for l in lines)
    confs = [l['conf'] for l in lines if 'conf' in l]
    metrics = score_text(text, dictionary)
    metrics.update({
        'mean_conf': sum(confs) / len(confs) if confs else 0.0,
        'lines': len(lines),
        'secs': time.time() - start,
        'text': text,
    })
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--models', default='kan,kanbest',
                    help='comma-separated traineddata names in tessdata/ (default: kan,kanbest)')
    ap.add_argument('--psm', default='3,4,6', help='comma-separated PSM values (default: 3,4,6)')
    ap.add_argument('--oem', default='1', help='comma-separated OEM values (default: 1)')
    ap.add_argument('--upscale', default='0,1', help='0=off, 1=on (default: both)')
    ap.add_argument('--with-eng', action='store_true', help='also try <model>+eng')
    ap.add_argument('--dump', metavar='DIR', help='write each config output to DIR')
    args = ap.parse_args()

    load_dictionary()
    dictionary = get_dictionary()
    print(f"dictionary: {len(dictionary)} surface forms\n")

    models = args.models.split(',')
    if args.with_eng:
        models = models + [f'{m}+eng' for m in models]

    for image_path in args.images:
        gt_path = os.path.splitext(image_path)[0] + '.txt'
        ground_truth = None
        if os.path.exists(gt_path):
            ground_truth = open(gt_path, encoding='utf-8').read()

        base = Image.open(image_path)
        print(f"=== {os.path.basename(image_path)}  {base.size[0]}x{base.size[1]}"
              f"{'  [ground truth found]' if ground_truth else ''} ===")

        rows = []
        for up in [bool(int(u)) for u in args.upscale.split(',')]:
            img = load_and_preprocess_image(image_path, enhance_contrast=True, upscale=up)
            for model in models:
                for psm in [int(v) for v in args.psm.split(',')]:
                    for oem in [int(v) for v in args.oem.split(',')]:
                        r = run_config(img, model, psm, oem, dictionary)
                        if 'error' in r:
                            print(f"  {model:12} psm={psm} oem={oem} up={int(up)}  ERROR: {r['error']}")
                            continue
                        if ground_truth is not None:
                            r['cer'] = char_error_rate(r['text'], ground_truth)
                        r.update({'model': model, 'psm': psm, 'oem': oem, 'up': int(up),
                                  'size': img.size})
                        rows.append(r)

                        if args.dump:
                            os.makedirs(args.dump, exist_ok=True)
                            tag = f"{model.replace('+','_')}_psm{psm}_oem{oem}_up{int(up)}"
                            with open(os.path.join(args.dump, tag + '.txt'), 'w',
                                      encoding='utf-8') as f:
                                f.write(r['text'])

        if not rows:
            print("  no successful configurations\n")
            continue

        if ground_truth is not None:
            rows.sort(key=lambda r: r['cer'])
            header = f"  {'model':12} {'psm':>3} {'oem':>3} {'up':>2} {'CER':>7} {'dict':>6} {'latin':>5} {'conf':>6} {'toks':>5} {'sec':>5}"
        else:
            # Rank by validity net of Latin contamination. Each Latin run is a token
            # that should have been Kannada and is unrecoverable downstream, so charge
            # it against the score rather than letting it silently leave the denominator.
            # Confidence breaks ties, but is not trusted alone: a wrong-but-crisp glyph
            # reads as high confidence.
            def rank(r):
                penalty = r['latin'] / max(1, r['tokens'] + r['latin'])
                return (-(r['dict_rate'] - penalty), -r['mean_conf'])
            rows.sort(key=rank)
            header = f"  {'model':12} {'psm':>3} {'oem':>3} {'up':>2} {'dict':>6} {'latin':>5} {'conf':>6} {'toks':>5} {'lines':>5} {'sec':>5}"

        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in rows:
            if ground_truth is not None:
                print(f"  {r['model']:12} {r['psm']:>3} {r['oem']:>3} {r['up']:>2} "
                      f"{r['cer']:>7.3f} {r['dict_rate']:>6.3f} {r['latin']:>5} "
                      f"{r['mean_conf']:>6.1f} {r['tokens']:>5} {r['secs']:>5.1f}")
            else:
                print(f"  {r['model']:12} {r['psm']:>3} {r['oem']:>3} {r['up']:>2} "
                      f"{r['dict_rate']:>6.3f} {r['latin']:>5} {r['mean_conf']:>6.1f} "
                      f"{r['tokens']:>5} {r['lines']:>5} {r['secs']:>5.1f}")

        best = rows[0]
        print(f"\n  best: {best['model']} psm={best['psm']} oem={best['oem']} "
              f"upscale={best['up']} (rendered at {best['size'][0]}x{best['size'][1]})\n")


if __name__ == '__main__':
    main()
