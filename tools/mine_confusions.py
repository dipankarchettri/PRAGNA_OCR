#!/usr/bin/env python3
"""
Derive optical confusion pairs empirically from real page/transcript pairs.

This is how rows get added to CONFUSION_PAIRS in
pipeline/correction/edit_distance.py. It exists because the intuitive method --
eyeball which glyphs OCR confuses and add those -- produces the wrong list.

Two questions, and only the second one decides:

  1. WHAT DID OCR GET WRONG?  Align raw Tesseract output against the human
     transcript word by word, keep pairs of equal akshara length that differ in
     exactly one cluster, and tally the cluster pair. Reported as `obs`/`pages`.

  2. WHAT CAN THE ENGINE NOT REACH?  A confusion row only pays off when
     rewriting through it lands on a word the dictionary will accept. So each
     candidate pair is injected into the live table and its instances are
     re-tested: how many move from absent-in-collect_kannada_candidates to
     present. Reported as `gain`.

The two rankings disagree badly. On the nine pages in tests/fixtures/real/, the
most-observed unlisted pair (ಕ/ಯ, 11 instances over 9 word types) has a gain of
zero -- its instances are subscripts already covered by the ್ಯ/್ಕ row, plus one
out-of-dictionary place name. Add on `gain`, never on `obs`.

Then measure. A pair that buys reachability can still cost precision, which
tools/correction_bench.py is what actually decides:

    ./venv/bin/python tools/correction_bench.py --pages 'tests/fixtures/real/*.png'

and note that its --synthetic corpus is generated FROM GLYPH_CONFUSIONS, so its
numbers are not comparable across a change to this table (see CLAUDE.md).

Usage:
    ./venv/bin/python tools/mine_confusions.py
    ./venv/bin/python tools/mine_confusions.py --pages 'tests/fixtures/real/*.png' --min-pages 2
"""

import argparse
import contextlib
import difflib
import glob
import io
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KANNADA_WORD_RE = re.compile(r'^[ಀ-೿‌‍]+$')
PUNCT = '.,;:!?“”‘’()—-।॥"\''


def _words(text: str) -> List[str]:
    return [w for w in (t.strip(PUNCT) for t in text.split())
            if w and KANNADA_WORD_RE.match(w)]


def mine(docs) -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
    """pair -> [(ocr_word, truth_word, page)] for single-cluster misreads."""
    from pipeline.correction.graphemes import aksharas, split_cluster

    found: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = defaultdict(list)
    for doc in docs:
        ocr_w, gt_w = _words(doc.raw_text), _words(doc.reference)
        sm = difflib.SequenceMatcher(a=ocr_w, b=gt_w, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            # An unequal-size replace block gives no reliable word alignment
            # inside it, so pairing positionally there would invent evidence.
            if tag != 'replace' or (i2 - i1) != (j2 - j1):
                continue
            for o, g in zip(ocr_w[i1:i2], gt_w[j1:j2]):
                ao, ag = aksharas(o), aksharas(g)
                if len(ao) != len(ag):
                    continue
                diff = [k for k in range(len(ao)) if ao[k] != ag[k]]
                if len(diff) != 1:
                    continue
                co, cg = ao[diff[0]], ag[diff[0]]
                bo, mo = split_cluster(co)
                bg, mg = split_cluster(cg)
                if bo != bg and mo == mg:
                    found[(bo, bg)].append((o, g, doc.name))       # base swap
                elif bo == bg and mo != mg:
                    found[(''.join(mo), ''.join(mg))].append((o, g, doc.name))
    return found


def _inject(a: str, b: str, cost: float) -> None:
    from pipeline.correction import edit_distance as ED
    ED.CONFUSION_PAIRS[(a, b)] = cost
    ED._COST_MAP[frozenset([a, b])] = cost
    for x, y in ((a, b), (b, a)):
        ED.GLYPH_CONFUSIONS.setdefault(x, [])
        if not any(t[0] == y for t in ED.GLYPH_CONFUSIONS[x]):
            ED.GLYPH_CONFUSIONS[x].append((y, cost))


def _retract(a: str, b: str) -> None:
    from pipeline.correction import edit_distance as ED
    ED.CONFUSION_PAIRS.pop((a, b), None)
    ED._COST_MAP.pop(frozenset([a, b]), None)
    for x, y in ((a, b), (b, a)):
        if x in ED.GLYPH_CONFUSIONS:
            ED.GLYPH_CONFUSIONS[x] = [t for t in ED.GLYPH_CONFUSIONS[x] if t[0] != y]
            if not ED.GLYPH_CONFUSIONS[x]:
                del ED.GLYPH_CONFUSIONS[x]


def _reachable(instances, dictionary) -> Tuple[int, List[str]]:
    """How many instances have the truth in the candidate set right now."""
    from pipeline.correction import corrector
    n, got = 0, []
    for o, g, _page in instances:
        corrector.clear_correction_caches()
        cands = corrector.collect_kannada_candidates(o, dictionary)
        if cands is not None and g in cands:
            n += 1
            got.append('%s->%s' % (o, g))
    return n, got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pages', default='tests/fixtures/real/*.png',
                    help='glob of page images with same-basename .txt transcripts')
    ap.add_argument('--lang', default='kan')
    ap.add_argument('--min-obs', type=int, default=2, help='minimum observations')
    ap.add_argument('--min-pages', type=int, default=2,
                    help='minimum distinct pages -- guards against one bad scan')
    ap.add_argument('--cost', type=float, default=0.25,
                    help='cost to price injected pairs at while testing')
    args = ap.parse_args()

    from pipeline import init_pipeline
    from pipeline.ocr.tesseract_engine import DEFAULT_OEM, DEFAULT_PSM
    init_pipeline()
    from pipeline.correction.dictionary import get_dictionary
    from pipeline.correction.edit_distance import CONFUSION_PAIRS
    from tools import correction_bench as CB

    paths = sorted(glob.glob(args.pages))
    if not paths:
        print('no pages matched %r' % args.pages)
        sys.exit(1)

    print('OCR-ing %d pages at production settings (psm %d, oem %d)...'
          % (len(paths), DEFAULT_PSM, DEFAULT_OEM))
    with contextlib.redirect_stdout(io.StringIO()):
        docs = CB.load_page_docs(paths, args.lang, DEFAULT_PSM, DEFAULT_OEM)

    found = mine(docs)
    known = {frozenset(p) for p in CONFUSION_PAIRS}
    dictionary = get_dictionary()

    rows = []
    for pair, instances in found.items():
        if frozenset(pair) in known:
            continue
        # Merge the mirrored direction: a confusion row is symmetric.
        merged = list(instances) + list(found.get((pair[1], pair[0]), []))
        if pair[1] < pair[0] and (pair[1], pair[0]) in found:
            continue                      # already counted from the other side
        # A pair with an empty side is an inserted or dropped mark, not a
        # substitution. CONFUSION_PAIRS has no way to express it (the row would
        # key on ''), and it is already the job of the deletion/insertion
        # generators in collect_kannada_candidates, so don't offer it.
        if not pair[0] or not pair[1]:
            continue
        pages = {p for _, _, p in merged}
        if len(merged) < args.min_obs or len(pages) < args.min_pages:
            continue
        before, _ = _reachable(merged, dictionary)
        _inject(pair[0], pair[1], args.cost)
        after, got = _reachable(merged, dictionary)
        _retract(pair[0], pair[1])
        lemmas = {g for _, g, _ in merged}
        rows.append((after - before, len(merged), len(pages), len(lemmas), pair, got))

    rows.sort(key=lambda r: (-r[0], -r[2], -r[1]))
    print('\n%-6s %-6s %5s %6s %7s %6s  %s'
          % ('a', 'b', 'obs', 'pages', 'lemmas', 'gain', 'newly reachable'))
    print('-' * 96)
    for gain, obs, pages, lemmas, (a, b), got in rows:
        print('%-6s %-6s %5d %6d %7d %6d  %s'
              % (a or '∅', b or '∅', obs, pages, lemmas, gain, ', '.join(got)[:52]))

    print('\nAdd rows with gain > 0 and lemmas > 1. A gain of 0 means the pair is')
    print('already covered or its targets are not in the dictionary -- adding it only')
    print('widens the search with no upside. Then re-run tools/correction_bench.py:')
    print('a pair that raises `broke` is a regression however much it raises `fixed`.')


if __name__ == '__main__':
    main()
