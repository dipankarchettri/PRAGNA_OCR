"""
Turn OCR line boxes into a training corpus.

The pipeline's exported .txt has always been one line per detected line box.
That is faithful to the page and close to useless as LLM training data: a hard
newline every ~48 characters, running headers inlined as if they were prose,
words split across a line-end hyphen becoming two junk tokens, and `\\n` doing
double duty as both "the print line wrapped here" and "a new paragraph starts
here".

How wrong that is has a number attached in this repo already. `tools/
filter_literary_corpus.py` rejects any text file averaging under ~55
characters per line as OCR output rather than prose. This pipeline's own
export measures 48.4. **It emits corpus text that the project's own corpus
filter would throw away.**

Reflow reverses the page layout to recover the paragraphs:

  1. drop lines flagged as non-text (OCR hallucinating on photos/banner art)
  2. drop running headers, footers and page numbers, detected by *positional
     repetition across pages* rather than by matching any particular string
  3. rejoin words broken across a line-end hyphen
  4. join wrapped lines into paragraphs, breaking only where the layout says
     a paragraph actually ended
  5. normalise to NFC

Nothing here is Kannada-specific beyond the terminal-punctuation set, and
nothing is tuned to a particular book: every decision is made against
statistics of the document being processed.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# A line ending in one of these is a finished sentence, so the next line starts
# a new one. Danda and double danda are the Indic sentence terminators; the
# rest cover Latin-script and quoted material in the same books.
TERMINAL_PUNCTUATION = ('.', '?', '!', '।', '॥', ':', '"', '"', "'", ')', '»')

# Line-end hyphens that indicate a word was broken across the line.
HYPHENS = ('-', '‐', '‑', '‒', '–', '­')

# A line shorter than this fraction of the page's typical line width ended
# early, which for justified or full-measure body text means the paragraph
# ended there. Deliberately generous: 0.75 would treat an ordinary slightly
# short line as a paragraph break and shatter the text back into fragments.
SHORT_LINE_RATIO = 0.62

# Extra indent, as a fraction of typical line width, that marks a new
# paragraph's first line.
INDENT_RATIO = 0.04

# A header/footer must repeat on at least this fraction of pages, and this many
# pages absolutely, before it is removed. Both bars exist: the fraction alone
# would strip real text from a 2-page document, the count alone would miss a
# header that only appears in one chapter of a long book.
HEADER_REPEAT_FRACTION = 0.5
HEADER_MIN_PAGES = 3

# A page needs at least this many lines before "near the top or bottom" means
# anything. On a page holding two or three lines, every line is simultaneously
# near the top and near the bottom, so the band check stops discriminating and
# ordinary body text becomes a header candidate -- which deletes it. Measured
# real pages carry 30-45 lines, so this only excludes genuinely sparse pages,
# where the safe answer is to remove nothing.
HEADER_MIN_LINES_PER_PAGE = 5

# How far from the top/bottom of the page a line must be to be *considered* a
# running header/footer at all, as a fraction of page height. Repetition alone
# is not enough -- a refrain in a poem repeats too, but it repeats mid-page.
HEADER_BAND = 0.16

_DIGITS_RE = re.compile(r'\d+|[೦-೯]+')
_WS_RE = re.compile(r'\s+')


def _normalize_for_repetition(text: str) -> str:
    """
    Collapse a line to what makes it recognisable across pages.

    Page numbers change every page, so a running footer is only detectable
    once the numbers are masked -- "ಅಧ್ಯಾಯ ೪ 57" and "ಅಧ್ಯಾಯ ೪ 58" have to
    compare equal. Kannada digits are masked alongside ASCII ones because
    these books number their pages in either.
    """
    t = _DIGITS_RE.sub('#', text)
    return _WS_RE.sub(' ', t).strip().lower()


def _median(values: List[float]) -> float:
    """
    Lower median. For an even-length list this takes the smaller of the two
    middle values, which matters for the line-pitch estimate: paragraph gaps
    are larger and rarer than line gaps, so on a page with few samples the
    upper median can land on a paragraph gap and declare it "typical", after
    which no gap ever looks big enough to break a paragraph.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def _page_geometry(lines: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """
    (typical line width, left margin, same-line threshold, paragraph pitch).

    Derived from glyph HEIGHT rather than from the spacing between successive
    extracted items, because that spacing is not trustworthy. Text layers
    routinely split one printed line into many fragments -- a real page here
    yields five pieces at tops 25.2, 26.5, 27.9, 30.1, 31.0, all the same
    printed line, drifting a point or two because the scan is slightly skewed.
    Taking the median gap over those fragments measures the drift *within* a
    line, not the distance *between* lines, and everything built on it inverts:
    every fragment then looks further apart than "typical" and so starts a new
    paragraph.

    Line height does not depend on how the extractor chose to segment, so it
    survives that. Two thresholds come off it: below `same_line` two items are
    side by side on one printed line; above `pitch` there is a real vertical
    gap, and the pitch itself is re-measured from only those gaps that already
    cleared `same_line`, so fragment drift cannot drag it down.
    """
    widths = [l.get('width', 0) or 0 for l in lines]
    lefts = sorted(l.get('left', 0) or 0 for l in lines)
    heights = [h for h in ((l.get('height', 0) or 0) for l in lines) if h > 0]
    if not widths:
        return 0.0, 0.0, 0.0, 0.0

    # Median, not mean: one full-width figure caption or a stray wide line
    # should not redefine what "a normal line" is.
    typical_width = _median(widths)
    left_margin = lefts[len(lefts) // 4] if lefts else 0.0

    line_height = _median(heights)
    same_line = line_height * 0.6

    deltas = [
        (b.get('top', 0) or 0) - (a.get('top', 0) or 0)
        for a, b in zip(lines, lines[1:])
    ]
    real_gaps = [d for d in deltas if d > same_line]
    pitch = _median(real_gaps) if real_gaps else line_height

    return typical_width, left_margin, same_line, pitch


def group_into_rows(lines: List[Dict[str, Any]], same_line: float) -> List[Dict[str, Any]]:
    """
    Merge fragments that sit on the same printed line into one row, in
    left-to-right order.

    Sorting by (top, left) is not enough, and gets the text wrong rather than
    merely untidy. On a slightly skewed scan the fragments of one printed line
    drift downward as they go right (tops 25.2, 26.5, 27.9, 30.1, 31.0), so a
    top-major sort interleaves them with the fragments of neighbouring lines
    and emits the words out of order -- a real page here produced
    "ಒಳಿತನ್ನು ಕಾಪಾಡುವವನಾಗಿ, ಮುಗಿಸಿ, ಜನತೆಯ ಕೆಲಸವನ್ನ ತನ್ನ ಸಾಹಸದಿಂದ ಮಾಡಿ",
    whose clauses are scrambled.

    Grouping by vertical proximity first, then ordering within the row by
    `left`, restores reading order. Each row then behaves as the single "line"
    the paragraph logic expects.
    """
    if not lines:
        return []

    ordered = sorted(lines, key=lambda l: (l.get('top', 0) or 0, l.get('left', 0) or 0))
    rows: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [ordered[0]]
    row_top = ordered[0].get('top', 0) or 0

    for line in ordered[1:]:
        top = line.get('top', 0) or 0
        if same_line > 0 and (top - row_top) < same_line:
            current.append(line)
        else:
            rows.append(current)
            current = [line]
            row_top = top
    rows.append(current)

    merged: List[Dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda l: l.get('left', 0) or 0)
        lefts = [l.get('left', 0) or 0 for l in row]
        rights = [(l.get('left', 0) or 0) + (l.get('width', 0) or 0) for l in row]
        merged.append({
            'text': _join_paragraph([l.get('text', '') for l in row]),
            'top': min(l.get('top', 0) or 0 for l in row),
            'left': min(lefts),
            'width': max(rights) - min(lefts),
            'height': _median([l.get('height', 0) or 0 for l in row]),
            'alignment': row[0].get('alignment', 'L'),
            'page_num': row[0].get('page_num', 1),
            'ocr_confidence': row[0].get('ocr_confidence'),
            'parts': row,
        })
    return merged


def _in_header_band(line: Dict[str, Any], page_top: float, span: float) -> bool:
    """Whether a line sits close enough to the page edge to be a running head."""
    rel = ((line.get('top', 0) or 0) - page_top) / span
    return rel <= HEADER_BAND or rel >= (1.0 - HEADER_BAND)


def find_running_lines(pages: Dict[int, List[Dict[str, Any]]]) -> set:
    """
    Normalised forms of lines that recur at the top or bottom of many pages.

    This is the only reliable way to catch running heads: they are ordinary,
    high-confidence, correctly-OCR'd text, so the non-text confidence filter
    never touches them, and their content differs from book to book so no
    fixed pattern can match them. What distinguishes them is that they appear
    in the same place on page after page.
    """
    if len(pages) < HEADER_MIN_PAGES:
        return set()

    seen: Dict[str, set] = {}
    for page_num, lines in pages.items():
        if len(lines) < HEADER_MIN_LINES_PER_PAGE:
            continue
        tops = [l.get('top', 0) or 0 for l in lines]
        page_top, page_bottom = min(tops), max(tops)
        span = max(1.0, page_bottom - page_top)
        for line in lines:
            if not _in_header_band(line, page_top, span):
                continue
            key = _normalize_for_repetition(line.get('text', ''))
            if not key:
                continue
            seen.setdefault(key, set()).add(page_num)

    n_pages = len(pages)
    threshold = max(HEADER_MIN_PAGES, int(n_pages * HEADER_REPEAT_FRACTION))
    return {k for k, ps in seen.items() if len(ps) >= threshold}


def _is_word_char(ch: str) -> bool:
    """
    Part of a word, including Indic dependent signs.

    str.isalpha() is False for combining marks (Unicode category Mn), and most
    Kannada words end in one -- ಮಹಾ ends in the matra ಾ. Testing isalpha alone
    therefore declared almost every Kannada word "not a word", which silently
    disabled dehyphenation for the entire language it was written for.
    """
    if not ch:
        return False
    return ch.isalpha() or unicodedata.category(ch) in ('Mn', 'Mc')


def _is_bare_page_number(text: str) -> bool:
    """A line that is nothing but a page number, in either digit system."""
    stripped = _WS_RE.sub('', text)
    if not stripped or len(stripped) > 6:
        return False
    return bool(re.fullmatch(r'[-–—\[\(]*[0-9೦-೯]+[-–—\]\)\.]*', stripped))


def _dehyphenate(prev: str, nxt: str) -> Optional[str]:
    """
    Join `prev` and `nxt` if `prev` ends in a word-breaking hyphen.

    Returns the joined string, or None if this is not a hyphen break. A
    trailing hyphen is only a line-break hyphen when real word characters sit
    on both sides of it -- a dash used as punctuation ("ಅವನು -") or a line
    that is all dashes must not swallow the next line.
    """
    if not prev or not nxt:
        return None
    if not prev.endswith(HYPHENS):
        return None
    stem = prev[:-1].rstrip()
    if not stem or not _is_word_char(nxt[:1]) or not _is_word_char(stem[-1:]):
        return None
    return stem + nxt.lstrip()


def _starts_new_paragraph(
    line: Dict[str, Any],
    prev_line: Optional[Dict[str, Any]],
    typical_width: float,
    left_margin: float,
    same_line: float,
    pitch: float,
) -> bool:
    """Whether `line` begins a new paragraph rather than continuing `prev_line`."""
    if prev_line is None:
        return True

    prev_text = (prev_line.get('text') or '').rstrip()
    if not prev_text:
        return True

    delta = (line.get('top', 0) or 0) - (prev_line.get('top', 0) or 0)

    # Same printed line, side by side. Text extraction routinely splits one
    # line into several fragments -- a real page produced the four pieces
    # "ಅಕ್ಷರಶಃ" / "ಸತ್ಯ ಕರ್ನಾಟಕದ" / "ಪ್ರತಿ ಭೇಟಿಯಲ್ಲೂ" / "ಜನರಲ್ಲಿ" out of what is
    # typographically one line. These must be joined, and judging them on width
    # alone would break the paragraph at every fragment.
    if same_line > 0 and delta < same_line:
        return False

    # A vertical gap noticeably bigger than the page's own line spacing is the
    # strongest paragraph signal a page offers, and unlike line width it
    # survives fragmented extraction: whatever the pieces, the next paragraph
    # still starts further down the page.
    if pitch > 0 and delta > pitch * 1.5:
        return True

    # The previous line finished a sentence AND stopped short of the margin.
    # Both are required: full-measure justified text routinely ends a sentence
    # mid-line and carries straight on, so terminal punctuation alone would
    # break every paragraph into single sentences.
    prev_width = prev_line.get('width', 0) or 0
    ended_short = typical_width > 0 and prev_width < typical_width * SHORT_LINE_RATIO
    if prev_text.endswith(TERMINAL_PUNCTUATION) and ended_short:
        return True

    # This line is indented past the body margin: a first line.
    if typical_width > 0:
        indent = (line.get('left', 0) or 0) - left_margin
        if indent > typical_width * INDENT_RATIO:
            return True

    # Centred or right-aligned blocks (headings, verse, attributions) are their
    # own units, not continuations of running prose.
    if line.get('alignment') in ('C', 'R') or prev_line.get('alignment') in ('C', 'R'):
        return True

    return False


def _join_paragraph(parts: List[str]) -> str:
    """
    Join wrapped line fragments into one paragraph, healing hyphen breaks.

    Lines are joined with a single space. That is right for this script:
    Kannada writes spaces between words exactly as Latin does, so a wrapped
    line ends at a word boundary and the space belongs there.
    """
    out = ''
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not out:
            out = part
            continue
        joined = _dehyphenate(out, part)
        out = joined if joined is not None else out + ' ' + part
    return out


def reflow_lines(
    lines: List[Dict[str, Any]],
    drop_running: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Reflow corrected layout lines into paragraph text.

    Returns (reflowed_text, provenance) where provenance carries one record per
    surviving line -- page, bbox, confidence, the paragraph it landed in -- so
    any span of the corpus can be traced back to the pixels it came from.

    `lines` are the dicts correct_layout_lines produces.
    """
    raw_pages: Dict[int, List[Dict[str, Any]]] = {}
    for line in lines:
        if line.get('is_likely_non_text'):
            continue
        raw_pages.setdefault(line.get('page_num', 1), []).append(line)

    # Rebuild printed lines from fragments before anything else looks at the
    # page, so header detection and paragraph logic both see whole lines.
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for page_num, page_lines in raw_pages.items():
        _w, _l, same_line, _p = _page_geometry(
            sorted(page_lines, key=lambda l: (l.get('top', 0) or 0, l.get('left', 0) or 0)))
        pages[page_num] = group_into_rows(page_lines, same_line)

    running = find_running_lines(pages) if drop_running else set()

    paragraphs: List[str] = []
    provenance: List[Dict[str, Any]] = []
    current: List[str] = []
    current_meta: List[Dict[str, Any]] = []

    def flush():
        if not current:
            return
        text = _join_paragraph(current)
        if not text:
            current.clear()
            current_meta.clear()
            return
        idx = len(paragraphs)
        paragraphs.append(text)
        for meta in current_meta:
            meta['paragraph'] = idx
            provenance.append(meta)
        current.clear()
        current_meta.clear()

    for page_num in sorted(pages):
        page_lines = pages[page_num]
        typical_width, left_margin, same_line, pitch = _page_geometry(page_lines)
        prev_line: Optional[Dict[str, Any]] = None

        tops = [l.get('top', 0) or 0 for l in page_lines]
        page_top = min(tops) if tops else 0.0
        span = max(1.0, (max(tops) if tops else 0.0) - page_top)

        for line in page_lines:
            text = (line.get('text') or '').strip()
            if not text:
                continue
            if _is_bare_page_number(text):
                continue
            # Position matters at removal time too, not only at detection time.
            # Matching on text alone deletes body prose that happens to share a
            # running head's normalised form -- which digit masking makes much
            # likelier than it sounds, since it collapses every line differing
            # only by a number onto one key.
            if (_normalize_for_repetition(text) in running
                    and _in_header_band(line, page_top, span)):
                continue

            if _starts_new_paragraph(line, prev_line, typical_width, left_margin, same_line, pitch):
                flush()

            current.append(text)
            current_meta.append({
                'page': page_num,
                'top': round(line.get('top', 0) or 0, 2),
                'left': round(line.get('left', 0) or 0, 2),
                'width': round(line.get('width', 0) or 0, 2),
                'height': round(line.get('height', 0) or 0, 2),
                'confidence': line.get('ocr_confidence'),
                # How many extracted fragments this printed line was rebuilt
                # from -- 1 for a clean text layer, more where the extractor
                # split the line up.
                'fragments': len(line.get('parts', []) or [1]),
                'text': text,
            })
            prev_line = line

        # A paragraph may continue across a page break, so deliberately do NOT
        # flush here. prev_line carries over, and the next page's first line is
        # judged on its own indent and its predecessor's width exactly as any
        # other line would be.

    flush()

    text = '\n\n'.join(paragraphs)
    return unicodedata.normalize('NFC', text), provenance


def corpus_stats(text: str) -> Dict[str, Any]:
    """
    Mean characters per line, the measure tools/filter_literary_corpus.py uses
    to decide whether a file is OCR output or prose. Under ~55 it rejects the
    file outright, so this is the number reflow exists to move.
    """
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return {'lines': 0, 'chars': 0, 'chars_per_line': 0.0}
    chars = sum(len(l) for l in lines)
    return {
        'lines': len(lines),
        'chars': chars,
        'chars_per_line': round(chars / len(lines), 1),
    }
