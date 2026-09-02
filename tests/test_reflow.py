"""
Tests for the corpus reflow exporter.

Reflow is the step that turns page line-boxes into training text, so its
failure modes are corpus-corrupting rather than cosmetic: scrambled word order
reads as fluent-but-wrong Kannada, an over-eager paragraph break shatters
prose, and an under-eager one welds unrelated sections together. Each test
below pins a failure that actually occurred on real books from the scan dump.

Run: ./venv/bin/python tests/test_reflow.py
"""

import os
import sys
import unicodedata
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.exporter.reflow import (
    corpus_stats,
    find_running_lines,
    group_into_rows,
    reflow_lines,
)


def line(text, top, left=72.0, width=400.0, height=12.0, page=1, align='L', conf=None):
    return {'text': text, 'top': top, 'left': left, 'width': width, 'height': height,
            'page_num': page, 'alignment': align, 'ocr_confidence': conf}


class TestRowGrouping(unittest.TestCase):
    """
    Fragments of one printed line must come back in left-to-right order.

    On a skewed scan the fragments of a single line drift downward as they go
    right, so a plain (top, left) sort interleaves them with the next line's
    fragments and emits the words scrambled. Observed on a real book as
    "ಒಳಿತನ್ನು ಕಾಪಾಡುವವನಾಗಿ, ಮುಗಿಸಿ, ಜನತೆಯ ಕೆಲಸವನ್ನ ತನ್ನ ಸಾಹಸದಿಂದ ಮಾಡಿ".
    """

    def test_skewed_fragments_keep_reading_order(self):
        # One printed line, four fragments, each a little lower than the last.
        fragments = [
            line('ಒಂದು', top=25.2, left=47.6, width=78.0),
            line('ರೀತಿಯಲ್ಲಿ', top=26.5, left=152.3, width=87.0),
            line('ವೀರ', top=27.9, left=269.5, width=160.0),
            line('ಗೀತೆಗಳೂ', top=30.1, left=459.9, width=46.0),
        ]
        rows = group_into_rows(list(reversed(fragments)), same_line=7.2)
        self.assertEqual(len(rows), 1, 'one printed line became several rows')
        self.assertEqual(rows[0]['text'], 'ಒಂದು ರೀತಿಯಲ್ಲಿ ವೀರ ಗೀತೆಗಳೂ')

    def test_separate_printed_lines_stay_separate(self):
        rows = group_into_rows([
            line('ಮೊದಲ ಸಾಲು', top=25.0),
            line('ಎರಡನೇ ಸಾಲು', top=45.0),
        ], same_line=7.2)
        self.assertEqual([r['text'] for r in rows], ['ಮೊದಲ ಸಾಲು', 'ಎರಡನೇ ಸಾಲು'])

    def test_row_bbox_spans_all_its_fragments(self):
        rows = group_into_rows([
            line('ಎಡ', top=25.0, left=50.0, width=100.0),
            line('ಬಲ', top=26.0, left=300.0, width=80.0),
        ], same_line=7.2)
        self.assertEqual(rows[0]['left'], 50.0)
        self.assertEqual(rows[0]['width'], 330.0)   # 380 right edge - 50 left


class TestParagraphAssembly(unittest.TestCase):
    def test_wrapped_lines_join_into_one_paragraph(self):
        text, _ = reflow_lines([
            line('ಗಾಂಧೀಜಿ ಅವರು ಕರ್ನಾಟಕಕ್ಕೆ', top=100),
            line('ಹದಿನೆಂಟು ಬಾರಿ ಭೇಟಿ ನೀಡಿದರು.', top=120),
        ])
        self.assertEqual(text, 'ಗಾಂಧೀಜಿ ಅವರು ಕರ್ನಾಟಕಕ್ಕೆ ಹದಿನೆಂಟು ಬಾರಿ ಭೇಟಿ ನೀಡಿದರು.')

    def test_a_vertical_gap_starts_a_new_paragraph(self):
        """
        The signal that survives fragmented extraction. Line width cannot carry
        this on its own -- a text layer that splits lines into pieces makes
        every piece look short.
        """
        text, _ = reflow_lines([
            line('ಮೊದಲ ಪ್ಯಾರಾದ ಒಂದನೇ ಸಾಲು', top=100),
            line('ಮೊದಲ ಪ್ಯಾರಾದ ಎರಡನೇ ಸಾಲು', top=120),
            line('ಹೊಸ ಪ್ಯಾರಾ ಇಲ್ಲಿ ಶುರು', top=180),   # 3x the line pitch
        ])
        self.assertEqual(len(text.split('\n\n')), 2, f'wrong paragraph split: {text!r}')

    def test_hyphen_break_is_healed(self):
        text, _ = reflow_lines([
            line('ಮಹಾ-', top=100),
            line('ಕಾವ್ಯ', top=120),
        ])
        self.assertEqual(text, 'ಮಹಾಕಾವ್ಯ')

    def test_a_dash_that_is_punctuation_is_not_a_hyphen_break(self):
        """A trailing dash only joins when real word characters flank it."""
        text, _ = reflow_lines([
            line('ಅವನು —', top=100),
            line('ಬಂದನು', top=120),
        ])
        self.assertIn(' ', text, 'punctuation dash swallowed the following line')

    def test_output_is_nfc_normalised(self):
        text, _ = reflow_lines([line('ಕನ್ನಡ ಪುಸ್ತಕ', top=100)])
        self.assertEqual(text, unicodedata.normalize('NFC', text))


class TestNoiseRemoval(unittest.TestCase):
    def test_running_header_repeated_across_pages_is_dropped(self):
        """
        Running heads are ordinary high-confidence text, so the non-text
        confidence filter never sees them and no fixed pattern can match them
        across books. Positional repetition is the only signal.
        """
        lines = []
        for page in range(1, 7):
            lines.append(line('ಅಂಶುಮತಿ ಕಲ್ಯಾಣ', top=35, page=page))
            # A realistic body: the band check only discriminates on a page
            # with enough lines that "near the top" is a minority position.
            words = ['ಕಾವ್ಯ', 'ಸಾಹಿತ್ಯ', 'ಪುಸ್ತಕ', 'ಲೇಖಕ', 'ಕಥೆ',
                     'ಜನಪದ', 'ಸಂಸ್ಕೃತಿ', 'ಇತಿಹಾಸ', 'ಕವಿತೆ', 'ನಾಟಕ']
            for i, w in enumerate(words):
                lines.append(line(f'ಪುಟದ ನಿಜವಾದ ಪಠ್ಯ {w} ಇಲ್ಲಿದೆ',
                                  top=200 + i * 20, page=page))
        text, _ = reflow_lines(lines)
        self.assertNotIn('ಅಂಶುಮತಿ ಕಲ್ಯಾಣ', text)
        self.assertIn('ಪುಟದ ನಿಜವಾದ ಪಠ್ಯ', text)

    def test_a_repeated_mid_page_line_is_kept(self):
        """
        A refrain in a poem repeats too. What makes a running head removable is
        that it repeats *at the top or bottom of the page*, so the band check
        has to carry as much weight as the repetition count.
        """
        lines = []
        for page in range(1, 7):
            lines.append(line('ಶೀರ್ಷಿಕೆ ಸಾಲು', top=35, page=page))
            for i in range(5):
                lines.append(line(f'ಪದ್ಯದ ಸಾಲು {page} {i}', top=200 + i * 20, page=page))
            lines.append(line('ಪುನರಾವರ್ತಿತ ಪಲ್ಲವಿ', top=400, page=page))
            for i in range(5):
                lines.append(line(f'ಇನ್ನಷ್ಟು ಪದ್ಯ {page} {i}', top=440 + i * 20, page=page))
        text, _ = reflow_lines(lines)
        self.assertIn('ಪುನರಾವರ್ತಿತ ಪಲ್ಲವಿ', text, 'a mid-page refrain was removed')

    def test_bare_page_numbers_are_dropped(self):
        text, _ = reflow_lines([
            line('ನಿಜವಾದ ಪಠ್ಯ ಇಲ್ಲಿದೆ', top=100),
            line('೨೪', top=760),
        ])
        self.assertNotIn('೨೪', text)

    def test_non_text_lines_are_excluded(self):
        """OCR hallucinating on photos and banner art must not reach the corpus."""
        noise = line('ಗ ಟ ೪ ಒ', top=300, conf=8.0)
        noise['is_likely_non_text'] = True
        text, prov = reflow_lines([line('ನಿಜವಾದ ಪಠ್ಯ', top=100), noise])
        self.assertNotIn('ಗ ಟ ೪ ಒ', text)
        self.assertTrue(all('ಗ ಟ' not in p['text'] for p in prov))


class TestProvenance(unittest.TestCase):
    def test_every_surviving_line_is_traceable(self):
        """
        Reflow joins lines and drops headers, so without provenance there is no
        route back from a span of corpus text to the page it came from.
        """
        text, prov = reflow_lines([
            line('ಮೊದಲ ಸಾಲು', top=100, page=1),
            line('ಎರಡನೇ ಸಾಲು', top=120, page=1),
            line('ಬೇರೆ ಪುಟ', top=100, page=2),
        ])
        self.assertEqual(len(prov), 3)
        self.assertEqual({p['page'] for p in prov}, {1, 2})
        for rec in prov:
            self.assertIn('paragraph', rec)
            self.assertLess(rec['paragraph'], len(text.split('\n\n')))


class TestCorpusGate(unittest.TestCase):
    def test_reflow_lifts_chars_per_line_past_the_corpus_filter_cutoff(self):
        """
        tools/filter_literary_corpus.py rejects any file averaging under ~55
        chars/line as OCR output. The pipeline's verbatim export measures 48.4,
        so it emitted corpus text its own filter would discard. This is the
        number reflow exists to move.
        """
        lines = [line(f'ಇದು ಸಾಲು ಸಂಖ್ಯೆ {i} ಆಗಿದೆ ಮತ್ತು ಇದು ಪ್ಯಾರಾ', top=100 + i * 20)
                 for i in range(12)]
        verbatim = '\n'.join(l['text'] for l in lines)
        text, _ = reflow_lines(lines)

        self.assertLess(corpus_stats(verbatim)['chars_per_line'], 55)
        self.assertGreater(corpus_stats(text)['chars_per_line'], 55)


if __name__ == '__main__':
    unittest.main(verbosity=2)
