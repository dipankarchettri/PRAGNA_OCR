#!/usr/bin/env python3
"""
Unit and Integration Tests for Kannada OCR & Autocorrect Pipeline
"""

import os
import sys
import unittest

# Ensure pipeline is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline import init_pipeline
from pipeline.correction.tokenizer import tokenize, reconstruct
from pipeline.correction.ocr_repairs import normalize_script, clean_unicode_glitches, normalize_indic_repha
from pipeline.correction.morphology import decompose_word, join_root_suffix
from pipeline.correction.dictionary import get_dictionary
from pipeline.correction.edit_distance import weighted_edit_distance
from pipeline.correction.corrector import correct_text, suggest_kannada_word
from pipeline.exporter.pdf_generator import generate_pdf_from_text


class TestKannadaPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load dictionary AND the real n-gram corpus model, matching the
        # exact initialization every real entry point (process_text_input,
        # process_document) goes through. Loading the dictionary alone left
        # candidate-ranking behavior that depends on real corpus frequency
        # (see corrector.MIN_CORPUS_ATTESTATION / KEEP_ORIGINAL_BASE_COST)
        # completely untested, since score_candidate() has no differentiating
        # signal at all against an untrained model -- a gap that let a real
        # ranking regression pass silently.
        init_pipeline()

    def test_tokenization(self):
        text = "ಕನ್ನಡ 123 English ! ಶಾಂತಿ ."
        tokens = tokenize(text)
        self.assertEqual(reconstruct(tokens), text)
        self.assertEqual(tokens[0]['type'], 'kannada')
        self.assertEqual(tokens[0]['value'], 'ಕನ್ನಡ')
        kannada_tokens = [t for t in tokens if t['type'] == 'kannada']
        self.assertEqual(len(kannada_tokens), 2)

    def test_script_normalization(self):
        # Universal Repha normalizations
        self.assertEqual(normalize_indic_repha("ಕನಾ೯ಟಕ"), "ಕರ್ನಾಟಕ")
        self.assertEqual(normalize_indic_repha("ಕತ೯ವ್ಯ"), "ಕರ್ತವ್ಯ")
        self.assertEqual(normalize_indic_repha("ಮಾ೯"), "ರ್ಮಾ")
        
        # Zero-digit anusvara cleanup
        self.assertEqual(clean_unicode_glitches("ಗ್ರಹಿಸಿಕೊ೦ಂಡೇ"), "ಗ್ರಹಿಸಿಕೊಂಡೇ")

    def test_dynamic_ocr_repairs(self):
        # Optical glyph repairs (Blue)
        cand1, dist1, type1 = suggest_kannada_word("ಕಾಥಿ")
        self.assertEqual(cand1, "ಕಾಫಿ")
        self.assertEqual(type1, "ocr_repair")

        cand2, dist2, type2 = suggest_kannada_word("ದೂಹಿಸು")
        self.assertEqual(cand2, "ದೂಷಿಸು")
        self.assertEqual(type2, "ocr_repair")

        cand3, dist3, type3 = suggest_kannada_word("ಯೋಪನೆ")
        self.assertEqual(cand3, "ಯೋಜನೆ")
        self.assertEqual(type3, "ocr_repair")

        cand4, dist4, type4 = suggest_kannada_word("ಕನಾ೯ಟಕ")
        self.assertEqual(cand4, "ಕರ್ನಾಟಕ")
        self.assertEqual(type4, "ocr_repair")

        # Systematic vowel length repairs (Green)
        cand5, dist5, type5 = suggest_kannada_word("ಜಿವನ")
        self.assertEqual(cand5, "ಜೀವನ")
        self.assertEqual(type5, "word_correction")

        cand6, dist6, type6 = suggest_kannada_word("ಸಂಗಿತ")
        self.assertEqual(cand6, "ಸಂಗೀತ")
        self.assertEqual(type6, "word_correction")

    def test_morphology_and_sandhi(self):
        # Test Sandhi join
        joined = join_root_suffix("ವಹಿಸು", "ುತ್ತದೆ")
        self.assertEqual(joined, "ವಹಿಸುತ್ತದೆ")

        joined2 = join_root_suffix("ಮನಸ್ಸು", "ಗೆ")
        self.assertEqual(joined2, "ಮನಸ್ಸಿಗೆ")

        joined3 = join_root_suffix("ಮಾಡಿಕೊಡು", "ತ್ತಾರೆ")
        self.assertEqual(joined3, "ಮಾಡಿಕೊಡುತ್ತಾರೆ")

    def test_weighted_edit_distance(self):
        # Confused characters should have lower cost than unrelated characters
        dist_similar = weighted_edit_distance("ಕ", "ಖ")
        dist_unrelated = weighted_edit_distance("ಕ", "ಮ")
        self.assertLess(dist_similar, dist_unrelated)

    def test_end_to_end_text_correction(self):
        input_text = "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತದೆ."
        res = correct_text(input_text)
        
        self.assertTrue(res['has_errors'])
        self.assertIn("ಜೀವನದಲ್ಲಿ", res['corrected'])
        self.assertIn("ವಹಿಸುತ್ತದೆ", res['corrected'])
        self.assertGreater(len(res['corrections']), 0)

    def test_pdf_generation(self):
        test_pdf_out = os.path.join(BASE_DIR, 'web', 'processed', 'test_output.pdf')
        corrected_text = "ಕನ್ನಡ ಸ್ವಯಂ ತಿದ್ದುಪಡಿ ವ್ಯವಸ್ಥೆ.\nಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜೀವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತ್ತದೆ."
        
        out = generate_pdf_from_text(corrected_text, test_pdf_out)
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 500)


class TestSuffixBucketing(unittest.TestCase):
    """
    decompose_word finds the longest suffix a word ends with. It used to scan
    all 152 entries of SUFFIXES calling str.endswith on each; it now looks up
    only the bucket for the word's final character, which was 38% of total
    correction runtime.

    That is a pure speedup *only* while the bucket yields exactly the same
    matches in exactly the same order as the full scan -- the first match wins,
    so a reordering silently changes which decomposition the engine picks.
    Nothing else in the suite would catch that, and the invariant is easy to
    break by editing SUFFIXES.
    """

    def test_bucket_lookup_matches_a_full_scan(self):
        from pipeline.correction.morphology import SUFFIXES, suffixes_ending_with

        # Every distinct final character in the vocabulary, not just the ones
        # a sample of words happens to end in.
        for ch in sorted({s[-1] for s in SUFFIXES} | set('ಕಮತಿೆುಂ್'), key=ord):
            with self.subTest(char=ch):
                self.assertEqual(
                    [s for s in suffixes_ending_with(ch)],
                    [s for s in SUFFIXES if s.endswith(ch)],
                    'bucket order diverged from the full-scan order')

    def test_buckets_cover_every_suffix(self):
        from pipeline.correction.morphology import SUFFIXES, suffixes_ending_with

        reachable = set()
        for ch in {s[-1] for s in SUFFIXES}:
            reachable.update(suffixes_ending_with(ch))
        self.assertEqual(reachable, set(SUFFIXES), 'a suffix is unreachable')

    def test_unknown_final_character_yields_nothing(self):
        from pipeline.correction.morphology import suffixes_ending_with

        self.assertEqual(list(suffixes_ending_with('Z')), [])


class TestCandidateCacheInvalidation(unittest.TestCase):
    """
    generate_kannada_candidates memoizes on the word alone, which is only
    sound while the dictionary behind it is unchanged. If a reload ever stops
    clearing the cache, corrections silently reflect the *previous*
    vocabulary -- a failure that would produce plausible-looking wrong output
    rather than an error.
    """

    def test_clearing_the_cache_makes_a_vocabulary_change_visible(self):
        init_pipeline()
        from pipeline.correction import corrector
        from pipeline.correction.dictionary import get_dictionary

        word = 'ಜಿವನದಲ್ಲಿ'   # a real OCR error the engine corrects
        dictionary = get_dictionary()
        first = corrector.generate_kannada_candidates(word, dictionary)
        self.assertIn(word, corrector._candidate_cache,
                      'candidate generation was not memoized at all')

        corrector.clear_correction_caches()
        self.assertNotIn(word, corrector._candidate_cache,
                         'clear_correction_caches did not drop the entry')

        # Recomputing from scratch must agree with the cached answer.
        self.assertEqual(corrector.generate_kannada_candidates(word, dictionary), first)


class TestOCRLayoutGrouping(unittest.TestCase):
    """
    Regression tests for how word boxes are grouped into lines.

    Tesseract's box hierarchy is page > block > paragraph > line > word, and line_num
    restarts at 1 inside every paragraph. Grouping on (block_num, line_num) alone
    merges the Nth line of every paragraph in a block into a single line, splicing
    together text from unrelated parts of the page.
    """

    def _fake_tsv(self, rows):
        """Build a pytesseract image_to_data DICT from (block, par, line, text) rows."""
        data = {k: [] for k in ('block_num', 'par_num', 'line_num', 'left', 'top',
                                'width', 'height', 'conf', 'text')}
        for idx, (block, par, line, text) in enumerate(rows):
            data['block_num'].append(block)
            data['par_num'].append(par)
            data['line_num'].append(line)
            data['left'].append(10)
            # Give each paragraph a distinct vertical band so ordering is well defined.
            data['top'].append(par * 1000 + line * 40)
            data['width'].append(100)
            data['height'].append(30)
            data['conf'].append(90)
            data['text'].append(text)
        return data

    def _run(self, rows):
        from PIL import Image
        import pipeline.ocr.tesseract_engine as engine

        fake = self._fake_tsv(rows)

        class _FakeOutput:
            DICT = 'dict'

        real_pt, real_avail = engine.pytesseract, engine.is_tesseract_available
        try:
            engine.is_tesseract_available = lambda: True
            engine.pytesseract = type('P', (), {
                'image_to_data': staticmethod(lambda *a, **k: fake),
                'Output': _FakeOutput,
            })
            img = Image.new('RGB', (800, 4000), 'white')
            return engine.ocr_image_with_layout(img, lang='kan')
        finally:
            engine.pytesseract, engine.is_tesseract_available = real_pt, real_avail

    def test_paragraphs_do_not_merge_into_one_line(self):
        # Two paragraphs in the same block, each with a line numbered 1 and 2.
        lines = self._run([
            (1, 1, 1, 'ALPHA'),
            (1, 1, 2, 'BETA'),
            (1, 2, 1, 'GAMMA'),
            (1, 2, 2, 'DELTA'),
        ])
        self.assertEqual(len(lines), 4, "each paragraph line must stay separate")
        for line in lines:
            self.assertEqual(len(line['text'].split()), 1,
                             f"unrelated paragraphs merged into one line: {line['text']!r}")

    def test_words_on_same_line_are_joined(self):
        lines = self._run([
            (1, 1, 1, 'ONE'),
            (1, 1, 1, 'TWO'),
        ])
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]['text'], 'ONE TWO')

    def test_line_confidence_is_reported(self):
        lines = self._run([(1, 1, 1, 'ONE')])
        self.assertIn('conf', lines[0])
        self.assertAlmostEqual(lines[0]['conf'], 90.0)


class TestResolutionNormalization(unittest.TestCase):
    def test_low_resolution_page_is_upscaled(self):
        from PIL import Image
        from pipeline.ingestion import normalize_resolution
        out = normalize_resolution(Image.new('RGB', (642, 912)))
        self.assertGreater(max(out.size), 2000, "low-DPI page should be scaled up")

    def test_aspect_ratio_is_preserved(self):
        from PIL import Image
        from pipeline.ingestion import normalize_resolution
        src = Image.new('RGB', (642, 912))
        out = normalize_resolution(src)
        self.assertAlmostEqual(src.size[0] / src.size[1], out.size[0] / out.size[1], places=2)

    def test_high_resolution_page_is_untouched(self):
        from PIL import Image
        from pipeline.ingestion import normalize_resolution
        src = Image.new('RGB', (2480, 3508))
        self.assertIs(normalize_resolution(src), src, "already-high-DPI page must not be resized")

    def test_upscale_is_capped(self):
        from PIL import Image
        from pipeline.ingestion import normalize_resolution
        from pipeline.ingestion.image_processor import MAX_UPSCALE_FACTOR
        src = Image.new('RGB', (100, 140))
        out = normalize_resolution(src)
        self.assertLessEqual(out.size[0] / src.size[0], MAX_UPSCALE_FACTOR + 0.01)


if __name__ == '__main__':
    unittest.main()

