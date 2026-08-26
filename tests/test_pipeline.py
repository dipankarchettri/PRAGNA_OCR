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

from pipeline.correction.tokenizer import tokenize, reconstruct
from pipeline.correction.ocr_repairs import normalize_script, clean_unicode_glitches, normalize_indic_repha
from pipeline.correction.morphology import decompose_word, join_root_suffix
from pipeline.correction.dictionary import load_dictionary, get_dictionary
from pipeline.correction.edit_distance import weighted_edit_distance
from pipeline.correction.corrector import correct_text, suggest_kannada_word
from pipeline.exporter.pdf_generator import generate_pdf_from_text


class TestKannadaPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load dictionary
        dic_path = os.path.join(BASE_DIR, 'data', 'kn_IN.dic')
        load_dictionary(dic_path)

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

    def test_process_text_input_modes(self):
        from pipeline import process_text_input
        # Test algorithmic mode
        res_algo = process_text_input("ಶಿಕ್ಷಣವು ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ", engine_mode='algo')
        self.assertIn("ಜೀವನದಲ್ಲಿ", res_algo['corrected'])
        self.assertEqual(res_algo['engine_mode'], 'algorithmic')

        # Test hybrid mode fallback resilience
        res_hybrid = process_text_input("ಶಿಕ್ಷಣವು ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ", engine_mode='hybrid')
        self.assertIn("ಜೀವನದಲ್ಲಿ", res_hybrid['corrected'])


if __name__ == '__main__':
    unittest.main()

