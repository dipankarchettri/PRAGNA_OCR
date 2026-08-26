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
from pipeline.correction.ocr_repairs import apply_ocr_repairs
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

    def test_ocr_repairs(self):
        dict_words = get_dictionary()
        
        # Missing virama repairs
        cand, r_type = apply_ocr_repairs("ಶಿಕಷ", dict_words)
        self.assertEqual(cand, "ಶಿಕ್ಷ")
        self.assertEqual(r_type, "ocr_repair")
        
        cand2, r_type2 = apply_ocr_repairs("ಕನಾ೯ಟಕ", dict_words)
        self.assertEqual(cand2, "ಕರ್ನಾಟಕ")
        self.assertEqual(r_type2, "ocr_repair")
        
        # Vowel repairs
        cand3, r_type3 = apply_ocr_repairs("ಜಿವನ", dict_words)
        self.assertTrue(cand3.startswith("ಜೀವನ"))
        self.assertEqual(r_type3, "word_correction")



    def test_morphology_and_sandhi(self):
        dict_words = get_dictionary()
        
        # Test Sandhi join
        joined = join_root_suffix("ವಹಿಸು", "ುತ್ತದೆ")
        self.assertEqual(joined, "ವಹಿಸುತ್ತದೆ")

        joined2 = join_root_suffix("ಮನಸ್ಸು", "ಗೆ")
        self.assertEqual(joined2, "ಮನಸ್ಸಿಗೆ")

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


if __name__ == '__main__':
    unittest.main()
