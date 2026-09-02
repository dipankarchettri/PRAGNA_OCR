"""
Precision regression tests: words the corrector must NOT change.

Over-correction is this engine's documented dominant failure mode, and for a
pipeline whose output is LLM training data it is the expensive one -- a
confidently wrong "fix" corrupts the corpus silently, where an untouched OCR
error at least stays visible downstream. Until now every assertion in the suite
checked that a *broken* word gets repaired; nothing checked that a *correct*
word is left alone.

Every case below is a real regression this project already hit once. The
constants in corrector.py were calibrated against them and the reasoning was
written into code comments, but none of it became an executable assertion, so
nothing stopped the same class of bug returning. That is how the space-merge
bug below survived: it broke 50 words against 2 genuine fixes across 24 clean
pages and no test noticed.

Run: ./venv/bin/python tests/test_correction_precision.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import init_pipeline
from pipeline.correction import correct_text
from pipeline.correction.corrector import suggest_kannada_word, heal_split_tokens
from pipeline.correction.dictionary import get_dictionary
from pipeline.correction.tokenizer import tokenize
from pipeline.correction.ocr_repairs import normalize_script


class PrecisionTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_pipeline()
        cls.dictionary = get_dictionary()


class TestWordsMustNotChange(PrecisionTestBase):
    """
    Single words that were each wrongly "corrected" at some point in this
    project's history. Sourced from the calibration notes on
    HIGH_OCR_CONFIDENCE_TRUST (corrector.py) and FREQUENCY_DOMINANCE_RATIO.
    """

    # From HIGH_OCR_CONFIDENCE_TRUST's docstring: every one of these was
    # already correct and got overwritten anyway.
    ALREADY_CORRECT = [
        'ಭಾವಃ', 'ಇಮೇ', 'ದೇಹಾ', 'ಅಜೋ', 'ಯಣ', 'ಅನುಸರಣವು', 'ಹಣ್ಣು', 'ಮೇಲಿ',
    ]

    def test_ocr_path_leaves_them_alone_when_tesseract_was_confident(self):
        """
        The document pipeline's real path. HIGH_OCR_CONFIDENCE_TRUST is what
        actually protects these words: all 8 were read at confidence 90-96 in
        the sample they were calibrated from.
        """
        from pipeline.correction import correct_layout_lines

        text = ' '.join(self.ALREADY_CORRECT)
        line = {'text': text, 'conf': 93.0,
                'word_confidences': [(w, 93.0) for w in self.ALREADY_CORRECT]}
        corrected, corrections = correct_layout_lines([line])
        self.assertEqual(corrected[0]['text'], text)
        self.assertEqual(corrections, [])

    @unittest.expectedFailure
    def test_KNOWN_GAP_no_confidence_path_still_corrupts_some(self):
        """
        Documents a real, currently-unfixed weakness rather than hiding it.

        HIGH_OCR_CONFIDENCE_TRUST can only fire where per-word OCR confidence
        exists. It does not on the text paths -- process_text_input, the web
        live editor, CLI --text -- so there these words are protected only by
        the corpus-evidence gates, and those are not sufficient:

            in sentence context : ಯಣ -> ಹಣ                    (1 of 8)
            as bare single words: ಭಾವಃ ಇಮೇ ದೇಹಾ ಅಜೋ ಯಣ ಮೇಲಿ   (6 of 8)

        Closing this needs a confidence-independent way to trust a word the
        dictionary does not contain -- which is what the frequency-weighted
        target set is for. When that lands, delete the expectedFailure marker;
        the test turning green is the signal it worked.
        """
        changed = []
        for word in self.ALREADY_CORRECT:
            result, _dist, corr_type = suggest_kannada_word(word)
            if result != word:
                changed.append(f"{word} -> {result} ({corr_type})")
        self.assertEqual(
            changed, [],
            "words that are already correct were overwritten:\n  " + "\n  ".join(changed))

    def test_rare_real_spelling_survives_a_commoner_variant(self):
        """
        ಮಂಡಲಿಗೆ is the real spelling in an actual exam-board document (corpus
        freq 198) and must survive even though ಮಂಡಳಿಗೆ is 89x more common.
        This is the 'keep' side of FREQUENCY_DOMINANCE_RATIO's calibration;
        raising that ratio's tolerance too far breaks this case first.
        """
        result, _dist, _type = suggest_kannada_word('ಮಂಡಲಿಗೆ')
        self.assertEqual(result, 'ಮಂಡಲಿಗೆ')


class TestSpaceHealingPrecision(PrecisionTestBase):
    """
    heal_split_tokens merges a word split across a space. The failure mode is
    the reverse: destroying a real word boundary between two correct words.
    That is worse than leaving a split alone, because a split word is visibly
    wrong downstream while a merged pair reads as one plausible token.
    """

    def _heal(self, text: str):
        tokens, corrections = heal_split_tokens(tokenize(text), self.dictionary)
        return ''.join(t['value'] for t in tokens), corrections

    def test_two_independently_valid_words_are_not_merged(self):
        """
        Measured on tests/fixtures/eval. Both pairs were merged because the
        guard tested bare `in dictionary` membership, which misses inflected
        forms the .dic does not list literally -- ಪರಿಚಯವು is ಪರಿಚಯ + ವು.
        """
        for phrase in ('ಪರಿಚಯವು ಅವರು', 'ಸ್ಪರ್ಶಿಸಿ ಅವರ'):
            with self.subTest(phrase=phrase):
                healed, corrections = self._heal(phrase)
                self.assertEqual(healed, phrase,
                                 f"word boundary destroyed: {phrase!r} -> {healed!r}")
                self.assertEqual(corrections, [])

    def test_genuine_splits_are_still_healed(self):
        """
        The other side of the same guard: a real OCR split leaves a *fragment*
        as the second half, and those must still be rejoined. If this fails,
        the precision fix above was applied too aggressively.
        """
        for phrase, expected in (('ಹಿನ್ನೆ ಲೆಯಲ್ಲಿ', 'ಹಿನ್ನೆಲೆಯಲ್ಲಿ'),
                                 ('ಸಾಧ್ಯ ವಾಗಬೇಕಿದೆ', 'ಸಾಧ್ಯವಾಗಬೇಕಿದೆ')):
            with self.subTest(phrase=phrase):
                healed, _ = self._heal(phrase)
                self.assertEqual(healed, expected)


class TestCompoundValidation(PrecisionTestBase):
    """
    is_compound_word accepts a word if it splits into two dictionary words. That
    test gets weaker as the dictionary grows, because more short pieces are
    listed -- and it is the LAST line of defence in resolve_valid_surface_form,
    so anything it waves through becomes a correction target that skips the
    bigram gate.
    """

    def test_a_word_glued_to_a_bound_morpheme_is_not_a_compound(self):
        """
        Found on tests/fixtures/real/03.png. The OCR error ನಿಪಾಸೆ (for ಪಿಪಾಸೆ)
        was "corrected" to ನಿವಾಸೆ -- not a word, not in the dictionary, and
        rejected by both morphology paths. It passed only because it splits as
        ನಿ + ವಾಸೆ, and ನಿ is a one-akshara particle that happens to be listed.
        """
        from pipeline.correction.morphology import is_compound_word
        from pipeline.correction.corrector import resolve_valid_surface_form

        self.assertFalse(is_compound_word('ನಿವಾಸೆ', self.dictionary))
        self.assertIsNone(resolve_valid_surface_form('ನಿವಾಸೆ', self.dictionary)[0],
                          'a non-word was validated as a correction target')

    def test_real_compounds_still_validate(self):
        """The guard must not disable genuine compound recognition."""
        from pipeline.correction.morphology import is_compound_word

        for word in ('ಮಹಾಕಾವ್ಯ', 'ರಾಷ್ಟ್ರಗೀತೆ', 'ಜನಪದಗೀತೆ'):
            with self.subTest(word=word):
                self.assertTrue(is_compound_word(word, self.dictionary))


class TestNonKannadaIsUntouched(PrecisionTestBase):
    """
    The tokenizer's contract: non-Kannada spans are never modified and
    reconstruct() rebuilds the original around them losslessly. A break here
    corrupts every English name, number and citation in the corpus.
    """

    def test_latin_numbers_and_punctuation_survive_verbatim(self):
        text = 'ಶಿಕ್ಷಣವು SHATHAMANADA GANA 2011, J.C.Road - 560 002 ಪ್ರಮುಖ'
        result = correct_text(text)['corrected']
        for fragment in ('SHATHAMANADA', 'GANA', '2011', 'J.C.Road', '560', '002'):
            self.assertIn(fragment, result, f"{fragment!r} was altered or dropped")

    def test_correct_sentence_is_returned_unchanged(self):
        """A sentence with no errors must come back byte-identical."""
        text = 'ಈ ಯೋಜನೆ ಸಾರ್ಥಕವಾಗುತ್ತದೆ ಎಂದು ಭಾವಿಸುತ್ತೇನೆ.'
        result = correct_text(text)
        self.assertEqual(result['corrected'], text)
        self.assertEqual(result['corrections'], [])


class TestNonTextLineFiltering(PrecisionTestBase):
    """
    Ported from the deleted tests/test_sarvam.py, which was the only place
    NON_TEXT_LINE_CONFIDENCE was exercised. Lines below the confidence floor
    are Tesseract hallucinating text out of photos or banner art; they must be
    flagged so pipeline/__init__.py can keep them out of the corpus text.
    """

    def test_low_confidence_lines_are_flagged_as_non_text(self):
        from pipeline.correction import correct_layout_lines

        lines = [
            {'text': 'ಈ ಯೋಜನೆ ಸಾರ್ಥಕವಾಗುತ್ತದೆ', 'conf': 92.0},
            {'text': 'ಗ ಟ ೪ ಒ', 'conf': 8.0},
        ]
        corrected, _ = correct_layout_lines(lines)
        self.assertFalse(corrected[0].get('is_likely_non_text'),
                         'high-confidence body text was flagged as non-text')
        self.assertTrue(corrected[1].get('is_likely_non_text'),
                        'single-digit-confidence graphic noise was not flagged')


class TestFinalJoinerNormalization(unittest.TestCase):
    """
    Word-final ZWNJ/ZWJ after a virama is free orthographic variation in written
    Kannada -- the corpus writes ಯಾದವ್ and ಯಾದವ್<ZWNJ> both, at a 3:1 ratio --
    and it renders identically either way. Normalizing it keeps one word from
    tokenizing as two. A joiner BETWEEN consonants is not free variation: it
    forces the half-form over the conjunct ligature, and must survive.
    """

    ZWNJ = '‌'
    ZWJ = '‍'

    def test_word_final_joiner_is_dropped(self):
        for bare in ('ಇಂಗ್ಲಿಷ್', 'ಕಾಮತ್', 'ಶಿವರಾವ್', 'ಯಾದವ್'):
            for joiner in (self.ZWNJ, self.ZWJ):
                self.assertEqual(normalize_script(bare + joiner), bare)

    def test_joiner_before_punctuation_is_dropped(self):
        for tail in ('.', ',', ' ಮತ್ತು', '-ಅಲಕ್'):
            self.assertEqual(
                normalize_script('ಕಾಮತ್' + self.ZWNJ + tail),
                'ಕಾಮತ್' + tail,
            )

    def test_medial_joiner_between_consonants_survives(self):
        for word in ('ಕ್' + self.ZWNJ + 'ವ', 'ಏಕ್' + self.ZWNJ + 'ನಿರಂಜನ'):
            self.assertEqual(normalize_script(word), word)

    def test_joiner_not_after_virama_survives(self):
        word = 'ಕ' + self.ZWNJ
        self.assertEqual(normalize_script(word), word)

    def test_clean_text_is_untouched(self):
        for word in ('ಇಂಗ್ಲಿಷ್', 'ಶಿಕ್ಷಣವು', 'ಕಾಫಿ'):
            self.assertEqual(normalize_script(word), word)


if __name__ == '__main__':
    unittest.main(verbosity=2)
