#!/usr/bin/env python3
"""
Tests for the pluggable correction engine and the Sarvam-1 corrector.

Split in two: everything that can be checked without the model (dispatch,
offset bookkeeping, the guards that stop a generative rewrite from entering the
corpus) runs always; anything that needs 4 GB of weights runs only when
SARVAM_TEST_MODEL=1 is set, so the normal test run stays fast and dependency-free.

    ./venv/bin/python tests/test_sarvam.py
    SARVAM_TEST_MODEL=1 ./venv/bin/python tests/test_sarvam.py
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline import init_pipeline, process_text_input
from pipeline.correction import ENGINES, correct_layout_lines_with, correct_text_with
from pipeline.correction.corrector import correct_text
from pipeline.correction import sarvam_corrector as sc

RUN_MODEL_TESTS = os.environ.get('SARVAM_TEST_MODEL') == '1'
SAMPLE = "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತದೆ"


class TestEngineDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_pipeline()

    def test_rule_engine_routes_unchanged(self):
        """The dispatcher must be a pass-through for the default engine."""
        direct = correct_text(SAMPLE)
        routed = correct_text_with(SAMPLE, engine='rule')
        self.assertEqual(direct['corrected'], routed['corrected'])
        self.assertEqual(len(direct['corrections']), len(routed['corrections']))

    def test_unknown_engine_rejected(self):
        with self.assertRaises(ValueError):
            correct_text_with(SAMPLE, engine='gpt-9')
        with self.assertRaises(ValueError):
            process_text_input(SAMPLE, engine='not-an-engine')

    def test_engine_list_is_complete(self):
        self.assertIn('rule', ENGINES)
        for name in ('sarvam-rerank', 'sarvam-generate', 'hybrid'):
            self.assertIn(name, ENGINES)

    def test_layout_shape_matches_rule_engine(self):
        """
        Both paths must emit the same keys, or export/diff-colouring silently
        loses fields depending on which engine ran.
        """
        lines = [{'text': SAMPLE, 'conf': 80.0, 'alignment': 'L',
                  'top': 10, 'left': 5, 'width': 100, 'height': 12, 'page_num': 1}]
        rule_lines, _ = correct_layout_lines_with(lines, engine='rule')
        from pipeline.correction.sarvam_corrector import correct_layout_lines_sarvam
        # Exercise the shape without the model by stubbing the per-line call.
        original = sc.correct_text_sarvam
        sc.correct_text_sarvam = lambda text, mode=None, word_confidences=None: {
            'original': text, 'corrected': text, 'has_errors': False,
            'total_words': 0, 'total_corrections': 0, 'accuracy_rate': 100.0,
            'corrections': []
        }
        try:
            sarvam_lines, _ = correct_layout_lines_sarvam(lines)
        finally:
            sc.correct_text_sarvam = original
        self.assertEqual(set(rule_lines[0]), set(sarvam_lines[0]))

    def test_non_text_lines_skipped(self):
        """A line below NON_TEXT_LINE_CONFIDENCE must never reach the LM."""
        from pipeline.correction.sarvam_corrector import correct_layout_lines_sarvam
        called = []
        original = sc.correct_text_sarvam
        sc.correct_text_sarvam = lambda *a, **k: called.append(a) or {
            'original': '', 'corrected': '', 'has_errors': False, 'total_words': 0,
            'total_corrections': 0, 'accuracy_rate': 100.0, 'corrections': []
        }
        try:
            lines, _ = correct_layout_lines_sarvam([{'text': SAMPLE, 'conf': 9.0}])
        finally:
            sc.correct_text_sarvam = original
        self.assertEqual(called, [])
        self.assertTrue(lines[0]['is_likely_non_text'])
        self.assertEqual(lines[0]['text'], SAMPLE)


class TestGenerationGuards(unittest.TestCase):
    """
    These guards are the only thing standing between a base model's
    free-running completion and the training corpus, so they are tested
    independently of whether the model is available.
    """

    def test_rejects_empty_and_non_kannada(self):
        self.assertFalse(sc._generation_is_plausible(SAMPLE, '')[0])
        self.assertFalse(sc._generation_is_plausible(SAMPLE, 'The capital of Karnataka is')[0])

    def test_rejects_truncation_and_runaway(self):
        self.assertFalse(sc._generation_is_plausible(SAMPLE, SAMPLE[:10])[0])
        self.assertFalse(sc._generation_is_plausible(SAMPLE, SAMPLE + ' ' + SAMPLE)[0])

    def test_rejects_paraphrase_but_accepts_repair(self):
        repaired = SAMPLE.replace('ಜಿವನದಲ್ಲಿ', 'ಜೀವನದಲ್ಲಿ').replace('ವಹಿಸುತದೆ', 'ವಹಿಸುತ್ತದೆ')
        ok, reason = sc._generation_is_plausible(SAMPLE, repaired)
        self.assertTrue(ok, reason)

        unrelated = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು ಆಗಿದೆ ಎಂದು ಎಲ್ಲರಿಗೂ ಗೊತ್ತಿದೆ ಅಲ್ಲವೇ"
        self.assertFalse(sc._generation_is_plausible(SAMPLE, unrelated)[0])

    def test_diff_offsets_point_at_the_original_word(self):
        repaired = SAMPLE.replace('ಜಿವನದಲ್ಲಿ', 'ಜೀವನದಲ್ಲಿ')
        diffs = sc.diff_corrections(SAMPLE, repaired)
        self.assertEqual(len(diffs), 1)
        d = diffs[0]
        self.assertEqual(SAMPLE[d['start']:d['end']], d['original'])
        self.assertEqual(d['correction'], 'ಜೀವನದಲ್ಲಿ')

    def test_prompt_ends_open_for_completion(self):
        prompt = sc.build_prompt(SAMPLE, examples=[('a', 'b')])
        self.assertTrue(prompt.endswith(f"{sc.CLEAN_LABEL}:"))
        self.assertIn(SAMPLE, prompt)


class TestContextWindow(unittest.TestCase):
    def test_window_is_centred_and_reconstructs(self):
        from pipeline.correction.tokenizer import tokenize
        # Distinct all-Kannada words: a digit suffix would tokenize separately.
        words = [c + 'ಮನ' for c in 'ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷ']
        text = ' '.join(words)
        tokens = tokenize(text)
        target = next(i for i, t in enumerate(tokens) if t['value'] == words[15])
        variants = sc._window_variants(tokens, target, [words[15], 'ಬದಲಿ'])
        self.assertIn(words[15], variants[0])
        self.assertIn('ಬದಲಿ', variants[1])
        self.assertNotIn(words[15], variants[1])
        # Window is bounded, not the whole line.
        self.assertLess(len(variants[0]), len(text))
        self.assertLessEqual(
            len(variants[0].split()), 2 * sc.LM_CONTEXT_WORDS + 1
        )


class TestVllmBackend(unittest.TestCase):
    """
    Response parsing for the vLLM backend, checked without a server.

    Worth testing directly: the scoring path depends on two details of the
    OpenAI completions response that are easy to get wrong and would silently
    corrupt every margin comparison rather than raise -- the first prompt token
    comes back as a null logprob, and choices are not guaranteed to arrive in
    request order.
    """

    def _stub(self, payload):
        from pipeline.correction import sarvam_vllm
        original_post, original_name = sarvam_vllm._post, sarvam_vllm.model_name
        sarvam_vllm._post = lambda path, body: payload
        sarvam_vllm.model_name = lambda: 'stub'
        self.addCleanup(setattr, sarvam_vllm, '_post', original_post)
        self.addCleanup(setattr, sarvam_vllm, 'model_name', original_name)
        return sarvam_vllm

    def test_skips_null_first_token_logprob(self):
        lm = self._stub({'choices': [
            {'index': 0, 'logprobs': {'token_logprobs': [None, -1.0, -2.0]}},
        ]})
        (total, count), = lm.score_sequences(['x'])
        self.assertAlmostEqual(total, -3.0)
        self.assertEqual(count, 2)

    def test_results_follow_choice_index_not_arrival_order(self):
        lm = self._stub({'choices': [
            {'index': 1, 'logprobs': {'token_logprobs': [None, -9.0]}},
            {'index': 0, 'logprobs': {'token_logprobs': [None, -1.0]}},
        ]})
        scores = lm.log_probs(['first', 'second'])
        self.assertEqual(scores, [-1.0, -9.0])

    def test_unreachable_server_is_not_available(self):
        from pipeline.correction import sarvam_vllm
        original = sarvam_vllm.VLLM_URL
        sarvam_vllm.VLLM_URL = 'http://127.0.0.1:9'  # discard port
        sarvam_vllm._model_name = None
        self.addCleanup(setattr, sarvam_vllm, 'VLLM_URL', original)
        self.assertFalse(sarvam_vllm.is_available())


@unittest.skipUnless(RUN_MODEL_TESTS, "set SARVAM_TEST_MODEL=1 to run tests that load Sarvam-1")
class TestSarvamModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_pipeline()
        from pipeline.correction import sarvam_lm
        sarvam_lm.load()
        cls.lm = sarvam_lm

    def test_scoring_prefers_real_kannada(self):
        """Sanity check that the LM is loaded and discriminating at all."""
        good = "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜೀವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತ್ತದೆ"
        garbled = "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತದೆ"
        good_lp, garbled_lp = self.lm.log_probs([good, garbled])
        self.assertGreater(good_lp, garbled_lp)

    def test_rerank_never_invents_vocabulary(self):
        """
        Every word the rerank engine writes must have come from the rule
        engine's candidate generator -- that containment is the whole safety
        argument for the mode.
        """
        from pipeline.correction.corrector import generate_kannada_candidates
        from pipeline.correction.dictionary import get_dictionary
        res = correct_text_with(SAMPLE, engine='sarvam-rerank')
        dictionary = get_dictionary()
        for corr in res['corrections']:
            if corr.get('engine') != 'sarvam-rerank':
                continue
            proposed = {c for c, _s, _t in generate_kannada_candidates(corr['original'], dictionary)}
            self.assertIn(corr['correction'], proposed)

    def test_hybrid_is_a_subset_of_rule(self):
        rule = correct_text_with(SAMPLE, engine='rule')
        hybrid = correct_text_with(SAMPLE, engine='hybrid')
        rule_pairs = {(c['original'], c['correction']) for c in rule['corrections']}
        hybrid_pairs = {(c['original'], c['correction']) for c in hybrid['corrections']}
        self.assertTrue(hybrid_pairs.issubset(rule_pairs))

    def test_generate_returns_kannada_or_falls_back(self):
        res = correct_text_with(SAMPLE, engine='sarvam-generate')
        self.assertTrue(res['corrected'].strip())
        # Guarded: output is either a repair of the input or the input itself.
        ok, _ = sc._generation_is_plausible(SAMPLE, res['corrected'])
        self.assertTrue(ok or res['corrected'] == SAMPLE)


if __name__ == '__main__':
    unittest.main(verbosity=2)
