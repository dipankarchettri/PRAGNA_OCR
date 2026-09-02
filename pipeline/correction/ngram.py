"""
N-Gram Language Model for Kannada Context Ranking
Scores alternative candidate words using real corpus unigram/bigram
frequencies, so correction ranking reflects actual usage rather than
alphabetical adjacency in the dictionary word list.
"""

import gzip
import math
import os
import pickle
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

# Bigram counts are keyed "prev\x1fword" -- a flat string key is faster to
# hash/lookup at scale than a tuple key across tens of millions of entries.
_BIGRAM_SEP = '\x1f'

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'ngram_model.pkl.gz'
)


class KannadaLanguageModel:
    def __init__(self):
        self.unigram: Dict[str, int] = defaultdict(int)
        self.bigram: Dict[str, int] = {}
        self.total_words = 0
        self._max_unigram = 0
        self._trained = False

    def train(self, sentences: Iterable[List[str]], min_bigram_count: int = 2, prune_every: int = 2_000_000):
        """
        Train on an iterable of sentences (each a list of Kannada words, in
        original order). Bigrams are only counted within a sentence, never
        across sentence boundaries, and bigrams seen fewer than
        min_bigram_count times are dropped to bound memory on large corpora.

        The raw bigram counter is periodically culled of below-threshold
        entries every `prune_every` sentences rather than only once at the
        end -- on a multi-gigabyte corpus the long tail of once-seen bigrams
        is what actually exhausts memory, so waiting until the final filter
        (the old behavior) lets that tail grow unbounded for the whole run.
        This is a lossy approximation (a bigram whose occurrences straddle a
        prune boundary can be undercounted) but bigram_bonus is only a coarse
        ranking signal, not a count anything else depends on being exact.
        """
        self.unigram = defaultdict(int)
        raw_bigram: Dict[str, int] = defaultdict(int)
        self.total_words = 0

        for n, words in enumerate(sentences, start=1):
            for w in words:
                self.unigram[w] += 1
                self.total_words += 1
            for i in range(len(words) - 1):
                raw_bigram[f"{words[i]}{_BIGRAM_SEP}{words[i + 1]}"] += 1

            if prune_every and n % prune_every == 0:
                raw_bigram = defaultdict(int, {k: c for k, c in raw_bigram.items() if c >= min_bigram_count})

        self.bigram = {k: c for k, c in raw_bigram.items() if c >= min_bigram_count}
        self._max_unigram = max(self.unigram.values(), default=0)
        self._trained = True

    def add_vocabulary(self, words: Iterable[str]):
        """
        Give dictionary words that never appear in the corpus a nonzero
        unigram count, so a valid-but-uncommon word isn't scored as if it
        were invalid.
        """
        for w in words:
            if w not in self.unigram:
                self.unigram[w] = 1
                self.total_words += 1
        self._max_unigram = max(self.unigram.values(), default=0)

    def score_candidate(self, candidate: str, prev_word: Optional[str] = None, next_word: Optional[str] = None) -> float:
        """
        Return a bounded [0.0, 0.3] ranking bonus: higher means the
        candidate (and its context) is more attested in the corpus.
        Bounded rather than a raw log-probability so the scale stays
        meaningful whether the model was trained on a few thousand
        dictionary words or a multi-billion-token corpus.
        """
        freq = self.unigram.get(candidate, 0)
        freq_bonus = 0.0
        if freq > 0 and self._max_unigram > 0:
            freq_bonus = 0.2 * (math.log1p(freq) / math.log1p(self._max_unigram))

        bigram_bonus = 0.0
        if prev_word and self.bigram.get(f"{prev_word}{_BIGRAM_SEP}{candidate}", 0) > 0:
            bigram_bonus += 0.1
        if next_word and self.bigram.get(f"{candidate}{_BIGRAM_SEP}{next_word}", 0) > 0:
            bigram_bonus += 0.1

        return round(min(0.3, freq_bonus + bigram_bonus), 4)

    def save(self, path: str):
        """
        Pickled rather than JSON -- for a ~30M-entry unigram/bigram table,
        pickle.load runs roughly 2x faster than json.load on the same
        gzip-compressed data (measured: ~29s vs ~56s), and it's stdlib-only.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, 'wb') as f:
            pickle.dump({
                'total_words': self.total_words,
                'unigram': dict(self.unigram),
                'bigram': self.bigram,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
        self.unigram = defaultdict(int, data['unigram'])
        self.bigram = data['bigram']
        self.total_words = data['total_words']
        self._max_unigram = max(self.unigram.values(), default=0)
        self._trained = True
        return True


_model = KannadaLanguageModel()


def train_model(sentences: Iterable[List[str]], min_bigram_count: int = 2):
    """Train on real corpus sentences -- see pipeline.correction.corpus."""
    _model.train(sentences, min_bigram_count=min_bigram_count)


def train_from_word_list(word_list: List[str]):
    """
    Fallback used when no trained corpus model is available: unigram-only
    counts from the dictionary vocabulary, no cross-word bigrams.
    """
    _model.train([[w] for w in word_list])


def add_vocabulary(words: Iterable[str]):
    _model.add_vocabulary(words)


def load_ngram_model(path: Optional[str] = None) -> bool:
    return _model.load(path or DEFAULT_MODEL_PATH)


def save_ngram_model(path: Optional[str] = None):
    _model.save(path or DEFAULT_MODEL_PATH)


def score_candidate(candidate: str, prev_word: Optional[str] = None, next_word: Optional[str] = None) -> float:
    return _model.score_candidate(candidate, prev_word, next_word)
