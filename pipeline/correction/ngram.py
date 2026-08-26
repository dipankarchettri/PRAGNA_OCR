"""
N-Gram Language Model for Kannada Context Ranking
Scores alternative candidate words based on character/word frequencies and context.
"""

from collections import defaultdict
import math
from typing import List, Optional


class KannadaLanguageModel:
    def __init__(self):
        self.unigram: dict[str, int] = defaultdict(int)
        self.bigram: dict[tuple[str, str], int] = defaultdict(int)
        self.total_words = 0
        self._trained = False

    def train(self, word_list: List[str]):
        """
        Train language model on the dictionary vocabulary list.
        """
        self.unigram.clear()
        self.bigram.clear()
        self.total_words = 0

        for word in word_list:
            self.unigram[word] += 1
            self.total_words += 1

        for i in range(len(word_list) - 1):
            self.bigram[(word_list[i], word_list[i + 1])] += 1

        self._trained = True

    def unigram_log_prob(self, word: str) -> float:
        """Calculate Laplace smoothed log probability."""
        count = self.unigram.get(word, 0)
        vocab_size = len(self.unigram) or 1
        return math.log((count + 1) / (self.total_words + vocab_size))

    def score_candidate(self, candidate: str, prev_word: Optional[str] = None, next_word: Optional[str] = None) -> float:
        """Score candidate using unigram and bigram if available."""
        score = self.unigram_log_prob(candidate)
        if prev_word and (prev_word, candidate) in self.bigram:
            score += 1.5
        if next_word and (candidate, next_word) in self.bigram:
            score += 1.5
        return score


_model = KannadaLanguageModel()


def train_model(word_list: List[str]):
    _model.train(word_list)


def score_candidate(candidate: str, prev_word: Optional[str] = None, next_word: Optional[str] = None) -> float:
    return _model.score_candidate(candidate, prev_word, next_word)
