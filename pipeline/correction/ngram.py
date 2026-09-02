"""
N-Gram Language Model for Kannada Context Ranking
Scores alternative candidate words using real corpus unigram/bigram
frequencies, so correction ranking reflects actual usage rather than
alphabetical adjacency in the dictionary word list.
"""

import gzip
import heapq
import math
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# Bigram counts are keyed "prev\x1fword" -- a flat string key is faster to
# hash/lookup at scale than a tuple key across tens of millions of entries.
_BIGRAM_SEP = '\x1f'

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'ngram_model.pkl.gz'
)

# Below this size a file is kept as one whole-file shard for train_parallel --
# aligning a shard boundary costs a seek+readline per boundary, not worth it
# for files where a whole extra worker's-worth of parallelism wouldn't matter.
_SMALL_FILE_SHARD_FLOOR = 50 * 1024 * 1024


def _shard_corpus(corpus_path: str, n_shards: int) -> List[Tuple[str, int, Optional[int]]]:
    """
    Split a corpus path (a single file or a directory of .txt/.txt.gz,
    same input train_parallel accepts) into up to n_shards roughly-equal,
    line-boundary-aligned (path, start_byte, end_byte) chunks so a single
    huge file -- IndicCorpV2's Kannada portion ships as one file -- still
    parallelizes instead of pinning one worker while the others idle.
    .gz files can't be seeked into without decompressing, so each one is
    always its own single shard (end=None means "read to EOF").
    """
    if os.path.isdir(corpus_path):
        paths = []
        for root, _, files in os.walk(corpus_path):
            for f in files:
                if f.endswith('.txt') or f.endswith('.txt.gz'):
                    paths.append(os.path.join(root, f))
        paths.sort()
    else:
        paths = [corpus_path]

    shards: List[Tuple[str, int, Optional[int]]] = []
    for p in paths:
        if p.endswith('.gz'):
            shards.append((p, 0, None))
            continue
        size = os.path.getsize(p)
        if size <= _SMALL_FILE_SHARD_FLOOR or n_shards <= 1:
            shards.append((p, 0, size))
            continue
        raw_bounds = [i * size // n_shards for i in range(1, n_shards)]
        aligned = [0]
        with open(p, 'rb') as f:
            for b in raw_bounds:
                f.seek(b)
                f.readline()  # consume the partial line straddling this boundary
                aligned.append(f.tell())
        aligned.append(size)
        for i in range(len(aligned) - 1):
            if aligned[i] < aligned[i + 1]:
                shards.append((p, aligned[i], aligned[i + 1]))
    return shards


def _iter_shard_lines(path: str, start: int, end: Optional[int]):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                yield line
        return
    with open(path, 'rb') as f:
        f.seek(start)
        pos = start
        for raw_line in f:
            pos += len(raw_line)
            yield raw_line.decode('utf-8', errors='ignore')
            if end is not None and pos >= end:
                break


def _train_shard(args: Tuple[str, int, Optional[int], int, int]):
    """
    train_parallel's worker entry point -- module-level (not a bound method)
    so multiprocessing can pickle it for a subprocess. Counts one shard with
    the same periodic-pruning behavior as the sequential train(), but
    returns the raw per-shard bigram counts rather than final-pruning them
    here: a bigram that individually falls below min_bigram_count in every
    shard but clears it once shard counts are summed would otherwise be
    dropped for good. The caller merges all shards and prunes exactly once.
    """
    from .tokenizer import tokenize

    path, start, end, min_bigram_count, prune_every = args
    unigram: Dict[str, int] = defaultdict(int)
    raw_bigram: Dict[str, int] = defaultdict(int)
    total_words = 0
    n = 0

    for line in _iter_shard_lines(path, start, end):
        line = line.strip()
        if not line:
            continue
        words = [t['value'] for t in tokenize(line) if t['type'] == 'kannada']
        if not words:
            continue
        n += 1
        for w in words:
            unigram[w] += 1
            total_words += 1
        for i in range(len(words) - 1):
            raw_bigram[f"{words[i]}{_BIGRAM_SEP}{words[i + 1]}"] += 1
        if prune_every and n % prune_every == 0:
            raw_bigram = defaultdict(int, {k: c for k, c in raw_bigram.items() if c >= min_bigram_count})

    return dict(unigram), dict(raw_bigram), total_words


class KannadaLanguageModel:
    def __init__(self):
        self.unigram: Dict[str, int] = defaultdict(int)
        self.bigram: Dict[str, int] = {}
        self.total_words = 0
        self._max_unigram = 0
        self._freq_reference = 1
        self._trained = False

    def _compute_freq_reference(self) -> int:
        """
        Unigram count at which score_candidate's frequency bonus saturates
        to its maximum, recomputed from whichever corpus is actually loaded
        rather than a fixed constant. Deliberately NOT self._max_unigram:
        the single most frequent word in a real corpus is typically a
        function word (ಮತ್ತು/ಈ/ಎಂದು, ~10M occurrences in the AI4Bharat
        corpus) that occurs orders of magnitude more often than the next
        tier of common content words. Normalizing against that one outlier
        compresses the log-scale bonus for genuinely common words (e.g.
        ~90K occurrences) down to a sliver near zero, which is too weak to
        reliably outrank a rarer dictionary neighbor that's only slightly
        cheaper on raw edit distance -- exactly the failure mode that
        surfaced once the dictionary grew large enough for rare-but-real
        words to start appearing as edit-distance-adjacent candidates.
        Using a high percentile of the actual distribution instead anchors
        "maximally common" to the top of the content-word tier, not to
        whichever single outlier happens to sit above it.
        """
        if not self.unigram:
            return 1
        n = len(self.unigram)
        k = max(1, n // 10000)  # ~99.99th percentile
        return max(1, heapq.nlargest(k, self.unigram.values())[-1])

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
        self._freq_reference = self._compute_freq_reference()
        self._trained = True

    def train_parallel(self, corpus_path: str, workers: int, min_bigram_count: int = 2,
                        prune_every: int = 2_000_000, progress_cb=None):
        """
        Same trained result as train(), computed by splitting corpus_path (a
        directory or single file of .txt/.txt.gz -- same input
        pipeline.correction.corpus.iter_corpus_sentences accepts) into
        `workers` line-aligned shards and counting each in its own process
        (see _shard_corpus/_train_shard). A single huge file -- IndicCorpV2's
        Kannada portion ships as one multi-GB file -- is split internally by
        byte offset so it still parallelizes across all workers rather than
        pinning one of them while the rest sit idle on smaller files.
        progress_cb(completed, total), if given, is called as each shard's
        result comes back (shards complete out of order).
        """
        shards = _shard_corpus(corpus_path, workers)
        tasks = [(p, s, e, min_bigram_count, prune_every) for p, s, e in shards]

        self.unigram = defaultdict(int)
        raw_bigram: Dict[str, int] = defaultdict(int)
        self.total_words = 0

        with mp.Pool(processes=workers) as pool:
            for i, (uni, bi, tot) in enumerate(pool.imap_unordered(_train_shard, tasks), 1):
                for w, c in uni.items():
                    self.unigram[w] += c
                for k, c in bi.items():
                    raw_bigram[k] += c
                self.total_words += tot
                if progress_cb:
                    progress_cb(i, len(tasks))

        self.bigram = {k: c for k, c in raw_bigram.items() if c >= min_bigram_count}
        self._max_unigram = max(self.unigram.values(), default=0)
        self._freq_reference = self._compute_freq_reference()
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
        if freq > 0 and self._freq_reference > 0:
            ratio = math.log1p(freq) / math.log1p(self._freq_reference)
            freq_bonus = 0.2 * min(1.0, ratio)

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
        self._freq_reference = self._compute_freq_reference()
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


def has_bigram_support(candidate: str, prev_word: Optional[str] = None, next_word: Optional[str] = None) -> bool:
    """
    True if `candidate` is actually attested next to prev_word or next_word
    in the trained corpus -- real local context evidence, distinct from the
    candidate's own standalone frequency. Used to gate the lowest-precision
    candidate-generation mechanisms (blind single-character deletion/
    insertion in corrector.py), which can reach a common, unrelated
    dictionary word by pure edit-distance coincidence with no linguistic
    relationship to the original word.
    """
    if prev_word and _model.bigram.get(f"{prev_word}{_BIGRAM_SEP}{candidate}", 0) > 0:
        return True
    if next_word and _model.bigram.get(f"{candidate}{_BIGRAM_SEP}{next_word}", 0) > 0:
        return True
    return False


def has_corpus_counts() -> bool:
    """
    Whether the loaded unigram counts carry real frequency information.

    init_pipeline falls back to train_from_word_list() when no corpus model is
    cached, and add_vocabulary() pads every known dictionary word to a count of
    exactly 1. In that state every word looks equally (un)common, so any gate
    of the form "frequency >= 2" would reject the entire vocabulary and switch
    correction off completely rather than merely loosening it. Callers use this
    to skip frequency-based gating when there is no frequency to speak of.
    """
    return _model._max_unigram > 1


def corpus_frequency(word: str) -> int:
    """
    Raw unigram count for `word` in the trained corpus. Distinct from
    score_candidate's bounded ranking bonus -- this is a plain attestation
    count, used to recognize real words the curated Hunspell dictionary
    doesn't list (very common when running against the full real-corpus
    model, which covers ~12M unique surface forms against the dictionary's
    ~46k). Note add_vocabulary() pads unseen dictionary words to exactly 1,
    so callers wanting genuine multi-occurrence corpus evidence (not just
    "is a dictionary word") should require count >= 2.
    """
    return _model.unigram.get(word, 0)
