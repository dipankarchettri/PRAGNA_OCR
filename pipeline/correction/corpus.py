"""
Plain-text corpus ingestion for training the Kannada n-gram language model.
Streams line by line so a multi-gigabyte corpus (e.g. IndicCorpV2) never has
to fit in memory at once -- see tools/build_ngram_model.py for how a corpus
is turned into the cached model this pipeline actually loads at runtime.
"""

import gzip
import os
from typing import Iterator, List

from .tokenizer import tokenize


def _open_text(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='ignore')
    return open(path, 'r', encoding='utf-8', errors='ignore')


def iter_corpus_sentences(path: str) -> Iterator[List[str]]:
    """
    Yield one Kannada-word list per corpus line (IndicCorpV2 is one
    sentence/paragraph per line). Accepts a single .txt/.txt.gz file or a
    directory containing several. Non-Kannada tokens on the line are
    dropped -- the n-gram model only ever scores Kannada candidates.
    """
    if os.path.isdir(path):
        paths = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith('.txt') or f.endswith('.txt.gz'):
                    paths.append(os.path.join(root, f))
        paths.sort()
    else:
        paths = [path]

    for p in paths:
        with _open_text(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = [t['value'] for t in tokenize(line) if t['type'] == 'kannada']
                if words:
                    yield words
