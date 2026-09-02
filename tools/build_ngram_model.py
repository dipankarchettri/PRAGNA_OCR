"""
Build the cached n-gram language model from a real Kannada text corpus.

This is a maintainer-only tool: regular users get the prebuilt
data/ngram_model.pkl.gz via setup.py (same pattern as the Hunspell
dictionary and fonts). Run this only when rebuilding that cache -- e.g.
after picking a new corpus or widening the training set.

Corpus source: AI4Bharat IndicCorpV2 (CC-0), Kannada portion
(https://huggingface.co/datasets/ai4bharat/IndicCorpV2). The exact file
layout inside that dataset repo isn't pinned here -- this script lists the
repo's files at run time and downloads whatever matches the language code,
so it doesn't break if AI4Bharat reorganizes the repo.

Usage:
    # Download the Kannada portion of IndicCorpV2 and train from it
    python tools/build_ngram_model.py --download --max-size-mb 2000

    # Train from a corpus you already have (one sentence/paragraph per line,
    # .txt or .txt.gz files, single file or a directory of them)
    python tools/build_ngram_model.py --corpus-dir ./my_kannada_corpus/

Requires `huggingface_hub` for --download (not a core project dependency --
install it just for this script: pip install huggingface_hub).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.correction.corpus import iter_corpus_sentences
from pipeline.correction.ngram import KannadaLanguageModel, DEFAULT_MODEL_PATH

REPO_ID = "ai4bharat/IndicCorpV2"
LANG_MATCH = ("kan_Knda", "/kn.", "/kn/", "_kn.", ".kn.")


def download_corpus(dest_dir: str, max_size_mb: float, limit_files: int) -> str:
    """
    Download Kannada corpus file(s) from IndicCorpV2, capped at max_size_mb
    total. Each matched file is fetched via an HTTP Range request rather than
    hf_hub_download's whole-file fetch -- IndicCorpV2 ships Kannada as one
    ~18.6GB file rather than many small shards, so a whole-file download
    would ignore the cap entirely. The partial fetch is truncated to the last
    complete line so corpus.py never sees a cut-off sentence.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_url
    except ImportError:
        print("Missing dependency: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    import requests

    api = HfApi()
    print(f"[*] Listing files in {REPO_ID} (dataset repo)...")
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    matches = [f for f in all_files if any(tag in f for tag in LANG_MATCH)]

    if not matches:
        print("[!] No Kannada-tagged files found via the usual naming patterns "
              f"({LANG_MATCH}). Inspect the repo manually:\n"
              f"    https://huggingface.co/datasets/{REPO_ID}/tree/main", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Found {len(matches)} Kannada file(s): {matches}")
    if limit_files:
        matches = matches[:limit_files]

    os.makedirs(dest_dir, exist_ok=True)
    cap_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else None
    remaining = cap_bytes
    for f in matches:
        if cap_bytes and remaining is not None and remaining <= 0:
            print(f"[*] Reached --max-size-mb {max_size_mb}, stopping.")
            break

        url = hf_hub_url(REPO_ID, f, repo_type="dataset")
        local_path = os.path.join(dest_dir, os.path.basename(f))
        headers = {}
        if cap_bytes:
            headers['Range'] = f'bytes=0-{remaining - 1}'

        print(f"[↓] {f} ({'first ' + str(max_size_mb) + ' MB' if cap_bytes else 'full file'})...")
        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()

        written = 0
        with open(local_path, 'wb') as out_f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                out_f.write(chunk)
                written += len(chunk)

        if cap_bytes:
            # Drop everything after the last full line -- the Range cut the
            # file off mid-byte-stream, possibly mid-sentence.
            with open(local_path, 'rb') as in_f:
                data = in_f.read()
            last_newline = data.rfind(b'\n')
            if last_newline != -1:
                with open(local_path, 'wb') as out_f:
                    out_f.write(data[:last_newline + 1])
            remaining -= written

        print(f"    -> {written / (1024 * 1024):.1f} MB written to {local_path}")

    print(f"[✓] Corpus ready at {dest_dir}")
    return dest_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus-dir', help='Directory or file of already-downloaded corpus text (.txt/.txt.gz)')
    parser.add_argument('--download', action='store_true', help='Download the Kannada portion of IndicCorpV2 first')
    parser.add_argument('--download-dir', default=os.path.join('data', 'corpus_kn'), help='Where to store downloaded corpus files')
    parser.add_argument('--max-size-mb', type=float, default=2000, help='Cap on how much corpus data to download (default: 2000 MB)')
    parser.add_argument('--limit-files', type=int, default=0, help='Cap on number of corpus shard files to download (0 = no cap)')
    parser.add_argument('--min-bigram-count', type=int, default=2, help='Drop bigrams seen fewer than this many times (bounds memory)')
    parser.add_argument('--output', default=DEFAULT_MODEL_PATH, help='Where to write the trained model')
    args = parser.parse_args()

    if not args.download and not args.corpus_dir:
        parser.error('Pass --download or --corpus-dir')

    corpus_path = args.corpus_dir
    if args.download:
        corpus_path = download_corpus(args.download_dir, args.max_size_mb, args.limit_files)

    print(f"[*] Training n-gram model from corpus at: {corpus_path}")
    model = KannadaLanguageModel()

    def sentence_stream():
        count = 0
        for sentence in iter_corpus_sentences(corpus_path):
            count += 1
            if count % 500000 == 0:
                print(f"    ... {count:,} sentences processed")
            yield sentence

    model.train(sentence_stream(), min_bigram_count=args.min_bigram_count)

    print(f"[*] Trained on {model.total_words:,} words, "
          f"{len(model.unigram):,} unique unigrams, {len(model.bigram):,} bigrams (post-pruning)")

    model.save(args.output)
    print(f"[✓] Saved model to {args.output}")


if __name__ == '__main__':
    main()
