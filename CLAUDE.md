# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PRAGNA_OCR is a Kannada (`kn_IN`) OCR and morphological autocorrection pipeline. It ingests scanned/searchable PDFs and images, runs Tesseract LSTM OCR where needed, repairs Indic-script OCR artifacts, corrects spelling via Sandhi-aware morphology + weighted edit distance + n-gram scoring, and exports layout-preserving Unicode PDFs, text, and JSON diff reports. There is both a Flask web dashboard and a CLI.

## Goal

The end purpose of this pipeline is to scan Kannada books at volume and turn them into a clean text dataset for training a Kannada LLM. This means CER/WER on real book pages (not just short benchmark snippets) is the metric that actually matters, and correctness/precision of automated corrections matters more than coverage — a wrong "fix" silently corrupts training data in a way that's much harder to catch downstream than leaving an OCR error untouched.

**Excluded: Krutrim (Chitrapathak / Krutrim-1 / Krutrim-2 / Chitrarth) models.** Chitrapathak-2 (Krutrim AI Labs) is currently one of the strongest OCR systems measured on Kannada specifically (word-level ANLS ~18.8, beating Surya OCR's ~24.4, close behind Gemini-2.5-Flash's ~17.2 — lower is better), but it and all other Krutrim-branded models ship under the **Krutrim Community License Agreement v1.0**, whose §5 ("Restriction on Use to Compete with Krutrim") bans using the software to "develop, market, sell, or support competing products or services" — with no research/non-commercial carve-out and no scale/revenue threshold, unlike the license's own MAU-gated definition of "Commercial Use" elsewhere. Since Krutrim itself ships multilingual Indic LLMs (Krutrim-1, Krutrim-2) with Kannada support, using their OCR model to build a Kannada LLM training corpus sits squarely inside that non-compete clause regardless of project scale. Do not suggest or reintroduce Krutrim models for this pipeline.

## Setup

```bash
# System deps (Tesseract + Poppler)
sudo apt install tesseract-ocr tesseract-ocr-kan tesseract-ocr-eng poppler-utils   # Debian/Ubuntu
brew install tesseract tesseract-lang poppler                                     # macOS

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Downloads Hunspell kn_IN dictionary, Noto Sans Kannada font, and tessdata models
python setup.py
```

## Common commands

```bash
# Run the full test suite (plain unittest, no pytest)
./venv/bin/python tests/test_pipeline.py

# Run a single test case/method
./venv/bin/python -m unittest tests.test_pipeline.TestKannadaPipeline.test_dynamic_ocr_repairs -v

# Web dashboard
./venv/bin/python web/app.py          # http://127.0.0.1:5010 (PORT env overrides)

# CLI: single file, direct text, batch folder
python cli.py scan.pdf --lang kan --dpi 400 --output-dir ./results
python cli.py --text "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತದೆ"
python cli.py --batch ./scans/ --output-dir ./batch_results/

# Sweep Tesseract PSM/upscale settings against sample pages to measure OCR quality
# (ranks by % Kannada tokens the corrector recognizes + Latin-script "unrecoverable" runs;
# drop a same-basename .txt transcript next to the image to get real CER instead)
./venv/bin/python tools/ocr_bench.py page.jpg --psm 3,4,6 --upscale 0,1
./venv/bin/python tools/ocr_bench.py page.jpg --dump ./bench_out

# Pick the correction engine (CLI default: rule; web dashboard default: hybrid).
python cli.py scan.pdf --engine hybrid
python cli.py --text "..." --engine sarvam-rerank

# Compare correction engines on identical input (CER/WER + fixed/broke counts)
./venv/bin/python tools/sarvam_bench.py --synthetic 200            # no ground truth needed
./venv/bin/python tools/sarvam_bench.py --pages scans/*.jpg        # real pages + .txt truth
./venv/bin/python tools/sarvam_bench.py --text-pair raw.txt truth.txt --engines rule,hybrid
```

Always run `tests/test_pipeline.py` after touching anything under `pipeline/correction/` or `pipeline/exporter/`.

## Architecture

Everything routes through `pipeline/__init__.py`'s two entry points, both of which call `init_pipeline()` first (loads the Hunspell dictionary and trains the n-gram model, once, lazily):

- `process_text_input(text)` — direct string in, corrected string out. Used by CLI `--text` and the web app's live autocorrect endpoint.
- `process_document(input_path, ...)` — full file pipeline: ingest → OCR/extract → correct → export. Used by CLI file/batch mode and the web app's upload flow.

### `process_document` per-page decision logic (`pipeline/__init__.py`)

For PDFs, `inspect_pdf()` classifies each page as `searchable`, `mixed`, or `scanned` — the decision is made **per page**, and for `mixed` pages, **per block within the page** (`_page_text_is_valid` / block-level `is_valid`):

- `searchable` pages → `extract_searchable_pdf_layout` (PyMuPDF), no OCR.
- `scanned`/`mixed` pages → `rasterize_page_masking_valid_text` paints over any already-trustworthy digital blocks before rasterizing, so Tesseract only OCRs what the text layer can't be trusted for; OCR line coordinates (pixels) are rescaled to PDF points (`effective_dpi = dpi * (ocr_img.width / img.width)`, `scale = effective_dpi / 72.0`) so they can be merged with digital-extraction lines by vertical position into one reading order.

Images always go through `load_and_preprocess_image` → `ocr_image_with_layout`.

Both paths converge on `correct_layout_lines()` before export.

### `pipeline/correction/` — the correction engine

Stages run in this fixed order (see `corrector.py`):
1. `tokenizer.py` — splits text into Kannada (`ಀ-೿` + ZWJ/ZWNJ) vs. non-Kannada tokens by exact byte offset; non-Kannada tokens (English, numbers, punctuation) are never touched, and `reconstruct(tokens)` must losslessly rebuild the original around them.
2. `ocr_repairs.py` — universal script-level normalization (Repha `...೯` → `ರ್...`, zero-digit→anusvara `೦`→`ಂ`, illegal vowel+matra cleanup), run before dictionary lookup.
3. `morphology.py` — Sandhi/agglutinative suffix stripping via **longest-match-first** against the Hunspell affix rules; `join_root_suffix()` must stay in sync with any suffix list changes here.
4. `edit_distance.py` — weighted Levenshtein with a Kannada optical-glyph confusion matrix (visually similar glyph pairs get lower substitution cost so they outrank generic dictionary stems) plus universal single-character insertion/deletion candidate generation.
5. `ngram.py` — unigram/bigram frequency scoring to rank among surviving candidates. `init_pipeline()` loads a cached real-corpus model from `data/ngram_model.pkl.gz` (gzip+pickle — ~2x faster to load than gzip+JSON at this scale) if present (built via `tools/build_ngram_model.py` from AI4Bharat IndicCorpV2), falling back to unigram-only counts from the dictionary word list if that cache doesn't exist. `score_candidate()` always returns a bounded `[0.0, 0.3]` bonus regardless of which one is loaded, so `corrector.py` never needs corpus-size-dependent tuning.
6. `dictionary.py` — loads `data/kn_IN.dic` + `data/kn_IN.aff` (Hunspell `kn_IN`, ~46k words) as the vocabulary source of truth.

`corrector.py` (`suggest_kannada_word`, `correct_text`, `correct_layout_lines`) combines all of the above and classifies each fix as `ocr_repair` (blue), `word_correction` (green), or `hybrid` (yellow) — this classification drives the diff coloring in the web UI.

**Non-text line filtering:** `correct_layout_lines()` flags each OCR'd line as `is_likely_non_text` when its mean Tesseract confidence falls below `NON_TEXT_LINE_CONFIDENCE` (40) -- below that, Tesseract isn't misreading degraded prose, it's hallucinating text rows out of non-text content (embedded photos, decorative banner art, logos). Validated against 7 real ground-truth pages spanning the hard real cases (faded scans, heavy Sanskrit vocabulary the dictionary doesn't cover, degraded photocopies): every genuine body-text line across all of them still cleared 55 confidence; only actual graphic content ever dropped below 40, into single digits. Deliberately confidence-only, not combined with a dictionary valid-word-ratio check -- ratio misfires on real Sanskrit-vocabulary text (0% dictionary hits at completely normal confidence), so it can't safely gate this. `pipeline/__init__.py` excludes flagged lines from the training-corpus text (`corrected_text` / exported `.txt`) but keeps them in `raw_text` and the generated PDF (`all_layout_lines`, unfiltered) for audit/visual fidelity -- nothing is silently discarded, only excluded from the clean-corpus output. Never applies to PDF-digital-extraction lines (no `conf` field -- never OCR'd from pixels).

**Zero-hardcoding rule (strict, enforced in review):** never add word-specific or stem-specific mappings to `CONFUSION_PAIRS`, `GLYPH_CONFUSIONS`, `ocr_repairs.py`, or anywhere else in the correction engine — no `'ಧ್ಯಯ' -> 'ಧ್ಯಯನ'`-style special cases. Every fix must fall out of the universal mechanisms above (script normalization, optical confusion matrices, longest-match morphology, 1-edit search, n-gram ranking). New vocabulary belongs in `data/kn_IN.dic`, not as a conditional branch in code. This history is visible in git log (search for "hardcoding" / "stem-specific").

### Correction engines (`pipeline/correction/engine.py`)

The pipeline calls correction through `correct_text_with(...)` /
`correct_layout_lines_with(...)` rather than `corrector.correct_text` directly, so the
corrector is selectable per run (`--engine`, or `engine=` on `process_document` /
`process_text_input`). Every engine returns identical shapes, so export, the JSON diff
report and the web UI's colouring don't know which one ran. Ingestion, OCR and export are
untouched by the choice — an engine comparison on the same input is comparing correction
quality and nothing else.

| Engine | What it does |
|---|---|
| `rule` (default) | The dictionary + morphology + weighted-edit-distance + n-gram engine described above. No extra dependencies. |
| `sarvam-rerank` | Candidates still come from `generate_kannada_candidates`; Sarvam-1 replaces only the *ranking* (n-gram score + frequency gates), scoring the whole sentence once per candidate. The LM can never emit a word the dictionary/morphology layer didn't propose. |
| `hybrid` | The rule engine runs unmodified, then the LM **vetoes** any word-level correction it scores below the original. Strictly a subset of `rule`'s changes — trades recall for precision, never the reverse. Deterministic `ocr_repair` fixes are never vetoed. |
| `sarvam-generate` | Few-shot completion; the LM rewrites whole lines. The only mode that can emit text nothing else proposed, and so the only one that can hallucinate fluent-but-wrong Kannada into the corpus. Guarded by `_generation_is_plausible` (drift, length ratio, script), but treat its output as unverified. |

`pipeline/correction/sarvam_lm.py` is a correction-agnostic model wrapper (load, score
strings, complete a prompt); `sarvam_corrector.py` holds the three modes above. Optional
deps live in `requirements-sarvam.txt` (kept out of `requirements.txt` — ~7 GB of CUDA
wheels for a non-default engine). `SARVAM_MODEL_ID` points the wrapper at a different
checkpoint or a local path, which is the seam for a fine-tune.

#### Which checkpoint

`SARVAM_MODEL_ID`, `--sarvam-model`, or `sarvam_lm.set_model()` selects it; `KNOWN_MODELS`
in `sarvam_lm.py` holds the two verified ones.

| | `sarvam-1` | `sarvam-30b` |
|---|---|---|
| Licence | Sarvam Research License — non-commercial, **binds Outputs too** | **Apache 2.0** |
| Tuning | base completion only | instruction-tuned |
| Size | 2B dense, ~4.3 GB bf16 | 30B MoE / **2.4B active**, ~20.9 GB W4A16 |
| Context | 8K | 65K |

**Getting sarvam-30b onto one 48 GB card took three attempts — don't retry the dead ends:**

| build | outcome |
|---|---|
| `sarvamai/sarvam-30b-fp8` | NVIDIA ModelOpt format; transformers has no `modelopt` quantizer. Won't load at all. |
| `RedHatAI/sarvam-30b-FP8-dynamic` | Same fp8 weights in `compressed-tensors`. Loads at 38.7 GB with `run_compressed`, then **OOMs on the first forward** — compressed-tensors has no true fp8 kernel here, it dequantizes per layer during compute. fp8 buys disk, not VRAM; it only pays off under vLLM. |
| `sarvamai/sarvam-30b-gguf` | `sarvam_moe` is not in llama.cpp mainline ([request closed as not planned](https://github.com/ggml-org/llama.cpp/issues/20175)), so Ollama / LM Studio / `llama-cpp-python` all reject it. Needs a patched build of an unmerged PR. |
| `mastersubhajit/sarvam-30b-AWQ-4bit` | **Works.** W4A16 int4 `pack-quantized` — real packed 4-bit kernels, 20.9 GB resident, ~26 GB headroom. |

**Backends.** `sarvam-1` runs in-process through transformers (`sarvam_lm.py`). `sarvam-30b`
cannot: compressed-tensors (both its fp8 and int4 builds) installs a forward pre-hook that
decompresses the *whole model* to bf16 on the first forward, so a 20.9 GB checkpoint loads
fine and then OOMs the moment you use it. Neither `run_compressed=True` nor
`use_optimized_inference=True` prevents it — the real quantized kernels for that format live
in vLLM, which also registers `SarvamMoEForCausalLM` natively. So the 30B is served by a
vLLM server and reached over HTTP (`sarvam_vllm.py`):

```bash
./tools/serve_sarvam.sh          # vLLM in its own venv-vllm, holds ~21 GB resident
python cli.py page.jpg --engine hybrid    # auto-detects the server
```

`SARVAM_BACKEND` (`auto` / `vllm` / `transformers`) forces the choice; `auto` uses the
server if one answers and transformers otherwise, so a machine with no server behaves
exactly as before. vLLM lives in a separate venv because it pins its own torch build.
Scoring goes through `/v1/completions` with `echo=True, max_tokens=0`, which returns
per-token logprobs for the prompt instead of a completion.

> **The 30B is blocked on this machine's NVIDIA driver, not on the code.** vLLM's published
> wheels (verified on 0.26.0 and 0.28.0) ship kernels linked against `libcudart.so.13`, and
> CUDA 13 needs driver **≥580**; this box runs **570.211 / CUDA 12.8**. Pinning torch does
> not help — vLLM compiles its own kernels, and `venv-vllm` had a correct `torch
> 2.11.0+cu128` while vLLM's `.so` files still demanded CUDA 13. Everything else for the 30B
> is written and tested; update the driver and `./tools/serve_sarvam.sh` should just work.
> Note also that PyPI's default `torch` is a cu130 build, so the pipeline venv needs
> `--index-url https://download.pytorch.org/whl/cu128` until the driver is updated.

For the transformers backend, loading is quantization-aware: a quantized checkpoint keeps
its own dtype and gets `device_map='auto'`; an unquantized one is placed directly in
bfloat16.

4-bit weights make log-probs noisier than bf16, and `LM_MIN_MARGIN` is a log-prob threshold
— **re-sweep it when changing checkpoints** rather than assuming sarvam-1's calibration
carries over.

Because sarvam-30b is instruction-tuned, `sarvam-generate` switches from few-shot
completion to the model's own chat template (`_rewrite_line` / `INSTRUCT_SYSTEM`) — feeding
an instruct model raw few-shot text works worse than either approach used properly.

#### Measured: sarvam-1, 100 synthetic-noise lines

Uncorrected baseline CER 0.0285. `LM_MIN_MARGIN=8`:

| engine | CER | fixed | broke | secs |
|---|---|---|---|---|
| `rule` | 0.0205 | 149 | 20 | 19 |
| `sarvam-rerank` | 0.0194 | 158 | 11 | 47 |
| `hybrid` | **0.0194** | 146 | **6** | 32 |
| `sarvam-generate` | 0.0681 | 5 | 37 | 372 |

`sarvam-generate` on a *base* model is actively destructive — it nearly triples CER over
doing nothing, and 37 of its 42 effective changes damaged correct words. That result says
nothing about instruction-tuned generation; re-measure before drawing conclusions about
sarvam-30b. `hybrid` and `sarvam-rerank` both beat the rule engine once the margin is
calibrated (at the untuned default of 1.0, `sarvam-rerank` was *worse* than `rule` — the
threshold, not the model, was the problem).

Caveat on all of it: synthetic noise is generated by inverting the corrector's own
glyph-confusion matrix, so the rule engine is being tested on exactly the error model it
was built around. This structurally favours `rule`/`hybrid`. Real ground-truth pages
(`--pages`) are the honest test.

Two things constrain what Sarvam-1 can be here:

- **It is a base text-completion model, not instruction-tuned** — its own model card says it
  "cannot be used directly as a chat or an instruction-following model". So "tell it to fix
  the text" is not a supported mode; few-shot completion and candidate *scoring* are. This
  is why `sarvam-rerank`/`hybrid` (LM as scorer) are the modes worth measuring first, and
  why fine-tuning on (noisy → clean) pairs is the real path to an LM corrector that beats
  the rule engine.
- **Licence: Sarvam AI Research License** — restricts the model, derivatives **and its
  Outputs** to non-commercial and research use. Corrected text is an Output, so a corpus
  built with these engines inherits that restriction. Unlike the Krutrim licence rejected
  above there is *no* non-compete clause, so nothing here bars building a Kannada LLM — the
  constraint is purely non-commercial/research.

`LM_MIN_MARGIN` (how much more probable the LM must find a candidate before it may
overwrite what OCR read) is **not yet calibrated** — the default 1.0 nat is a starting
point, not a measured threshold. `tools/sarvam_bench.py` exists to sweep it; results
produced before that sweep are provisional. The benchmark reports `fixed` / `broke` counts
separately from CER on purpose: an engine that lowers CER while raising `broke` is not an
improvement for this project's goal.

### Other pipeline modules

- `pipeline/ingestion/` — `pdf_processor.py` (PyMuPDF digital-text + page classification + `pdf2image` rasterization at configurable DPI) and `image_processor.py` (EXIF auto-rotation, `normalize_resolution` auto-upscales low-DPI input toward ~300 DPI since Kannada ottakshara/subscript conjuncts alias badly below that). `preprocess_for_ocr()` runs deskew (via `detect_skew_angle`'s projection-profile search) on every page right before OCR, for both the image-file path (inside `load_and_preprocess_image`) and the PDF-rasterization path in `pipeline/__init__.py` -- validated empirically: near-zero effect on already-straight scans (skew below `MIN_SKEW_DEGREES` is left untouched), meaningful CER/WER improvement on skewed ones (e.g. a synthetic 3° skew: CER 2.65%→1.28%, WER 20%→10%). `enhance_contrast` on `preprocess_for_ocr`/`load_and_preprocess_image` itself defaults to `False` and should stay that way: a blanket +20% contrast boost applied unconditionally can take a clean scan from near-perfect OCR to complete word-salad, confirmed on a real document, and grayscale std-dev/percentile-spread showed no clean signal separating the pages it helps from the one it catastrophically breaks. Instead, `pipeline/__init__.py`'s `_ocr_with_adaptive_contrast()` (used by `process_document` when `adaptive_contrast=True`, the default; disable via CLI `--no-adaptive-contrast`) OCRs each page twice -- once untouched, once with the boost -- and keeps whichever run **Tesseract's own mean word confidence** scores higher, ties going to the unboosted run. Validated against 7 real ground-truth pages: catches genuine wins on faded scans (CER 16.68%→8.80% on one), stays neutral on near-ties, and correctly discards the boosted run on the catastrophic-corruption case (confidence drops 25-35 points there, vs. a ≤1-point wobble on genuine improvements) -- confidence *compared between two runs* separates the cases that a static image-statistics threshold could not. Costs a second Tesseract pass per page.
- `pipeline/ocr/tesseract_engine.py` — multi-language Tesseract wrapper (`ocr_image`, `ocr_image_with_layout`), baseline-aware reading-order sort, alignment classification.
- `pipeline/exporter/` — `pdf_generator.py` (FPDF2 + embedded `NotoSansKannada-Regular.ttf`, must use `pdf.epw` and `new_x="LMARGIN", new_y="NEXT"` for correct line wrapping) and `text_exporter.py` (per-page/combined TXT, JSON report).

### Web app (`web/app.py`)

Flask + SSE: uploads go through `_SESSIONS` for progress streaming back to the browser during long OCR jobs; `init_pipeline()` runs via `@app.before_request`. Frontend is `web/static/js/app.js` (live autocorrect + diff viewer) and `web/templates/index.html`.

**Engine selection.** The dashboard defaults to **`hybrid`**, not `rule` — it makes
measurably fewer wrong corrections at the same error rate, which is what the corpus goal
cares about. That is only viable because `start_warmup()` loads the dictionary, n-gram model
and LM on a background thread at boot (`PRAGNA_WEB_ENGINE` overrides the default; set it to
`rule` to skip the LM entirely). The server binds its port immediately and serves the page
while the model loads — measured 43.8 s warmup, after which the first real correction takes
0.58 s instead of stalling ~20 s. `/api/system-status` exposes `warmup`
(`pending`/`loading`/`ready`/`failed`) and the frontend polls it so the hint text stops
saying "loading" without a page reload. If warmup fails (no torch, no GPU) the state records
why and `effective_default_engine()` degrades to `rule`, so the dashboard stays usable.

Both the live editor and the document pipeline carry a Correction
Engine dropdown; the choice rides along as `engine` on `/api/correct-text` (JSON),
`/api/process-stream` (query param) and `/api/process-document` (form field).
`/api/system-status` returns `engines` from `engine_status()`, and the frontend disables
options the backend can't run — so a machine without torch shows the LM engines greyed out
with the reason rather than failing after an upload. `requested_engine()` falls back to
`rule` for an *unknown* name (a stale browser tab shouldn't kill a long OCR job) but 503s
for a known-but-unavailable one, since silently correcting with a different engine than the
user picked would misreport what produced the text. Results report the engine the server
actually used, read from the response rather than the dropdown.

The live editor's keystroke debounce stretches from 300 ms to 1200 ms with an LM engine, since
a forward pass per uncertain word can't keep up with typing. `sarvam_lm` also holds an
`_infer_lock` around forward passes: Flask runs threaded, and one CUDA model must not be
entered concurrently.

### OCR defaults and why

| Setting | Default | Rationale |
|---|---|---|
| `--lang` | `kan` | Adding `eng` on a monolingual Kannada page makes Tesseract emit Latin glyphs for ambiguous shapes; the corrector skips non-Kannada tokens, so that output is unrecoverable. |
| `--psm` | `6` | Best measured on single-column book pages; use `3` for mixed/multi-column layouts. |
| `--oem` | `1` | LSTM only — the bundled `tessdata_best` models ship no legacy engine. |
| `--dpi` | `400` | Subscript conjuncts alias badly at 300. |

`tessdata/kan.traineddata` is the `tessdata_best` LSTM model; `tessdata_standard/` holds the smaller stock model kept for comparison via `tools/ocr_bench.py`.
