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
./venv/bin/python tests/test_correction_precision.py   # words that must NOT change

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

# Build the evaluation fixture set (deterministic; images are gitignored)
./venv/bin/python tools/build_eval_set.py --docs 12 --degrade 0,1,2,3

# Measure the correction engine: CER/WER before vs after + fixed/broke counts
./venv/bin/python tools/correction_bench.py --pages 'tests/fixtures/eval/*_p0??.png'
./venv/bin/python tools/correction_bench.py --synthetic 200   # smoke test, no ground truth

# Rank candidate GLYPH_CONFUSIONS rows by how many corrections they make
# *reachable* (not by how often OCR makes the mistake -- see the tool's docstring)
./venv/bin/python tools/mine_confusions.py

# Same bench against the Surya engine (needs venv-surya, see below)
./venv/bin/python tools/correction_bench.py --pages 'tests/fixtures/real/*.png' --engine surya
```

Always run `tests/test_pipeline.py` and `tests/test_correction_precision.py` after touching
anything under `pipeline/correction/` or `pipeline/exporter/`.

## Measuring changes

`tools/build_eval_set.py` typesets clean Kannada book text from
`data/kanaja_docx_raw/` into page images at four degradation levels, writing the exact text
it laid out as the reference. `tools/correction_bench.py` then scores the engine on them.

**Read `tools/build_eval_set.py`'s docstring before trusting a number from it.** These are
typeset pages, not scans — a deterministic regression gate, not evidence about real book
scans. Real page/transcript pairs dropped into `tests/fixtures/eval/` as `page.jpg` +
`page.txt` are the honest test. (The obvious shortcut — LibreOffice `.docx`→PDF, then read
the PDF's text layer — was tried and does not work here: on this corpus LibreOffice emits
justified Kannada with **no space characters at all** and duplicates matras, so the
"reference" was itself wrong. Hence typesetting.)

**CER alone is not the metric.** The bench reports `fixed` / `broke` separately on purpose:
a wrong correction silently corrupts training data, where an untouched OCR error stays
visible downstream. An engine that lowers CER while raising `broke` is a regression.

Current state, all three sets, at the production defaults (`kan`, psm 3, oem 1):

| set | CER base → corrected | WER base → corrected | fixed | broke | precision |
|---|---|---|---|---|---|
| **9 real pages** | 0.0558 → **0.0516** | 0.2889 → **0.2658** | 15 | **0** | **1.000** |
| 24 typeset pages | 0.0053 → **0.0035** | 0.0335 → **0.0172** | 2 | 1 | 0.667 |
| 300 synthetic lines | 0.0151 → **0.0101** | 0.1639 → **0.0970** | 352 | 44 | 0.889 |

The real pages are the honest number; the other two are regression gates. Note the synthetic
CER is understated by the joiner normalization below — measured with joiners neutralized on
both sides it is 0.0091, and the gap is the metric penalizing a deliberate normalization
rather than real damage.

**The synthetic row is not comparable across a change to `CONFUSION_PAIRS`.**
`correction_bench` manufactures its noise *from* `GLYPH_CONFUSIONS` (`_confusion_sources`),
so adding a row changes the corpus as well as the engine and the before/after are two
different datasets. To compare, generate the documents once and ablate the rows out of the
engine afterwards. Done that way for the 2c rows in `edit_distance.py`: on one fixed corpus,
fixed 327 → 352, broke 39 → 43, precision 0.893 → 0.891, CER 0.0102 → 0.0097. And for the 2d
rows: fixed 347 → 352, broke 44 → 44, precision 0.887 → 0.889. Both look like regressions in
the uncontrolled table and are not.

**Where the remaining error actually is.** Of 128 single-word substitution errors across the
nine real pages, the correct word is in the generated candidate set for only **16 (12.5%)**
— and of those, **zero** are then discarded by the corpus-frequency target gate. At the time
this was measured the engine's ceiling was candidate-generation *recall*, not its gates or its
ranking. **That is no longer true** — see "It is the gates now" below. Of the 112 it never
proposes, 75 are 1 akshara away and 37 are 2 or more, so widening the search radius is not
sufficient either: the 1-akshara misses were glyph pairs absent from `GLYPH_CONFUSIONS`
(ಮ/ಥ inside a conjunct — since added, see 2d) and proper nouns absent from the dictionary.

**The error budget, on the corrected output.** Across 10,439 reference characters on the nine
real pages, what is left is no longer mostly single-word:

| bucket | events | ref chars | share |
|---|---|---|---|
| **garbled multi-word runs** | 60 | **1,407** | 13.5% |
| 1:1 word substitutions | 115 | 929 | 8.9% |
| OCR split one word into two | 23 | 265 | 2.5% |
| non-Kannada tokens, drops, merges | 32 | ~220 | 2.1% |

and of the 115 remaining substitutions: **54** have a target absent from the dictionary
(unreachable by any generator — loanwords, place names, productive compounds), 24 the engine
declines to touch as already-valid, 22 are in the dictionary with no generator producing them,
and 15 are reachable but lose to ranking. So roughly **half the residual substitutions are not
a correction-engine problem at all**, which is the argument for a second OCR engine rather than
more tuning.

**Do not build a two-edit search.** Of 76 residual errors whose truth is a dictionary word, 21
are one confusion substitution away (ranking/gating losses, not table gaps), **3** are two away,
and 52 are not reachable by confusion substitutions at any depth. A constrained
two-substitution pass would chase three errors.

**It is the gates now, not recall and not ranking.** The 12.5% figure above was measured before
three rounds of recall work and no longer describes the binding constraint. Of the 27 residual
errors whose truth is in the candidate set today: 8 are already corrected, **14 have the truth
ranked #1 and a gate rejects it**, and only 5 are genuinely out-ranked. Phase 3d (ranking and
tie-breaks) is therefore worth 5 errors at most — do it for tidiness, not for accuracy.

The gate doing the rejecting is `FREQUENCY_DOMINANCE_RATIO` (250). Nine of the 14 fail it
outright, e.g. ಪೋಲೀಸರ → ಪೊಲೀಸರ at 39×, ನಡೆಯುತ್ತಾನೆ → ಪಡೆಯುತ್ತಾನೆ at 14×, ಜಪದಕಟ್ಟಿ → ಜಪದಕಟ್ಟೆ at
5×. The rest fall to the correction-target gate (candidate frequency < 2) or the bigram-support
requirement on unconstrained mechanisms.

**Measured negative result — conditioning the dominance gate on OCR confidence (do not retry
without new evidence).** The gate cannot be loosened on frequency, because frequency does not
order the two cases it has to separate: ಪೋಲೀಸರ must be CORRECTED at 39× while ಮಂಡಲಿಗೆ must be
KEPT at 89×, so the keep case sits at the *higher* ratio. Tesseract's own per-word confidence
looked like the missing signal, and statistically it is — over the nine pages, words the gate
wrongly blocks have median confidence 63 against 91 for words the engine correctly leaves alone:

| confidence | <60 | <70 | <80 | <85 |
|---|---|---|---|---|
| should-correct (13) | 46% | 62% | 77% | 77% |
| should-keep (213) | 8% | 10% | 15% | 22% |

That separation is real in aggregate and useless per word. Swept over thresholds {60, 70, 80} ×
ratios {3, 5, 15, 50}, every cell bought fixes and paid for them in breaks, with **CER flat or
worse**: the baseline is fixed 15 / broke 0 / CER 0.0516, and the best cell in the entire sweep
is fixed 16 / broke 1 / CER 0.0516 — identical CER, precision 1.000 → 0.94. The 10% of
should-keep words below the threshold are enough to eat the whole gain. Adding a bigram-support
requirement on top makes it *inert* on real pages instead (fixed 15, broke 0, CER and WER
unchanged to four decimals) — the bigram check rejects precisely the corrections the relaxation
was meant to admit. Reverted.

The remaining lesson: these 14 need evidence the pipeline does not currently have. Frequency
cannot order them and confidence cannot either. That is a second, independent argument for a
better OCR pass rather than more post-hoc correction.

**Reading that list is not enough to act on it** — the 2c audit in `edit_distance.py` is the
worked example. Counting which glyphs OCR confuses tells you what went *wrong*, not what the
engine cannot *reach*, and the two lists differ sharply. ಕ/ಯ was named here as a missing pair
on exactly that reasoning and is not one: it is the single most-observed unlisted confusion on
these pages (12 instances, 10 word types, 3 pages), and adding it makes **zero** new corrections
reachable, because nearly every instance is a subscript already covered by the `್ಯ`/`್ಕ` row
and the residue is an out-of-dictionary place name. A confusion row can only pay off where the
*rewritten word resolves to a real dictionary word*, so the test that decides is: inject the
pair, and count instances that move from absent-in-`collect_kannada_candidates` to present.
Two of the seven best-attested candidate pairs (ಕ/ಯ, ಕ/ರ) bought nothing by it, and a
third (ಬ/ಲ) passed it on two instances that turn out to be one proper noun.

**Observability and reachability are different axes.** `mine_confusions.py` was widened to
align akshara streams *inside* a differing run rather than comparing whole words, on the
reasoning that garbled runs hold more error mass than clean substitutions (1,407 chars vs 929)
and the word-level miner could not see into them. It surfaces plenty more observations — ನ/ಸ
at 7 over 4 pages, plus ಬ/ಟ, ಚ/ಟ, ನ/ಮ, ಪ/ಶ, ತ/ರ — and nearly all gain zero, because the
multi-error words the wider alignment can now *see* are exactly the words a
single-substitution generator cannot *fix*. Keep the widening (it is strictly more evidence,
and it found ನ/ಸ) but do not expect scope to convert into recall.

`tools/mine_confusions.py` runs this whole audit and ranks candidates by that gain. Re-run it
whenever a new page/transcript pair lands in `tests/fixtures/real/` — it is the one part of
this engine that gets better purely by being given more ground truth. As of the nine current
pages the only unlisted pair still showing a gain is ಂ/ು (2 observations, gain 1), held back
as too thin to justify widening the search.

This is also why growing the dictionary has stopped paying. 28.1% of correct Kannada words on
these pages are absent from the 622k-form membership set and 11.7% are unattested in the
n-gram corpus, but those words are *productively formed* (ಪತ್ರಿಕೋದ್ಯೋಗಿಯಾಗಲು,
ಸೃಷ್ಟಿಸೌಂದರ್ಯದ), so no flat word list closes the gap at any size.

**Measured negative result — compositional analysis (do not retry without new evidence).**
The obvious fix for the above is to score a word's *parts* instead of its surface string:
strip inflection, split the stem into dictionary words, and accept the word if every part is
well attested (ಸೃಷ್ಟಿಸೌಂದರ್ಯದ is unattested; ಸೃಷ್ಟಿ 36,606 and ಸೌಂದರ್ಯ 47,038 are common).
This was implemented and wired into all three gates, then ablated per site:

| config | typeset broke | typeset precision | synth broke | synth precision | real CER |
|---|---|---|---|---|---|
| baseline | 1 | 0.667 | 40 | 0.897 | 0.0532 |
| validation only | 1 | 0.667 | 40 | 0.897 | 0.0532 |
| resolution only | 1 | 0.667 | 40 | 0.896 | 0.0532 |
| **target gate** | **3** | **0.400** | **65** | **0.844** | **0.0537** |

It recovered 33% of the unattested vocabulary as *legal* targets and improved nothing,
because legal is not the same as *reachable*: the 12.5% recall figure above means the gate
was never what blocked those words. Widening it only gave the edit search more places to
land. The validation half is provably inert — `collect_kannada_candidates` only
short-circuits on validity when `corpus_frequency >= VALID_WORD_TRUST_FREQUENCY` (1,000),
and a compositionally-attested word is by construction below 2. Reverted.

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
2. `ocr_repairs.py` — universal script-level normalization (Repha `...೯` → `ರ್...`, zero-digit→anusvara `೦`→`ಂ`, illegal vowel+matra cleanup, word-final joiner removal), run before dictionary lookup.

   **Word-final ZWNJ/ZWJ after a virama** is dropped (`FINAL_JOINER_REGEX`). Tesseract emits it constantly (19 occurrences across the 9 real pages, all after a virama, 17 word-final; the human transcripts have 2), it renders identically, and it makes the token compare unequal to the correct word — an invisible corruption, the worst kind for this corpus. Worth 0.0532 → 0.0520 CER and 0.2782 → 0.2697 WER on real pages, and it takes the corrector net-positive on typeset pages for the first time (0.0053 → 0.0035, previously 0.0057).

   Scoped to word-final on purpose, and *not* gated on frequency. A medial joiner is meaningful — it forces the half-form over the conjunct ligature (`ಕ್<ZWNJ>ವ` vs `ಕ್ವ`) — so it survives. Word-finally it is free orthographic variation: clean published Kannada writes it 12,166 times in 161k corpus lines, and the bare form outnumbers the joined one by only 1.4–3.4× (ಯಾದವ್ 51,930 / 16,708), far below `FREQUENCY_DOMINANCE_RATIO`, so no frequency test can separate "Tesseract added one" from "the writer meant it". Normalizing free variation is the right call for a training corpus — it stops one word tokenizing as two — but it is a normalization, not a correction, so it does not appear in `fixed`/`broke`, and `raw_text` keeps the original for audit.
3. `morphology.py` — Sandhi/agglutinative suffix stripping via **longest-match-first** against the Hunspell affix rules; `join_root_suffix()` must stay in sync with any suffix list changes here.
4. `edit_distance.py` — weighted Levenshtein with a Kannada optical-glyph confusion matrix (visually similar glyph pairs get lower substitution cost so they outrank generic dictionary stems) plus universal single-character insertion/deletion candidate generation.
5. `ngram.py` — unigram/bigram frequency scoring to rank among surviving candidates. `init_pipeline()` loads a cached real-corpus model from `data/ngram_model.pkl.gz` (gzip+pickle — ~2x faster to load than gzip+JSON at this scale) if present (built via `tools/build_ngram_model.py` from AI4Bharat IndicCorpV2), falling back to unigram-only counts from the dictionary word list if that cache doesn't exist. `score_candidate()` always returns a bounded `[0.0, 0.3]` bonus regardless of which one is loaded, so `corrector.py` never needs corpus-size-dependent tuning.
6. `dictionary.py` — loads `data/kn_IN.dic` + `data/kn_IN.aff` and maintains **two** vocabularies, which is the thing to understand about this module:
   - **membership** — all 589,521 entries plus affix expansion, **622,407 forms**. A word here is never flagged as an error.
   - **targets** — the subset also attested in the n-gram corpus, **246,861 forms**. Only these may be *proposed* as corrections.

   The errors are not symmetric, hence the split: a real word missing from membership gets "corrected" into something else (silent corpus corruption), while a junk string admitted as a target is somewhere the edit search can land.

   **The dictionary ships with the repo as `data/kn_IN.dic.gz`** (2.5 MB gzipped, 18.4 MB raw). `dictionary.py` reads the `.gz` in place — no decompression step, no build artifact to keep in sync, and a fresh clone works without `setup.py` fetching anything. An uncompressed `data/kn_IN.dic` still takes precedence if present, which keeps local experiments working; that path is gitignored so one person's experiment can't be committed over everyone's dictionary. The `.aff` rules are committed separately (small, hand-maintained, load-bearing).

   Note the trap `setup.py` now guards: its fallback downloads the **stock LibreOffice `kn_IN`, 19,645 entries — 3% of this vocabulary**. Since a plain `.dic` beats the `.gz`, letting that run on a repo that already ships the dictionary would silently downgrade the pipeline. It is skipped whenever the `.gz` is present, and warns loudly if it ever does run.

   **Bigger is not better here.** A 2.5M-entry build was measured and rejected: harvested from corpus text containing OCR output, it admitted OCR errors as entries (`ಕಾಥಿ`, the misreading of `ಕಾಫಿ`, was listed at frequency 37) — and since membership means "never flag this", the dictionary taught the corrector that the error was correct. At identical settings 2.5M was worse on precision and broken-word count alike, on identical real-page results.

   `VALID_WORD_TRUST_FREQUENCY` (corrector.py) exists for the residue of the same problem: dictionary membership only short-circuits correction for words attested ≥1,000 times. Below that a "valid" word still goes through candidate generation and is defended by `FREQUENCY_DOMINANCE_RATIO` instead — which is the check built to separate "rare but real" (ಮಂಡಲಿಗೆ, 82× behind its variant, kept) from "an error swamped by its own correct form" (ಕಾಥಿ, 2,442× behind ಕಾಫಿ, corrected).

`corrector.py` (`suggest_kannada_word`, `correct_text`, `correct_layout_lines`) combines all of the above and classifies each fix as `ocr_repair` (blue), `word_correction` (green), or `hybrid` (yellow) — this classification drives the diff coloring in the web UI.

**Non-text line filtering:** `correct_layout_lines()` flags each OCR'd line as `is_likely_non_text` when its mean Tesseract confidence falls below `NON_TEXT_LINE_CONFIDENCE` (40) -- below that, Tesseract isn't misreading degraded prose, it's hallucinating text rows out of non-text content (embedded photos, decorative banner art, logos). Validated against 7 real ground-truth pages spanning the hard real cases (faded scans, heavy Sanskrit vocabulary the dictionary doesn't cover, degraded photocopies): every genuine body-text line across all of them still cleared 55 confidence; only actual graphic content ever dropped below 40, into single digits. Deliberately confidence-only, not combined with a dictionary valid-word-ratio check -- ratio misfires on real Sanskrit-vocabulary text (0% dictionary hits at completely normal confidence), so it can't safely gate this. `pipeline/__init__.py` excludes flagged lines from the training-corpus text (`corrected_text` / exported `.txt`) but keeps them in `raw_text` and the generated PDF (`all_layout_lines`, unfiltered) for audit/visual fidelity -- nothing is silently discarded, only excluded from the clean-corpus output. Never applies to PDF-digital-extraction lines (no `conf` field -- never OCR'd from pixels).

**Zero-hardcoding rule (strict, enforced in review):** never add word-specific or stem-specific mappings to `CONFUSION_PAIRS`, `GLYPH_CONFUSIONS`, `ocr_repairs.py`, or anywhere else in the correction engine — no `'ಧ್ಯಯ' -> 'ಧ್ಯಯನ'`-style special cases. Every fix must fall out of the universal mechanisms above (script normalization, optical confusion matrices, longest-match morphology, 1-edit search, n-gram ranking). New vocabulary belongs in `data/kn_IN.dic`, not as a conditional branch in code. This history is visible in git log (search for "hardcoding" / "stem-specific").

### Surya, the second OCR engine (`--engine surya`)

Added because two independent measurements said the headroom is in the OCR pass, not in
correction: about half the residual substitutions have an out-of-dictionary target no generator
can reach, and the reachable-but-failing ones need evidence neither frequency nor confidence
supplies.

**Raw OCR on the nine real pages, each engine with its own best non-text filtering:**

| engine | filter | CER | WER |
|---|---|---|---|
| Tesseract | conf < 40 | 0.0539 | 0.2847 |
| **Surya** | conf < 80 + Latin-majority | **0.0461** | **0.2628** |

Surya wins on 7 of 9 pages, including the degraded page 04 that motivated this work (0.0951 →
0.0681). Note the confidence thresholds are **not comparable between engines** — dropping
Tesseract lines below 70 destroys real text (CER 0.0558 → 0.1133), while for Surya it helps.
Compare each at its own setting.

**End-to-end, and the finding that matters:**

| config | CER | WER | fixed | broke | precision |
|---|---|---|---|---|---|
| Tesseract + full corrector (production) | 0.0516 | 0.2658 | 15 | **0** | **1.000** |
| **Surya, no correction** | **0.0462** | **0.2621** | 0 | **0** | **1.000** |
| Surya + `ocr_repair` only | 0.0456 | 0.2519 | 22 | 10 | 0.688 |
| Surya + full corrector | 0.0471 | 0.2578 | 24 | 17 | 0.585 |

**The correction engine is net-harmful on Surya output.** It takes CER from 0.0462 to 0.0471
and breaks 17 words at precision 0.585, against 1.000 on Tesseract. That is not a Surya defect
— every gate and cost in `pipeline/correction/` was calibrated against Tesseract's error
distribution, and Surya's is different. Until it is re-tuned, **the best configuration measured
is Surya with correction off**, which beats production on CER and WER at zero corruption risk.
Re-tuning the gates against Surya is the obvious next piece of work and has not been done.

**Page 08 is the cautionary tale.** Surya first measured 13× worse there than Tesseract
(0.1909 vs 0.0146). Its Kannada was near-perfect at 99+ confidence; the entire gap was
hallucinated Latin (`'Carl State State'`, `'LS ARS / COURS'`) at 40–60 confidence, overlapping
the range real degraded prose occupies. Confidence could not separate it; script could.
`is_latin_majority` in `corrector.py` now feeds `is_likely_non_text`, taking that page to
0.0065. It is a measured no-op for Tesseract, which with `--lang kan` has no Latin to emit.

**Setup** (isolated venv — Surya pins `pillow<11` and this pipeline runs 12.x, so a shared
install silently downgrades PIL underneath ingestion):

```bash
python3 -m venv venv-surya
./venv-surya/bin/pip install 'surya-ocr==0.16.7' 'transformers>=4.56.1,<5'
# match the CUDA build to your driver; on a 12.8 driver:
./venv-surya/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0+cu128 torchvision
```

Pinned to **0.16.7 deliberately**: 0.22 removed the in-process torch predictor and drives
inference through vLLM (spawning a Docker container), llama.cpp, or a remote OpenAI-compatible
endpoint. None is a reasonable dependency here. 0.16.7 is the last line with plain
`FoundationPredictor` + `RecognitionPredictor`. Roughly 10 s/page on an RTX 6000 Ada after a
one-off model load, so callers batch whole documents (`surya_ocr_images_with_layout` takes a
list) rather than looping per page.

**Licensing.** Code is Apache-2.0; weights are a modified AI Pubs Open RAIL-M (free for
research, personal use, and organisations under $5M funding/revenue). Unlike the excluded
Krutrim models there is **no non-compete clause**, and Datalab ships document-OCR tooling
rather than an Indic LLM, so nothing here restricts building a Kannada training corpus. The
OpenRAIL-M use restrictions are harm-based and propagate to derivatives — read them before
redistributing the corpus itself.

### Other pipeline modules

- `pipeline/ingestion/` — `pdf_processor.py` (PyMuPDF digital-text + page classification + `pdf2image` rasterization at configurable DPI) and `image_processor.py` (EXIF auto-rotation, `normalize_resolution` auto-upscales low-DPI input toward ~300 DPI since Kannada ottakshara/subscript conjuncts alias badly below that). `preprocess_for_ocr()` runs deskew (via `detect_skew_angle`'s projection-profile search) on every page right before OCR, for both the image-file path (inside `load_and_preprocess_image`) and the PDF-rasterization path in `pipeline/__init__.py` -- validated empirically: near-zero effect on already-straight scans (skew below `MIN_SKEW_DEGREES` is left untouched), meaningful CER/WER improvement on skewed ones (e.g. a synthetic 3° skew: CER 2.65%→1.28%, WER 20%→10%). `enhance_contrast` on `preprocess_for_ocr`/`load_and_preprocess_image` itself defaults to `False` and should stay that way: a blanket +20% contrast boost applied unconditionally can take a clean scan from near-perfect OCR to complete word-salad, confirmed on a real document, and grayscale std-dev/percentile-spread showed no clean signal separating the pages it helps from the one it catastrophically breaks. Instead, `pipeline/__init__.py`'s `_ocr_with_adaptive_contrast()` (used by `process_document` when `adaptive_contrast=True`, the default; disable via CLI `--no-adaptive-contrast`) OCRs each page twice -- once untouched, once with the boost -- and keeps whichever run **Tesseract's own mean word confidence** scores higher, ties going to the unboosted run. Validated against 7 real ground-truth pages: catches genuine wins on faded scans (CER 16.68%→8.80% on one), stays neutral on near-ties, and correctly discards the boosted run on the catastrophic-corruption case (confidence drops 25-35 points there, vs. a ≤1-point wobble on genuine improvements) -- confidence *compared between two runs* separates the cases that a static image-statistics threshold could not. Costs a second Tesseract pass per page.
- `pipeline/ocr/tesseract_engine.py` — multi-language Tesseract wrapper (`ocr_image`, `ocr_image_with_layout`), baseline-aware reading-order sort, alignment classification.
- `pipeline/exporter/` — `pdf_generator.py` (FPDF2 + embedded `NotoSansKannada-Regular.ttf`, must use `pdf.epw` and `new_x="LMARGIN", new_y="NEXT"` for correct line wrapping) and `text_exporter.py` (per-page/combined TXT, JSON report).

### Web app (`web/app.py`)

Flask + SSE: uploads go through `_SESSIONS` for progress streaming back to the browser during long OCR jobs; `init_pipeline()` runs via `@app.before_request`. Frontend is `web/static/js/app.js` (live autocorrect + diff viewer) and `web/templates/index.html`.

**Startup warmup.** `start_warmup()` loads the dictionary and n-gram model on a background
thread at boot, so the server binds its port and serves the page immediately instead of
stalling on the ~33 s model load. `/api/system-status` exposes `warmup`
(`pending`/`loading`/`ready`/`failed`) and the frontend reads it.

### OCR defaults and why

| Setting | Default | Rationale |
|---|---|---|
| `--lang` | `kan` | Adding `eng` on a monolingual Kannada page makes Tesseract emit Latin glyphs for ambiguous shapes; the corrector skips non-Kannada tokens, so that output is unrecoverable. |
| `--psm` | `3` | Automatic segmentation. Wins or ties on 8 of 9 real pages (mean raw CER 0.0549 vs 0.0585 for psm 6) and improves the corrected result too — CER 0.0532 vs 0.0572, `broke` 0 vs 1. Was `6` until real transcripts existed; that choice came from proxy metrics (vocabulary-validity ratio, mean confidence) which both reward the confident dictionary-shaped output psm 6 produces *while segmenting a page wrongly*. Try `6` on a page that genuinely is one uniform block. |
| `--oem` | `1` | LSTM only — the bundled `tessdata_best` models ship no legacy engine. |
| `--dpi` | `400` | Subscript conjuncts alias badly at 300. |

`tessdata/kan.traineddata` is the `tessdata_best` LSTM model; `tessdata_standard/` holds the smaller stock model kept for comparison via `tools/ocr_bench.py`.
