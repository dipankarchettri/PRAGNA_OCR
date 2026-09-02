"""
Correction-engine selection.

The pipeline calls correction through this module rather than importing
corrector.correct_text directly, so that swapping the rule engine for a
language-model-backed one is a parameter rather than a code change. Every
engine returns identical shapes, so nothing downstream (export, diff colouring,
the web UI) needs to know which one ran.

  'rule'             The dictionary + morphology + weighted-edit-distance +
                     n-gram engine. No extra dependencies. The default, and
                     the baseline any LM engine has to beat.
  'sarvam-rerank'    Rule-engine candidates, ranked by Sarvam-1.
  'hybrid'           Rule-engine corrections, vetoed by Sarvam-1.
  'sarvam-generate'  Sarvam-1 rewrites whole lines, few-shot.

See sarvam_corrector.py for what each Sarvam mode does and what it risks.
The Sarvam engines import torch/transformers lazily, so this module stays
importable on a machine that has neither.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from .corrector import correct_layout_lines, correct_text

ENGINE_RULE = 'rule'
ENGINE_SARVAM_RERANK = 'sarvam-rerank'
ENGINE_SARVAM_GENERATE = 'sarvam-generate'
ENGINE_HYBRID = 'hybrid'

ENGINES = (ENGINE_RULE, ENGINE_SARVAM_RERANK, ENGINE_HYBRID, ENGINE_SARVAM_GENERATE)
SARVAM_ENGINES = (ENGINE_SARVAM_RERANK, ENGINE_HYBRID, ENGINE_SARVAM_GENERATE)


def validate_engine(engine: str) -> str:
    if engine not in ENGINES:
        raise ValueError(f"Unknown correction engine '{engine}'. Expected one of: {', '.join(ENGINES)}")
    return engine


def engine_status() -> Dict[str, Dict[str, Any]]:
    """
    Which engines can actually run right now, and why not when they can't.

    Exists so the web UI can offer the LM engines without pretending they are
    installed -- selecting one on a machine with no torch and no vLLM server
    would otherwise fail only after the user uploads a document. Cheap enough
    to call per request: it checks an import and, at most, one HTTP call to a
    local port.
    """
    from . import sarvam_lm, sarvam_vllm

    if sarvam_vllm.is_available():
        sarvam_ok, reason = True, ''
    elif sarvam_lm.is_installed():
        sarvam_ok, reason = True, ''
    else:
        sarvam_ok = False
        reason = ('Needs torch + transformers (pip install -r requirements-sarvam.txt) '
                  'or a running vLLM server (tools/serve_sarvam.sh).')

    status = {ENGINE_RULE: {'available': True, 'reason': ''}}
    for name in SARVAM_ENGINES:
        status[name] = {'available': sarvam_ok, 'reason': reason}
    return status


def preload_engine(engine: str) -> None:
    """
    Load whatever the engine needs up front.

    Worth calling before a batch or a long document so the ~10 s model load
    doesn't land in the middle of the first page and get attributed to it.
    """
    if engine in SARVAM_ENGINES:
        from .sarvam_corrector import _lm
        _lm().load()


def correct_text_with(
    text: str,
    engine: str = ENGINE_RULE,
    allowed_types: Optional[Set[str]] = None,
    word_confidences: Optional[List[Tuple[str, float]]] = None
) -> Dict[str, Any]:
    validate_engine(engine)
    if engine == ENGINE_RULE:
        return correct_text(text, allowed_types=allowed_types, word_confidences=word_confidences)

    from .sarvam_corrector import correct_text_sarvam
    return correct_text_sarvam(text, mode=engine, word_confidences=word_confidences)


def correct_layout_lines_with(
    layout_lines: List[Dict[str, Any]],
    engine: str = ENGINE_RULE,
    allowed_types: Optional[Set[str]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    validate_engine(engine)
    if engine == ENGINE_RULE:
        return correct_layout_lines(layout_lines, allowed_types=allowed_types)

    from .sarvam_corrector import correct_layout_lines_sarvam
    return correct_layout_lines_sarvam(layout_lines, mode=engine)
