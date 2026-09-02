"""
vLLM-server backend for the Sarvam correctors.

Why this exists at all: transformers cannot run either published quantization of
sarvam-30b *as* a quantized model. compressed-tensors (both the fp8 and the
W4A16 int4 builds) installs a forward pre-hook that decompresses the entire
model to bf16 on the first forward -- so a 20.9 GB checkpoint loads correctly,
then tries to become ~60 GB the moment you use it and OOMs a 48 GB card.
`run_compressed` and `use_optimized_inference` do not prevent it; the real
quantized kernels for that format live in vLLM, not in the compressed-tensors
package. vLLM also registers SarvamMoEForCausalLM natively, so the custom MoE
architecture needs no remote code here.

vLLM is talked to over its OpenAI-compatible HTTP server rather than imported,
for two reasons: it pins its own torch build (installing it into the pipeline's
venv would replace the working one), and a server keeps the 20 GB of weights
resident across CLI invocations instead of reloading them per run. Start it
with tools/serve_sarvam.sh.

This module mirrors sarvam_lm's interface -- score_sequences / log_probs /
complete / chat / has_chat_template -- so sarvam_corrector does not know or
care which backend is answering.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# 8077, not 8000: 8000/8001 are commonly already taken by other local
# services. Must match tools/serve_sarvam.sh's default.
VLLM_URL = os.environ.get('SARVAM_VLLM_URL', 'http://127.0.0.1:8077')
REQUEST_TIMEOUT = float(os.environ.get('SARVAM_VLLM_TIMEOUT', '120'))

_model_name: Optional[str] = None
_chat_template_known: Optional[bool] = None


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"{VLLM_URL}{path}",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', 'replace')[:400]
        raise RuntimeError(f"vLLM returned {e.code} for {path}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach the vLLM server at {VLLM_URL} ({e.reason}). "
            "Start it with tools/serve_sarvam.sh, or set SARVAM_VLLM_URL."
        ) from None


def is_available() -> bool:
    """True if a vLLM server is reachable and serving a model."""
    try:
        model_name()
        return True
    except Exception:
        return False


def model_name() -> str:
    """Id of the model the server has loaded (cached after the first call)."""
    global _model_name
    if _model_name is not None:
        return _model_name

    req = urllib.request.Request(f"{VLLM_URL}/v1/models")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach the vLLM server at {VLLM_URL} ({e.reason}). "
            "Start it with tools/serve_sarvam.sh, or set SARVAM_VLLM_URL."
        ) from None

    models = data.get('data') or []
    if not models:
        raise RuntimeError(f"vLLM at {VLLM_URL} is serving no models.")
    _model_name = models[0]['id']
    return _model_name


def describe() -> str:
    try:
        return f"{model_name()} via vLLM at {VLLM_URL}"
    except Exception as e:
        return f"vLLM unavailable: {e}"


def load(model_id: Optional[str] = None, device: Optional[str] = None) -> None:
    """
    Interface parity with sarvam_lm.load -- the server owns the weights, so this
    only verifies it is up, converting a mid-benchmark connection error into a
    clear failure at startup instead.
    """
    model_name()


def unload() -> None:
    """No-op: the server owns the weights and outlives this process."""
    global _model_name, _chat_template_known
    _model_name = None
    _chat_template_known = None


def score_sequences(texts: List[str], batch_size: int = 0) -> List[Tuple[float, int]]:
    """
    (total_log_probability, scored_token_count) per input, matching
    sarvam_lm.score_sequences.

    Uses the completions endpoint with echo=True and max_tokens=0, which makes
    vLLM return per-token logprobs for the *prompt* rather than for a
    completion -- the model scores the text we hand it instead of continuing
    it. The first prompt token has no predecessor and comes back as null; it is
    skipped, exactly as the transformers backend drops the first position.

    batch_size is accepted and ignored: vLLM batches server-side, and the whole
    candidate list is submitted in one request.
    """
    if not texts:
        return []

    data = _post('/v1/completions', {
        'model': model_name(),
        'prompt': texts,
        'max_tokens': 0,
        'echo': True,
        'logprobs': 0,
        'temperature': 0.0,
    })

    # Choices come back indexed; do not assume order.
    out: List[Tuple[float, int]] = [(0.0, 0)] * len(texts)
    for choice in data['choices']:
        token_logprobs = (choice.get('logprobs') or {}).get('token_logprobs') or []
        values = [lp for lp in token_logprobs if lp is not None]
        out[choice['index']] = (float(sum(values)), len(values))
    return out


def log_probs(texts: List[str], batch_size: int = 0) -> List[float]:
    return [lp for lp, _ in score_sequences(texts)]


def has_chat_template() -> bool:
    """
    True if the served model is instruction-tuned.

    Probed once by asking the chat endpoint for a single token: vLLM rejects
    chat requests for a model with no chat template, so a successful reply is
    the answer. Cheaper and more honest than duplicating the template lookup.
    """
    global _chat_template_known
    if _chat_template_known is not None:
        return _chat_template_known

    try:
        _post('/v1/chat/completions', {
            'model': model_name(),
            'messages': [{'role': 'user', 'content': 'ok'}],
            'max_tokens': 1,
            'temperature': 0.0,
        })
        _chat_template_known = True
    except RuntimeError:
        _chat_template_known = False
    return _chat_template_known


def complete(
    prompt: str,
    max_new_tokens: int = 256,
    stop: Optional[str] = '\n',
    temperature: float = 0.0
) -> str:
    payload: Dict[str, Any] = {
        'model': model_name(),
        'prompt': prompt,
        'max_tokens': max_new_tokens,
        'temperature': temperature,
    }
    if stop:
        payload['stop'] = [stop]

    data = _post('/v1/completions', payload)
    return data['choices'][0]['text'].strip()


def chat(
    instruction: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    system: Optional[str] = None
) -> str:
    messages = ([{'role': 'system', 'content': system}] if system else [])
    messages.append({'role': 'user', 'content': instruction})

    data = _post('/v1/chat/completions', {
        'model': model_name(),
        'messages': messages,
        'max_tokens': max_new_tokens,
        'temperature': temperature,
    })
    return (data['choices'][0]['message'].get('content') or '').strip()
