"""
Sarvam-1 (2B) language-model wrapper.

Thin, correction-agnostic layer over the model: load it once, score strings,
complete a prompt. All the actual correction logic lives in
sarvam_corrector.py -- this file knows nothing about Kannada OCR.

Two things about Sarvam-1 shape everything built on top of it:

1. It is a **base text-completion model**, not instruction-tuned. Its own
   model card states it "cannot be used directly as a chat or an
   instruction-following model". So an instruction like "fix the OCR errors
   in this text" is not a supported use -- the two usable modes are
   few-shot completion (sarvam_corrector's 'generate' mode) and, more
   reliably, using it purely as a *scorer* over candidates something else
   generated ('rerank' / 'hybrid' modes).

2. Its licence (Sarvam AI Research License) restricts the model, its
   derivatives **and its Outputs** to non-commercial and research purposes.
   Text corrected through this module is an Output under that licence, so any
   training corpus built with it inherits that restriction. Unlike the
   Krutrim licence this project rejected outright (see CLAUDE.md), there is
   no non-compete clause here -- the constraint is purely
   non-commercial/research use.

Import of torch/transformers is deferred to load() so that importing
pipeline.correction stays cheap and dependency-free for the rule engine.
"""

import os
import threading
from typing import List, Optional, Tuple

# Overridable so a fine-tuned checkpoint (or a local path) can be swapped in
# without touching code -- fine-tuning is the intended way to use a base
# model for a downstream task, and this is the seam for it.
MODEL_ID = os.environ.get('SARVAM_MODEL_ID', 'sarvamai/sarvam-1')

# Known-good checkpoints, for reference and for the benchmark's --model flag.
#
# sarvam-30b is a different proposition from sarvam-1 in three ways that matter
# here: it is Apache 2.0 (so neither it nor its Outputs constrain what the
# corrected corpus can be used for, unlike sarvam-1's research licence), it is
# instruction-tuned (so `sarvam-generate` is a real mode rather than a base
# model guessing at a task it never saw), and it is a Mixture-of-Experts with
# only ~2.4B of its 30B parameters active per token, so it is nowhere near 15x
# slower than sarvam-1 despite the size.
#
# Getting sarvam-30b onto a single 48 GB card under transformers took three
# attempts; the two dead ends are recorded so they aren't retried:
#
#   sarvamai/sarvam-30b-fp8       NVIDIA ModelOpt format. transformers has no
#                                 modelopt quantizer at all -- will not load.
#   RedHatAI/...-FP8-dynamic      Same fp8 weights in compressed-tensors, which
#                                 transformers does read. Loads at 38.7 GB with
#                                 run_compressed, then OOMs on the first
#                                 forward: compressed-tensors has no true fp8
#                                 kernel here, it dequantizes each layer during
#                                 compute, and that spike doesn't fit in the
#                                 ~8.5 GB left. fp8 buys disk, not VRAM --
#                                 it only pays off under vLLM, which does have
#                                 fp8 kernels.
#   ...-AWQ-4bit                  W4A16 int4, pack-quantized -- real packed
#                                 4-bit kernels, 20.9 GB resident, ~26 GB of
#                                 headroom. This is the one that runs.
#
# The GGUF builds are a dead end for an unrelated reason: sarvam_moe is not in
# llama.cpp mainline (feature request closed as not planned), so Ollama,
# LM Studio and llama-cpp-python all reject it as an unknown architecture.
#
# 4-bit weights make log-probs noisier than bf16, which matters for the
# margin-thresholded rerank/hybrid modes -- re-sweep LM_MIN_MARGIN when
# switching checkpoints rather than assuming sarvam-1's calibration carries.
KNOWN_MODELS = {
    'sarvam-1': 'sarvamai/sarvam-1',
    'sarvam-30b': 'mastersubhajit/sarvam-30b-AWQ-4bit',
    'sarvam-30b-fp8': 'RedHatAI/sarvam-30b-FP8-dynamic',  # needs vLLM, not transformers
}

# Sequences scored per forward pass. Candidate sets per word are small
# (MAX_LM_CANDIDATES), so this mostly bounds memory on long context windows.
DEFAULT_BATCH_SIZE = int(os.environ.get('SARVAM_BATCH_SIZE', '16'))

_model = None
_tokenizer = None
_device = None
_load_lock = threading.Lock()


def set_model(name: str) -> None:
    """
    Choose the checkpoint for subsequent load() calls.

    Accepts a KNOWN_MODELS key ('sarvam-1', 'sarvam-30b-fp8'), a Hugging Face
    repo id, or a local path. Unloads any currently-loaded model if this is a
    change, so a benchmark can compare checkpoints in one process.
    """
    global MODEL_ID
    target = KNOWN_MODELS.get(name, name)
    if target == MODEL_ID:
        return
    unload()
    MODEL_ID = target


def _quant_method(config) -> Optional[str]:
    """Quantization family of a checkpoint, or None if it is unquantized."""
    qc = getattr(config, 'quantization_config', None)
    if qc is None:
        return None
    if isinstance(qc, dict):
        return qc.get('quant_method')
    return getattr(qc, 'quant_method', None)


def is_installed() -> bool:
    """True if the optional torch/transformers dependencies are importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def is_loaded() -> bool:
    return _model is not None


def describe() -> str:
    """One-line description of what is currently loaded, for CLI/bench output."""
    if not is_loaded():
        return f"{MODEL_ID} (not loaded)"
    return f"{MODEL_ID} on {_device} ({next(_model.parameters()).dtype})"


def load(model_id: Optional[str] = None, device: Optional[str] = None) -> None:
    """
    Load the model and tokenizer once, process-wide.

    Uses bfloat16 on CUDA (the dtype Sarvam-1 was trained in) and float32 on
    CPU, where bfloat16 matmuls are slow. A quantized checkpoint carries its own
    dtype in its quantization config, so the request is dropped in that case
    rather than fighting it.

    `trust_remote_code` is on because sarvam-30b ships a custom MoE
    implementation (`modeling_sarvam_moe.py`) that transformers has no built-in
    class for. That executes code from the model repo, which is why
    KNOWN_MODELS names specific publishers rather than accepting any string
    silently.

    Sizing: sarvam-1 is ~4.3 GB in bfloat16; sarvam-30b-fp8 is ~38.7 GB and
    needs a 48 GB card, which is why `device_map='auto'` is used for anything
    that large -- it will spill to CPU rather than OOM if the card is smaller.
    """
    global _model, _tokenizer, _device

    if _model is not None:
        return

    with _load_lock:
        if _model is not None:
            return

        if not is_installed():
            raise RuntimeError(
                "The Sarvam corrector needs torch + transformers, which are not "
                "installed. Install them with: pip install -r requirements-sarvam.txt"
            )

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        target = KNOWN_MODELS.get(model_id or '', model_id or MODEL_ID)
        dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        tokenizer = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
        # Base checkpoints ship no pad token; scoring is batched, so we need
        # one. EOS as pad is safe here because every pad position is masked
        # out of both the attention mask and the log-prob sum below.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        config = AutoConfig.from_pretrained(target, trust_remote_code=True)
        quantized = getattr(config, 'quantization_config', None) is not None

        kwargs: dict = {'trust_remote_code': True}
        if quantized:
            # The checkpoint dictates its own dtype and placement; overriding
            # either is how you get silent dequantization or a dtype clash.
            kwargs['device_map'] = 'auto'

            # compressed-tensors defaults to DEQUANTIZING on load -- it expands
            # the fp8 weights back to bf16, so a 30B fp8 checkpoint that is
            # 38.7 GB on disk tries to occupy ~60 GB of VRAM and OOMs a 48 GB
            # card. Forcing compressed execution keeps it at 38.7 GB resident
            # (measured: loads in 29 s, 38.7 GB allocated). Without this the
            # fp8 build saves disk but no memory at all.
            method = _quant_method(config)
            if method == 'compressed-tensors':
                from transformers.utils.quantization_config import CompressedTensorsConfig
                try:
                    kwargs['quantization_config'] = CompressedTensorsConfig(run_compressed=True)
                except TypeError:
                    # Newer transformers spells the same thing as dequantize.
                    kwargs['quantization_config'] = CompressedTensorsConfig(dequantize=False)
        else:
            kwargs['dtype'] = torch.bfloat16 if dev.startswith('cuda') else torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(target, **kwargs)
        except TypeError:
            # transformers < 4.54 spells it torch_dtype.
            if 'dtype' in kwargs:
                kwargs['torch_dtype'] = kwargs.pop('dtype')
            model = AutoModelForCausalLM.from_pretrained(target, **kwargs)

        if 'device_map' not in kwargs:
            model.to(dev)
        model.eval()

        # With device_map the weights decide where they live; ask the model
        # rather than assuming, so scoring puts its inputs on the right device.
        _device = str(next(model.parameters()).device)
        _tokenizer, _model = tokenizer, model


def unload() -> None:
    """Drop the model and free GPU memory (used by the benchmark harness)."""
    global _model, _tokenizer, _device
    if _model is None:
        return
    import torch
    _model = None
    _tokenizer = None
    _device = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_sequences(
    texts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE
) -> List[Tuple[float, int]]:
    """
    Return (total_log_probability, scored_token_count) for each input string.

    The log-probability is summed over tokens, not averaged. For ranking
    candidate corrections that is the right quantity: each scored string is a
    complete alternative reading of the same sentence, so the comparison is
    "which of these strings is more likely under the model", and a mean would
    reward whichever candidate happens to tokenize into more pieces. The token
    count is returned alongside so callers that need a length-normalised view
    (e.g. comparing *different* sentences in the benchmark) can compute it.

    The first token of each sequence has no prediction context and is excluded
    from the sum; the tokenizer's BOS makes that a no-op for real content.
    """
    if not texts:
        return []

    load()
    import torch

    results: List[Tuple[float, int]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = _tokenizer(batch, return_tensors='pt', padding=True)
        enc = {k: v.to(_device) for k, v in enc.items()}

        with torch.inference_mode():
            logits = _model(**enc).logits

        # Shift: logits[t] predicts token[t+1].
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = enc['input_ids'][:, 1:]
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        mask = enc['attention_mask'][:, 1:].to(token_lp.dtype)
        totals = (token_lp * mask).sum(dim=-1)
        counts = mask.sum(dim=-1)

        results.extend(
            (float(t), int(c)) for t, c in zip(totals.tolist(), counts.tolist())
        )

    return results


def log_probs(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[float]:
    """score_sequences without the token counts."""
    return [lp for lp, _ in score_sequences(texts, batch_size=batch_size)]


def has_chat_template() -> bool:
    """
    True if the loaded checkpoint is instruction-tuned.

    This is the switch between the two ways of asking a model to repair a line:
    a base model (sarvam-1) only supports few-shot completion, while an
    instruct model (sarvam-30b) should be given the instruction through its own
    chat template -- feeding an instruct model raw few-shot text works far
    worse than either does properly.
    """
    load()
    return bool(getattr(_tokenizer, 'chat_template', None))


def chat(
    instruction: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    system: Optional[str] = None
) -> str:
    """
    Single-turn instruction through the model's own chat template.

    Raises if the checkpoint has no template rather than silently falling back
    to raw completion, so a caller can't quietly get base-model behaviour from
    a model it believed was instruction-tuned.
    """
    load()

    if not getattr(_tokenizer, 'chat_template', None):
        raise RuntimeError(
            f"{MODEL_ID} has no chat template -- it is a base model. "
            "Use complete() with few-shot conditioning instead."
        )

    messages = ([{'role': 'system', 'content': system}] if system else [])
    messages.append({'role': 'user', 'content': instruction})
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return complete(prompt, max_new_tokens=max_new_tokens, stop=None, temperature=temperature)


def complete(
    prompt: str,
    max_new_tokens: int = 256,
    stop: Optional[str] = '\n',
    temperature: float = 0.0
) -> str:
    """
    Greedy (or low-temperature) completion of `prompt`, truncated at `stop`.

    Greedy by default: this is a text-repair task where sampling diversity is
    pure downside -- an invented-but-fluent word is exactly the failure mode
    that silently corrupts a training corpus.
    """
    load()
    import torch

    enc = _tokenizer(prompt, return_tensors='pt')
    enc = {k: v.to(_device) for k, v in enc.items()}
    prompt_len = enc['input_ids'].shape[1]

    gen_kwargs = {
        'max_new_tokens': max_new_tokens,
        'pad_token_id': _tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs.update({'do_sample': True, 'temperature': temperature})
    else:
        gen_kwargs['do_sample'] = False

    with torch.inference_mode():
        out = _model.generate(**enc, **gen_kwargs)

    text = _tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    if stop and stop in text:
        text = text.split(stop)[0]
    return text.strip()
