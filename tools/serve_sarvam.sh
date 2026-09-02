#!/usr/bin/env bash
# Start the vLLM server that backs the sarvam-30b correction engines.
#
# Needed because transformers cannot run either published quantization of
# sarvam-30b as a quantized model -- compressed-tensors decompresses the whole
# model to bf16 on the first forward (~60 GB, OOMs a 48 GB card). vLLM has the
# real W4A16 kernels and registers SarvamMoEForCausalLM natively. See
# pipeline/correction/sarvam_vllm.py.
#
# vLLM lives in its own venv (venv-vllm) because it pins its own torch build;
# installing it into the pipeline's venv would replace the working one.
#
#   ./tools/serve_sarvam.sh                    # default: sarvam-30b W4A16
#   ./tools/serve_sarvam.sh <hf-repo-or-path>  # any vLLM-supported checkpoint
#
# Then point the pipeline at it (auto-detected if it is on the default port):
#   python cli.py page.jpg --engine hybrid
#
# Leave it running: it holds ~21 GB of weights resident, so CLI invocations and
# benchmark sweeps stop paying a model load each time.

set -euo pipefail

MODEL="${1:-mastersubhajit/sarvam-30b-AWQ-4bit}"
# 8000/8001 are commonly taken by other local services; 8077 keeps this out
# of their way. Override with SARVAM_VLLM_PORT (and SARVAM_VLLM_URL client-side).
PORT="${SARVAM_VLLM_PORT:-8077}"
# 0.85 of a 48 GB card leaves room for the OCR pass and the desktop session to
# share the GPU; the 4-bit weights themselves are only ~21 GB.
UTIL="${SARVAM_GPU_UTIL:-0.85}"
# The correctors score short context windows and rewrite single lines, so a
# large max-len would only reserve KV cache that never gets used.
MAXLEN="${SARVAM_MAX_LEN:-4096}"

cd "$(dirname "$0")/.."

if [ ! -x venv-vllm/bin/vllm ]; then
    echo "venv-vllm not found. Create it with:" >&2
    echo "  python3 -m venv venv-vllm && ./venv-vllm/bin/pip install vllm" >&2
    exit 1
fi

echo "Serving $MODEL on port $PORT (gpu-memory-utilization=$UTIL, max-model-len=$MAXLEN)"

# --trust-remote-code: the sarvam-30b repos declare an auto_map for their custom
# MoE classes, and vLLM refuses to even parse such a config without it. vLLM has
# its own native SarvamMoEForCausalLM implementation, so the repo's Python is
# used for the config only, not for the forward pass.
exec ./venv-vllm/bin/vllm serve "$MODEL" \
    --port "$PORT" \
    --trust-remote-code \
    --gpu-memory-utilization "$UTIL" \
    --max-model-len "$MAXLEN"
