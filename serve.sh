#!/bin/bash
# serve.sh <model-dir> [extra vllm flags]: Qwen3.8-27B INT4 + MTP5 on one Arc Pro B70.
# Needs a vLLM 0.31 XPU build with SergiioB's patches and patches/patch_draft_vocab_trim.py applied.
# GPTQ_OVERRIDE=autoround-gptq-override.json loads an auto_round:auto_gptq checkpoint on CPUs without AVX-512.
set -e
: "${VENV:?set VENV to the vLLM XPU venv}"
. "$VENV/bin/activate"
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH}"
export VLLM_TARGET_DEVICE=xpu ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export B70_DRAFT_LMHEAD_INT4=1 B70_DRAFT_VOCAB="${B70_DRAFT_VOCAB:-98304}"
# Checkpoints with a BF16 MTP head: B70_MTP_BF16_DRAFT=1 B70_DRAFT_MTP_INT4=1. With an int4 MTP head: leave both unset.
model=$1; shift
extra=()
[ -n "$GPTQ_OVERRIDE" ] && extra=(--quantization gptq --hf-overrides "$(cat "$GPTQ_OVERRIDE")")
exec vllm serve "$model" --served-model-name "${NAME:-qwen38}" --port "${PORT:-8000}" \
  --dtype float16 --max-model-len 131072 \
  --kv-cache-dtype fp8 --kv-cache-memory-bytes 6442450944 --kv-offloading-size 8 \
  --max-num-seqs 4 --max-num-batched-tokens 3584 \
  --enable-prefix-caching --prefix-match-unit 64 --mamba-ssm-cache-dtype float16 \
  --enable-prompt-tokens-details \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  "${extra[@]}" "$@"
