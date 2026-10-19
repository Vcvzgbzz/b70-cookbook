#!/bin/bash
# serve.sh <model-dir> [extra vllm flags]: Qwen3.8-27B INT4 on one Arc Pro B70, 262144 context, DFlash2 drafter (7 drafts)
# when DRAFTER points at the Qwen3.8-27B-DFlash2 checkpoint, MTP5 otherwise.
# Needs a vLLM 0.31 XPU build with the base B70 MTP patches, patches/patch_draft_vocab_trim.py and patches/vllm/apply.sh.
#   VENV         vLLM XPU venv                     KERNELS   side package from patches/kernels/make-side-package.sh
#   DRAFTER      DFlash2 drafter dir (optional)
#   GPTQ_OVERRIDE=autoround-gptq-override.json loads an auto_round:auto_gptq checkpoint on CPUs without AVX-512.
set -e
: "${VENV:?set VENV to the vLLM XPU venv}"
: "${KERNELS:?set KERNELS to the rebuilt vllm_xpu_kernels side package}"
. "$VENV/bin/activate"
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH}" PYTHONPATH="$KERNELS"
export VLLM_TARGET_DEVICE=xpu ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Draft: int4 LM head scoring the first 98304 ids + specials. Checkpoints with a BF16 MTP head also want
# B70_MTP_BF16_DRAFT=1 B70_DRAFT_MTP_INT4=1; with an int4 MTP head leave both unset.
export B70_DRAFT_LMHEAD_INT4=1 B70_DRAFT_VOCAB="${B70_DRAFT_VOCAB:-98304}"
# Target: GDN metadata without host syncs, fp8 LM head, packed MTP verify, embeddings read from host RAM.
export B70_GDN_NO_MASK_SYNC=1 B70_TARGET_LMHEAD_FP8=1 B70_PACKED_ROWMASK=1 B70_EMBED_UVA=1
spec='{"method":"mtp","num_speculative_tokens":5}'
if [ -n "$DRAFTER" ]; then
  # DFlash2: drafter body int4 with bf16 activations (it overflows fp16) under the fp16 target, and the
  # sliding-window prefix-cache fix (without it a hybrid target reuses nothing from the prefix cache).
  export B70_DFLASH_DRAFT_INT4=1 B70_DRAFT_DTYPE=bfloat16 B70_SWA_FINE_GRAINED=1
  spec="{\"method\":\"dflash\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\"}"
fi
# The compile cache is keyed on flags, not env: give every env combination its own root.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm-b70-qwen38${DRAFTER:+-dflash2}}"
model=$1; shift
extra=()
[ -n "$GPTQ_OVERRIDE" ] && extra=(--quantization gptq --hf-overrides "$(cat "$GPTQ_OVERRIDE")")
exec vllm serve "$model" --served-model-name "${NAME:-qwen38}" --port "${PORT:-8000}" \
  --dtype float16 --max-model-len 262144 \
  --kv-cache-dtype fp8 --kv-cache-memory-bytes 13958643712 \
  --cpu-offload-gb 2 --cpu-offload-params visual \
  --max-num-seqs 4 --max-num-batched-tokens 3584 \
  --enable-prefix-caching --prefix-match-unit 64 --mamba-ssm-cache-dtype float16 \
  --enable-prompt-tokens-details \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --speculative-config "$spec" \
  "${extra[@]}" "$@"
