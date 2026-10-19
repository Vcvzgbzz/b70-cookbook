#!/bin/bash
# apply.sh <vllm-src>: our vLLM 0.31.0 changes on top of a tree that already has the base B70 MTP patches
# (patch_draft_lmhead_int4.py creates b70_draft_lmhead_int4.py, which draft-head-release.patch edits, and
# b70_draft_mtp_int4.py, whose int4 quantizer the DFlash2 drafter reuses).
set -e
V=$(realpath "${1:?usage: apply.sh <vllm-src>}")
HERE=$(cd "$(dirname "$0")" && pwd)
cp "$HERE/b70_target_lmhead_fp8.py" "$HERE/b70_embed_uva.py" "$HERE/b70_dflash_draft_int4.py" "$V/vllm/model_executor/models/"
for p in gdn-spec-no-sync qwen3_5-hooks draft-head-release swa-prefix-cache dflash2-drafter; do
  git -C "$V" apply --reverse --check "$HERE/$p.patch" 2>/dev/null && { echo "already applied $p"; continue; }
  git -C "$V" apply "$HERE/$p.patch" && echo "applied $p"
done
