#!/bin/bash
# build.sh <workdir>: vllm-xpu-kernels 0.1.15.4 + sycl-tla 87f68506 with the B70 patches, FA2 kernels only, AOT for
# BMG-G31 (Arc Pro B70) -> <workdir>/wheel. About 30 minutes; each compile job can take ~8 GB of RAM, so MAX_JOBS stays low.
#   VENV       the vLLM XPU venv (torch 2.14 XPU)          ONEAPI   oneAPI 2026.x root (default /opt/intel/oneapi)
#   MAX_JOBS   parallel compile jobs (default 1)
set -e
: "${VENV:?set VENV to the vLLM XPU venv}"
W=$(realpath -m "${1:?usage: build.sh <workdir>}")
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"
. "${ONEAPI:-/opt/intel/oneapi}/setvars.sh" --force >/dev/null 2>&1 || true
. "$VENV/bin/activate"

[ -d "$W/vllm-xpu-kernels" ] || git clone -q https://github.com/vllm-project/vllm-xpu-kernels "$W/vllm-xpu-kernels"
[ -d "$W/sycl-tla" ] || git clone -q https://github.com/intel/sycl-tla "$W/sycl-tla"
git -C "$W/vllm-xpu-kernels" checkout -q origin/release/0.1.15.4
git -C "$W/sycl-tla" checkout -q 87f6850680a580654b9ea2c80dbc01aeb36ad231
apply_once() { git -C "$1" apply --reverse --check "$2" 2>/dev/null || git -C "$1" apply "$2"; }
apply_once "$W/vllm-xpu-kernels" "$HERE/vllm-xpu-kernels-b70.patch"
apply_once "$W/sycl-tla" "$HERE/sycl-tla-e4m3-fast.patch"
cp "$HERE/cp_b70.conf" "$HERE/pd_b70.conf" "$W/vllm-xpu-kernels/csrc/xpu/attn/kernel_configs/"

cd "$W/vllm-xpu-kernels"
# B70_E4M3_FAST=3: 3-instruction fp8 -> fp16 convert; B70_DECODE_Q8_HEAD256 + B70_SPEC_ROW_MASK: packed MTP verify
export VLLM_CUTLASS_SRC_DIR="$W/sycl-tla" MAX_JOBS="${MAX_JOBS:-1}"
export CXXFLAGS="-DB70_E4M3_FAST=3 -DB70_DECODE_Q8_HEAD256=1 -DB70_SPEC_ROW_MASK=1"
export VLLM_XPU_AOT_DEVICES=bmg-g31-a0 VLLM_XPU_XE2_AOT_DEVICES=bmg-g31-a0 VLLM_XPU_ENABLE_XE3P=0
export BASIC_KERNELS_ENABLED=0 FA2_KERNELS_ENABLED=1 MOE_KERNELS_ENABLED=0 GDN_KERNELS_ENABLED=0 \
  MQA_LOGITS_KERNELS_ENABLED=0 MHC_KERNELS_ENABLED=0 XPU_SPECIFIC_KERNELS_ENABLED=0 XPUMEM_ALLOCATOR_ENABLED=0 \
  VLLM_XPU_ENABLE_ONEDNN=0
export VLLM_CHUNK_PREFILL_CONFIG=cp_b70.conf VLLM_PAGED_DECODE_CONFIG=pd_b70.conf VLLM_VERSION_OVERRIDE=0.1.15.4+b70
rm -rf "$W/wheel" build
pip wheel . --no-build-isolation --no-deps -w "$W/wheel"
echo "built: $(ls "$W"/wheel/*.whl)"
