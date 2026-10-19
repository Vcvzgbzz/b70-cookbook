#!/bin/bash
# make-side-package.sh <wheel> <dest>: a copy of the venv's installed vllm_xpu_kernels with the two rebuilt attention
# libraries swapped in and packed_verify.py appended, for PYTHONPATH=<dest>. The venv itself stays stock.
set -e
: "${VENV:?set VENV to the vLLM XPU venv}"
WHL=$(realpath "${1:?usage: make-side-package.sh <wheel> <dest>}"); DEST=$(realpath -m "${2:?usage: make-side-package.sh <wheel> <dest>}")
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$("$VENV/bin/python" -c "import vllm_xpu_kernels, os; print(os.path.dirname(vllm_xpu_kernels.__file__))")
TMP=$(mktemp -d)
"$VENV/bin/python" -m zipfile -e "$WHL" "$TMP"
rm -rf "$DEST" && mkdir -p "$DEST"
cp -a "$SRC" "$DEST/"
cp "$TMP/vllm_xpu_kernels/libattn_kernels_xe_2.so" "$TMP/vllm_xpu_kernels/_vllm_fa2_C.abi3.so" "$DEST/vllm_xpu_kernels/"
cat "$HERE/packed_verify.py" >> "$DEST/vllm_xpu_kernels/flash_attn_interface.py"
rm -rf "$DEST/vllm_xpu_kernels/__pycache__" "$TMP"
echo "side package: PYTHONPATH=$DEST"
