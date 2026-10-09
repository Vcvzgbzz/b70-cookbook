#!/bin/bash
# ref_run.sh <q8_0.gguf> <corpus.txt> <out.npz>: reference logits (top-20 per position) from a llama.cpp Q8_0 run.
# LLAMA_BIN = llama.cpp build dir with llama-perplexity; PY = python with numpy. Free the GPU first.
set -e
gguf=$1; corpus=$2; out=$3
fifo=$(mktemp -u).fifo; mkfifo "$fifo"; trap 'rm -f -- "$fifo"' EXIT
"${PY:-python3}" "$(dirname "$0")/ref_extract.py" "$fifo" "$out" 2048 &
"$LLAMA_BIN/llama-perplexity" -m "$gguf" -f "$corpus" -c 2048 --chunks "${CHUNKS:-40}" -ngl 99 -fa on --kl-divergence-base "$fifo" | grep -E "Final estimate"
wait
