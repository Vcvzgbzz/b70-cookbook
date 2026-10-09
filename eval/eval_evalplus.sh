#!/bin/bash
# eval_evalplus.sh <out-dir>: sanitize and run generated code in a sandboxed systemd unit (no network, read-only FS except out-dir).
# EVALPLUS_VENV = venv with evalplus installed.
d=$(realpath "$1"); unit=evalplus-$(basename "$d")
mkdir -p "$HOME/.cache/evalplus"
systemd-run --wait --collect --unit="$unit" -p PrivateNetwork=yes -p ProtectSystem=strict -p ReadWritePaths="$d" -p ReadWritePaths="$HOME/.cache/evalplus" \
  -p PrivateTmp=yes -p MemoryMax=12G -p RuntimeMaxSec=3600 -p StandardOutput=file:"$d/exec.log" -p StandardError=file:"$d/exec.log" -E HOME="$HOME" \
  bash -c "for ds in humaneval mbpp; do $EVALPLUS_VENV/bin/evalplus.sanitize --samples $d/\$ds.jsonl && $EVALPLUS_VENV/bin/evalplus.evaluate --dataset \$ds --samples $d/\$ds-sanitized.jsonl --parallel 8 --i_just_wanna_run; done"
grep -E "humaneval|mbpp|pass@1" "$d/exec.log" | tail -12
