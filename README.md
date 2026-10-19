# b70-cookbook

Qwen3.8-27B on a single Intel Arc Pro B70, served by vLLM on XPU with DFlash2 speculative decoding at the model's
full 262,144-token context. This is what I run day to day, the patches that got it there, and the tests I used to decide
what to keep.

## Numbers

Single stream, fp8 KV, prefix caching on. Measured with the scripts in `tests/`; MTP is the same stack with the
model's own MTP head instead of the DFlash2 drafter.

| test | DFlash2, 7 drafts | MTP, 5 drafts |
|---|---|---|
| short code generation, 3 prompts, greedy (`short_code.py`) | 160.3 tok/s | 124.6 tok/s |
| code / edit / prose 24k (`eval/bench.py`) | 152 / 216 / 68 tok/s | 128 / 158 / 62 tok/s |
| decode after a cached 23k / 55k / 110k-token prompt (`deep_decode.py`) | 86 / 81 / 87 tok/s | 87 / 86 / 75 tok/s |
| time to first token, cold (`prefill_ab.py`) | 19.8 s at 33k / 111.4 s at 126k | 17.8 s at 30k / 101.9 s at 119k |
| repeated 25k-token prompt, time to first token (`cache_probe.py`) | 0.2 s (100% reused) | 0.2 s |
| four overlapping requests, total (`arrivals.py`) | 74–75 tok/s | 69–70 tok/s |
| context per request / KV cache | 262,144 / ~289k tokens | 262,144 / ~373k tokens |

Everything except the fp8 LM head is lossless: the kernel changes produce output bit-identical to stock, the
drafter only decides what gets proposed, and the fp8 head moves the model by about as much as noise (see
[FINDINGS.md](FINDINGS.md)). With greedy decoding, the DFlash2 and MTP setups give the same answer to a
250k-token prompt, word for word.

Model is [devan-carlin/Qwen3.8-27B-int4-AutoRound](https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound),
an INT4 G128 checkpoint. Drafter is [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
a 5-layer block-diffusion drafter trained for blocks of 8 (one anchor + 7 drafts).

## What's in here

**vLLM patches** (`patches/`, applied to a vLLM 0.31.0 tree)

- `patch_draft_vocab_trim.py`: the draft LM head scores only the first 98,304 token ids plus specials, about 60% fewer
  draft-head reads. The full-vocab target still verifies every token.
- `vllm/gdn-spec-no-sync.patch`: on MTP decode steps the Gated DeltaNet metadata builder gathered state indices with a
  host-side bool mask, a blocking copy that drains the GPU queue twice per step. Padded rows always trail the real
  ones, so a slice gives the same values without the sync.
- `vllm/b70_target_lmhead_fp8.py` + `qwen3_5-hooks.patch` + `draft-head-release.patch`: the 2.5 GB fp16 target LM
  head becomes per-row fp8 e4m3 at load (half the bytes read per step), and the fp16 copy is released once the
  int4 draft head has been built from it.
- `vllm/b70_embed_uva.py` (hooked by `qwen3_5-hooks.patch`): the input embedding table moves to pinned host RAM and
  the GPU reads rows in place. With `--cpu-offload-params visual` for the vision tower, that frees ~3.2 GiB for KV.
- `vllm/swa-prefix-cache.patch`: the DFlash2 drafter's sliding-window layers answered prefix-cache lookups only at
  whole-block granularity, which disables fine-grained hits for the whole hybrid model and with them the Gated
  DeltaNet prompt-tail checkpoints, so a repeated prompt reused nothing. Sliding-window groups now find hits at
  64-token granularity (the cached prompt-tail block plus the window's full blocks) and cache the partial tail.
- `vllm/dflash2-drafter.patch` + `vllm/b70_dflash_draft_int4.py`: runs the drafter in BF16 under the FP16 target
  (`B70_DRAFT_DTYPE`, casting at the three places data crosses between them), quantizes its linears to int4 with
  BF16 activations (`B70_DFLASH_DRAFT_INT4`), and picks its candidates from the trimmed int4 draft head instead of
  the full target head (`B70_DRAFT_LMHEAD_INT4`).

**Kernel patches** (`patches/kernels/`, vllm-xpu-kernels 0.1.15.4 + sycl-tla)

- `sycl-tla-e4m3-fast.patch` + the scale fold in `vllm-xpu-kernels-b70.patch`: fp8 KV is converted to fp16 inside
  the attention kernels, by every subgroup, for every tile. Stock does it in 11 instructions per 64 values with NaN
  handling; this does it in 3 (sign-extending shift, one mask) and folds the leftover ×256 into the softmax and V
  scales. Same output, prefill attention 39 → 70 TFLOPS. The BF16 variant (5 instructions, ×2^120 applied exactly)
  serves the drafter's BF16 KV.
- Packed MTP verify, in `vllm-xpu-kernels-b70.patch` + `packed_verify.py`: verify runs every draft position × GQA head
  of a sequence as rows of one decode query, with each row's causal limit applied inside the decode kernel. Also an
  8-row decode tile for that head-256 group (the 16-row tile spills) and a guard for splits where a row sees no keys.
- `build.sh`, `make-side-package.sh`, `cp_b70.conf`, `pd_b70.conf`: build only the kernels this model and its
  drafter use (~30 min) and install them as a side package on `PYTHONPATH`, so the venv stays stock.

**Serving and tests**

- `serve.sh`: the launch flags and environment I run in production.
- `autoround-gptq-override.json`: loads `auto_round:auto_gptq` checkpoints as plain GPTQ, for CPUs without AVX-512.
- `tests/`: speed (`short_code.py`, `deep_decode.py`, `prefill_ab.py`, `arrivals.py`), prefix-cache reuse
  (`cache_probe.py`, plus `swa_fine_test.py` for the sliding-window lookup), a vision check (`imgcheck.py`) and a
  kernel exactness check against stock (`kcheck.py`, `kcompare.py`).
- `eval/`: quant quality against a Q8_0 reference, token by token.

## Setup

1. Build vLLM 0.31 for XPU and apply the base B70 MTP patches from
   [SergiioB's B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook). You need
   `patch_draft_lmhead_int4.py` at minimum; for prefix caching with MTP on this hybrid model also
   `patch_fix_accepted_sync.py`, `patch_fix_backward_copy.py` and `patch_fix_eagle_drop.py`. Without those, prefix
   caching with MTP silently corrupts state.
2. Apply ours on top:
   ```bash
   python patches/patch_draft_vocab_trim.py        # inside the vLLM venv
   ./patches/vllm/apply.sh /path/to/vllm-src
   ```
3. Build the kernels and make the side package:
   ```bash
   VENV=/path/to/vllm-venv ./patches/kernels/build.sh ~/xk-build
   VENV=/path/to/vllm-venv ./patches/kernels/make-side-package.sh ~/xk-build/wheel/*.whl ~/xk-b70
   ```
4. Check the kernels against stock before serving with them (prints how many cases are bit-identical):
   ```bash
   KCHECK_DIST=normal python tests/kcheck.py stock stock.pt
   KCHECK_DIST=normal PYTHONPATH=~/xk-b70 python tests/kcheck.py b70 b70.pt
   python tests/kcompare.py b70.pt stock.pt
   ```
5. Serve, with the drafter (leave `DRAFTER` unset for MTP):
   ```bash
   huggingface-cli download incoai/Qwen3.8-27B-DFlash2 --local-dir ~/models/qwen38-dflash2
   VENV=/path/to/vllm-venv KERNELS=~/xk-b70 GPTQ_OVERRIDE=autoround-gptq-override.json \
     DRAFTER=~/models/qwen38-dflash2 ./serve.sh /path/to/qwen38-autoround
   ```

The first start takes about two and a half minutes, most of it `torch.compile`. vLLM keys its compile cache on flags,
not environment, so give every environment combination its own `VLLM_CACHE_ROOT` or a changed switch will replay a
stale graph.

## Measuring quant quality

Every INT4 candidate I tried tied on EvalPlus (HumanEval+ and MBPP+), within four problems out of 542, yet about 33
problems flip each way between two of them. EvalPlus is too coarse to rank 4-bit quants of the same model. Comparing
against a Q8_0 reference token by token does separate them:

- **Confident flips**: of the positions where Q8_0 puts >50% on its top token, how often the candidate picks
  something else.
- **KL divergence** over Q8_0's top 20.
- **PPL gap**, scored on the exact same token ids (the GGUF and HF tokenizers differ slightly).

```bash
LLAMA_BIN=~/llama.cpp/build/bin ./eval/ref_run.sh Qwen3.8-27B-Q8_0.gguf code-corpus.txt ref-code.npz
python eval/score.py ref-code.npz score.json http://127.0.0.1:8000 qwen38
python eval/gen_evalplus.py out/ http://127.0.0.1:8000 qwen38
EVALPLUS_VENV=~/ev-venv ./eval/eval_evalplus.sh out/
```

The corpus is 40 chunks of 2048 tokens of whatever text you care about. `eval/merge_layers.py` builds a checkpoint
that mixes whole decoder layers from two quants.

## Hardware

Arc Pro B70 (32 GB, BMG-G31), `xe` driver, vLLM 0.31.0, vllm-xpu-kernels 0.1.15.4, torch 2.14 XPU, oneAPI 2026.1.
The kernel build is AOT for BMG-G31 only; for an Arc Pro B60 (BMG-G21), add `bmg-g21-a0` to both AOT device lists in
`build.sh`. The host CPU has no AVX-512, which is the reason for the override file.

## License

MIT. The base MTP patches this builds on are SergiioB's, also MIT. The kernel patches modify vllm-xpu-kernels
(Apache-2.0) and sycl-tla (BSD-3-Clause) and are under those licenses.
