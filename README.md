# b70-cookbook

Qwen3.8-27B on a single Intel Arc Pro B70, served by vLLM on XPU with MTP
speculative decoding. This is what I run day to day, plus the scripts I used to
decide what to run.

It builds on [SergiioB's B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook).
The engine patches that make MTP work on this card are theirs. What's here is
what I added on top, and the measurements behind it.

## Numbers

Single stream, 131k context, fp8 KV, MTP with 5 draft tokens, prefix caching on.

| prompt | tok/s |
|---|---|
| whole-file edit | ~142 |
| code from scratch | ~106 |
| prose, 24k-token prompt | ~54 |

Draft acceptance is about 52%. Code numbers move ±8% run to run because the
benchmark samples, so run it a few times before believing a difference.

Model is [devan-carlin/Qwen3.8-27B-int4-AutoRound](https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound) (G128)
INT4 checkpoint. [FINDINGS.md](FINDINGS.md) has the full comparison against
other quants and what didn't work.

## What's in here

- **`patches/patch_draft_vocab_trim.py`** — the draft LM head only scores the
  first 98,304 token ids plus specials. That covers 99.8% of the tokens in my
  code and docs, cuts the draft head's memory reads by about 60%, and is worth
  about +5%. The full-vocab target still verifies every token, so output doesn't
  change.
- **`autoround-gptq-override.json`** — vLLM's AutoRound loader dies with
  `Illegal instruction` on CPUs without AVX-512. AutoRound checkpoints packed as
  `auto_gptq` load fine as plain GPTQ with this override.
- **`serve.sh`** — the launch flags I run in production.
- **`eval/`** — compares a quant to a Q8_0 reference on identical token ids.
  More on why below.

## Setup

1. Build vLLM 0.31 for XPU and apply SergiioB's patches, following their README.
   You need `patch_draft_lmhead_int4.py` at minimum. For prefix caching on this
   hybrid model you also need `patch_fix_accepted_sync.py`,
   `patch_fix_backward_copy.py`, and `patch_fix_eagle_drop.py`. Without them,
   prefix caching with MTP silently corrupts state.
2. Apply the trim inside the same venv:
   ```bash
   python patches/patch_draft_vocab_trim.py
   ```
3. Serve:
   ```bash
   VENV=/path/to/vllm-venv GPTQ_OVERRIDE=autoround-gptq-override.json \
     ./serve.sh /path/to/qwen38-autoround
   ```
   devan's checkpoint ships an int4 MTP head, so `serve.sh` leaves
   `B70_MTP_BF16_DRAFT` and `B70_DRAFT_MTP_INT4` unset. For a checkpoint with a
   BF16 MTP head (cyankiwi AWQ, RedHat INT4), set both to `1`.

The first start takes about two minutes, most of it `torch.compile`. Set
`VLLM_CACHE_ROOT` to something per-config, or a changed flag will reuse a stale
compile cache.

## Measuring quant quality

Every INT4 candidate I tried tied on EvalPlus (HumanEval+ and MBPP+), within
four problems out of 542. They aren't the same, though: between AWQ-G32 and
AutoRound-G128, about 33 problems flip each way. EvalPlus is too coarse to rank 4-bit quants
of the same model.

What does separate them is comparing against a Q8_0 reference token by token:

- **Confident flips**: of the positions where Q8_0 puts >50% on its top token,
  how often does the candidate pick something else?
- **KL divergence** over Q8_0's top 20.
- **PPL gap**, scored on the exact same token ids. The GGUF and HF tokenizers
  differ slightly, so token ids have to come from one side.

```bash
# reference: Q8_0 logits through llama.cpp, reduced to top-20 per position
LLAMA_BIN=~/llama.cpp/build/bin ./eval/ref_run.sh Qwen3.8-27B-Q8_0.gguf code-corpus.txt ref-code.npz

# candidate: whatever vLLM is serving
python eval/score.py ref-code.npz score.json http://127.0.0.1:8000 qwen38

# sanity check that nothing broke outright
python eval/gen_evalplus.py out/ http://127.0.0.1:8000 qwen38
EVALPLUS_VENV=~/ev-venv ./eval/eval_evalplus.sh out/

# speed
python eval/bench.py ~/llama.cpp http://127.0.0.1:8000 qwen38
```

The corpus is 40 chunks of 2048 tokens of whatever text you care about. Mine was
mixed Python, Java, and TypeScript.

`eval/merge_layers.py` builds a checkpoint that's one quant with whole decoder
layers swapped in from another. I used it to test whether mixing G32 and G128
layers beats either one alone. It doesn't, and FINDINGS.md has the details.

## Hardware

Arc Pro B70 (32 GB), `xe` driver, vLLM 0.31.0, vllm-xpu-kernels 0.1.15.4,
torch 2.14 XPU. The host CPU has no AVX-512, which is the reason for the
override file.

## License

MIT. The patches this depends on are SergiioB's, also MIT.
