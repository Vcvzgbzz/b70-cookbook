# Findings

Everything below is from one B70 running the setup in the README. Speeds are
single-stream tok/s on `eval/bench.py`'s three prompts plus a 56k-token prompt
("deep") where noted.

## Where the time goes

During decode the GPU is 100% busy, so host overhead isn't worth chasing. At
~48 ms per step, the GEMMs take 83% of the time and read about 22 GB per step.
That's ~543 GB/s against a 608 GB/s peak, so the matmuls are close to the floor.

Per-step bytes: int4 target body 14.2 GB, target LM head (fp16) 2.5 GB, draft LM
head ×5 about 4 GB before the trim, draft MTP layer ×5 about 1.3 GB.

The draft LM head was the cheapest thing left to cut, which is what the vocab
trim does.

Flash attention is the other cost that grows with depth: ~25 µs per call at
short context, 156 to 204 µs per call at 24k, 21 calls per step.

## Draft vocab trim

AWQ-G32 checkpoint, vLLM 0.30, MTP 5:

| | prose | code | edit | deep 56k |
|---|---|---|---|---|
| full vocab | 47.3 | 88.7 | 120.4 | 49.7 |
| first 98,304 ids | 48.2 | 91.1 | 128.6 | 52.8 |

Acceptance didn't change. Trimming to 65,536 was no better.

## How many draft tokens

Same checkpoint, trim on:

| MTP n | prose | code | edit | deep 56k | acceptance |
|---|---|---|---|---|---|
| 5 | 50.3 | 89.8 | 128.4 | 53.2 | 49% |
| 6 | 50.0 | 106.8 | 142.2 | 49.8 | 45% |
| 7 | 48.2 | 108.6 | 152.0 | 50.5 | 41% |

More draft tokens help code and edits, where the next tokens are predictable,
and hurt long-context prose. I run 5, the best of the three at depth.

## Quants against Q8_0

Code corpus, Q8_0 reference PPL 2.6828. "Flips" means positions where Q8_0 is
>50% sure and the candidate picks a different token.

| checkpoint | PPL gap | flips | EvalPlus /542 | code | edit | prose |
|---|---|---|---|---|---|---|
| cyankiwi AWQ-G32 | +0.94% | 0.74% | 446 | 102 | 128 | 52 |
| devan AutoRound-G128 | +1.59% | 1.25% | 443 | 113 | 142 | 53–56 |
| RedHat INT4 G128 | +2.27% | 1.38% | 442 | 113–116 | 145 | 52–58 |
| G128 + G32 on layers 24–31, 56–63 | +1.82% | 1.10% | 444 | 108.5 | 139 | 54 |

G128 is about 11% faster than G32 for 1.7 to 1.9x the drift, regardless of who
made it. EvalPlus can't see any of this.

The last row mixes the two. A per-layer sweep showed layers 56–63 carry about a
quarter of the drift, so I moved those plus 24–31 back to G32. It landed exactly
on the line between the other two: it recovered 45% of the accuracy and gave up
45% of the speed. Mixing layers doesn't get you anything free.

I run devan's checkpoint. It's faster than the hybrid and drifts less than
RedHat at the same speed.

## Loading AutoRound without AVX-512

vLLM's AutoRound path repacks weights on the CPU with AVX-512 and crashes with
`Illegal instruction` if the CPU doesn't have it. Checkpoints saved as
`auto_round:auto_gptq` are GPTQ underneath, so `--quantization gptq` plus
`autoround-gptq-override.json` loads them.

Gotcha: none of the `dynamic` exclusion keys can contain the string `mtp`. If
one does, vLLM's MTP module drops quantization for the whole MTP head and the
weights won't load.

## Tried, didn't help

- **vllm-xpu-kernels PR #608** (split-K across head dims in paged decode): the
  build is correct and passes checks. deep went 49.8 → 50.6, which is noise.
  Building it needed an FA2-only kernel build because the full build runs out
  of memory at 16 GB.
- **vLLM 0.31.0 final vs 0.31.0rc2**: flat. Everything that matters for XPU was
  already in rc2. 0.31 also dropped `--enable-mamba-fine-grained-prefix-cache`,
  so remove that flag when upgrading.
- **Overclocking from Linux**: `xe` clamps the core at 2800 MHz and rejects
  anything higher. Arc Control in a Windows VM shows the full set of sliders,
  but none of the settings survive a reboot into Linux.

## llama.cpp, for comparison

Before vLLM I ran this model on llama.cpp with SYCL. Two things worth knowing if
you're still there:

- **Build with `-DGGML_SYCL_F16=ON`.** Without it, prefill GEMMs run in FP32 and
  never touch the XMX engines. Q4_K_M prefill at ~12k tokens went from 324 to
  954 tok/s, and decode didn't change.
- **Vulkan vs SYCL**: Vulkan prefills faster (+49 to +88%), but SYCL wins
  decode, and MTP on Vulkan degrades with two streams (30 vs 44 tok/s combined). I stayed on SYCL for
  decode.

vLLM beat llama.cpp on every decode prompt (+28 to +42% in the first run) and
cold prefill. llama.cpp restarts in ~18 s against vLLM's ~2 minutes, and its
prompt cache replays faster.
