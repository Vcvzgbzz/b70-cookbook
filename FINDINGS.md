# Findings

Everything below is from one Arc Pro B70 running the setup in the README. Each change was A/B'd against the config
before it in the same session, alternating arms, with the serving entry held so no other traffic touched the card.
"prose / code / edit" are `eval/bench.py`'s prompts; the rest are the scripts in `tests/`.

## Where the time goes

Decode is memory-bound. Per decoded step the model reads the int4 body (11.3 GiB), its G128 scales (0.44 GiB), the
LM head (1.19 GiB at fp8) and the draft's int4 head and MTP layer. Only ~0.05 GiB of the checkpoint is still bf16.
The decode GEMMs run at ~89% of the card's 608 GB/s, at a steady 2800 MHz with no throttle reasons set, so there is
little left in the matmuls themselves.

What grows with context is attention, and on this card most of attention's cost was not the math but converting the
fp8 KV cache to fp16: every subgroup converts every K and V tile it touches.

## What each change did

| change | effect | accuracy |
|---|---|---|
| draft LM head vocab trim (98,304 ids) | prose +2%, code +3%, edit +7%, deep 56k +6% | lossless |
| GDN metadata without host syncs | prose +6–10%, code +3%, edit +3.5%, deep flat | lossless |
| fp8→fp16 convert, 11 → 5 instructions | prefill attention −34%, MTP verify −24–30%; deep 56k +8%, first token at 119k −18% | bit-identical |
| fp8→fp16 convert, 5 → 3 instructions | prefill attention 58 → 70 TFLOPS; first token at 119k −8% | bit-identical |
| fp8 target LM head | head read per step 4.3 → 2.2 ms; code +8%, edit +5% | see below |
| embeddings to host RAM + vision tower offload | KV 9 → 13 GiB (~250k → ~373k tokens), context 204,800 → 262,144; speed unchanged | lossless |
| packed MTP verify, in-kernel row mask | verify attention −13% (32k) to −18% (110k); decode 55k +5%, 23k/110k +1.5% | bit-identical ≥32k, ~1e-6 below |

**The 3-instruction convert.** An e4m3 byte read as a *signed* byte and shifted left by 7 into a 16-bit word lands
its sign in bits 15 and 14 and its exponent and mantissa in bits 13–7. Clearing bit 14 leaves exactly the fp16 value
divided by 256, for every finite code including subnormals. The ×256 goes into the softmax and V scales the kernel
already applies once per tile. NaN codes aren't handled, and the KV cache never holds them. With it, fp8 KV prefill
runs at ~94% of the speed of fp16 KV, which needs twice the memory.

**Packed verify.** Stock MTP verify runs each draft position as its own decode query, so the KV cache is read once
per position. Packing all positions × GQA heads into one query and masking each row's newest keys inside the kernel
cuts that to one pass per 8-row tile. Two traps: the stock 16-row tile for head size 256 spills registers and is 4–5×
slower, and with split-K a split can contain only keys that are masked for an early row, which writes 0/0 unless the
epilogue treats a zero softmax sum as zero weight.

**fp8 LM head accuracy**, against a Q8_0 reference (code corpus):

| | PPL gap | confident flips | KLD | EvalPlus /542 |
|---|---|---|---|---|
| fp16 head | +1.590% | 1.263% | 0.02296 | 443 |
| fp8 head | +1.612% | 1.269% | 0.02345 | 444 |

## How many draft tokens

More draft tokens help output that's easy to predict and hurt output that isn't, and the verify cost of each extra
position grows with context.

| MTP n | short code (greedy) | prose 24k | edit | decode after 55k / 110k prompt |
|---|---|---|---|---|
| 4 | 119.1 | | | 84.1 / 76.3 |
| 5 | 124.6 | ~59 | ~149 | 86.4 / 75.4 |
| 6 | 127.6 | ~56 | ~163 | 86.8 / 76.1 |
| 7 | | ~52 | ~177 | |

Long prose at depth drafts worst of all, so it favors short drafts. Code and edits favor long ones. I run 5 as the
all-rounder.

## DFlash2 instead of MTP

[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) drafts a whole block of 7 tokens in
one pass from five of the target's hidden layers, then a small selector re-ranks each position's top-16
candidates. Stock, on this card, it ran into four problems. Each row below is the previous one plus one change,
all at the 262,144-token config.

| config | short code | code | edit | prose 24k | decode 23k / 55k / 110k | first token 33k / 126k |
|---|---|---|---|---|---|---|
| MTP 5 (all the changes above) | 124.6 | 128 | 158 | 62 | 87 / 86 / 75 | 17.8 s (30k) / 101.9 s (119k) |
| DFlash2, everything BF16, int4 drafter | 140.0 | 150.9 | 202.0 | 60.6 | 73.6 / 77.6 / 78.2 | 24.0 / 136.6 s |
| + FP16 target, BF16 drafter | 149.4 | 142.7 | 206.9 | 65.8 | 82.0 / 79.1 / 83.4 | 19.8 / 111.5 s |
| + int4 draft head for candidates | 160.3 | 152.5 | 216.0 | 67.6 | 85.7 / 81.2 / 87.0 | 19.8 / 111.4 s |

**It must not run in FP16.** The drafter was trained in BF16 and its activations overflow FP16: acceptance is
exactly 0%. vLLM gives the drafter the target's dtype, so the first working setup ran the whole 27B in BF16, which
costs the target its faster FP16 paths (the 3-instruction KV convert, the tuned int4 GEMMs). `B70_DRAFT_DTYPE`
builds the drafter alone in BF16 and casts where data crosses: aux hidden states and shared embeddings in, the
drafter's final hidden state out to the shared LM head. Values cross FP16 → BF16 exactly; the outgoing hidden state
is normalized and fits FP16.

**Prefix caching reused nothing.** The drafter's layers are all sliding-window (2,048 tokens). vLLM's sliding-window
cache manager only finds hits at whole-block granularity (896 tokens here), and one such group turns fine-grained
hits off for every group. The Gated DeltaNet state is then only checkpointed where prefill chunks happened to end,
so a repeated 24k-token prompt reused 90% and a repeated 61k-token prompt reused nothing. `swa-prefix-cache.patch` lets the sliding-window
group answer at 64-token granularity; all three now reuse 100% and answer in 0.2 s.

**The draft head was the biggest read.** Each step the drafter scored candidates against the full fp8 target head,
1.27 GB, more than the whole int4 drafter (0.93 GB). Picking them from the trimmed int4 draft head (0.26 GB) instead
lowers acceptance slightly (47.3 → 45.0% per draft token) and is faster everywhere.

**Seven drafts is the design point.** The drafter is trained for blocks of 8 (anchor + 7), and 8 verify rows fill the
packed-verify tile exactly; 8 drafts would spill into the 16-row tile. Six drafts accept more often (49.5%) but
produce fewer tokens per step (3.97 vs 4.17) and were slower on every test: short code −6%, edit −11%, 110k −5%.

Under load it also holds up: one long decode with three fresh 16k / 40k / 80k-token prompts arriving mid-stream
(`arrivals.py`) gives 74–75 tok/s total against MTP's 69–70, with the same first-token times.

The cost is KV capacity. The drafter's BF16 KV takes part of the same 13 GiB pool: ~289k tokens instead of ~373k.
A single 262,144-token request still fits.

## Quants against Q8_0

Code corpus, Q8_0 reference PPL 2.6828. "Flips" means positions where Q8_0 is >50% sure and the candidate picks a
different token.

| checkpoint | PPL gap | flips | EvalPlus /542 | code | edit | prose |
|---|---|---|---|---|---|---|
| cyankiwi AWQ-G32 | +0.94% | 0.74% | 446 | 102 | 128 | 52 |
| devan AutoRound-G128 | +1.59% | 1.25% | 443 | 113 | 142 | 53–56 |
| RedHat INT4 G128 | +2.27% | 1.38% | 442 | 113–116 | 145 | 52–58 |
| G128 + G32 on layers 24–31, 56–63 | +1.82% | 1.10% | 444 | 108.5 | 139 | 54 |

(Speeds from before the kernel work, relative to each other.) G128 is about 11% faster than G32 for 1.7–1.9× the
drift, regardless of who made it. Mixing whole layers lands exactly on the line between the two. With every large
matrix already int4, a faster checkpoint means fewer bits (G256, per-channel, 3-bit), which costs accuracy; the XPU
kernels only do int4 anyway.

## Loading AutoRound without AVX-512

vLLM's AutoRound path repacks weights on the CPU with AVX-512 and crashes with `Illegal instruction` without it.
Checkpoints saved as `auto_round:auto_gptq` are GPTQ underneath, so `--quantization gptq` plus
`autoround-gptq-override.json` loads them. None of the `dynamic` exclusion keys can contain `mtp`, or vLLM's MTP
module drops quantization for the whole MTP head and the weights won't load.

## Tried, didn't help

- **Packed verify with the newest keys handled in Python** (decode over the shared prefix + a small torch tail +
  log-sum-exp merge): correct, but the ~15 extra ops per layer cost more than the saved KV pass (−5% at 84k).
  Moving the mask into the kernel is what made packing pay.
- **Prefill tile shape**: a 64-key KV tile instead of 32 was flat with the old convert and worse (−5%) with the new
  one; three pipeline stages instead of two, flat.
- **Split-K count for packed verify under XPU graphs**: the kernel's own choice is within 1–2% of the best fixed
  count from 32k up; a fixed occupancy rule gains ~1% at short context and loses up to 20% at depth with 4 streams.
- **GDN without host syncs on mixed prefill + decode steps**: neutral on every test, including requests arriving
  mid-decode. Those steps are dominated by the prefill chunk.
- **MTP n7**: +8% on edits, −5% at 110k, nothing on code.
- **vllm-xpu-kernels PR #608** (split-K across head dims in paged decode): correct, gain within noise.
- **vllm-xpu-kernels 0.1.16, vLLM 0.31.0 final vs rc2**: flat for this setup. 0.31 dropped
  `--enable-mamba-fine-grained-prefix-cache`, so remove that flag when upgrading.
- **Adaptive verification** (vLLM trims each request's verify length on device by predicted acceptance): vLLM
  refuses it for this model, because the Gated DeltaNet backend can't take per-request verify lengths decided on
  the GPU.
- **Drafter KV in fp8**: KV capacity ~289k → ~313k tokens, but decode at 23k −8% and code −2 to −9% (the drafter's
  BF16 attention converts every tile back), and acceptance drops slightly.
- **GDN state in FP32** (`--mamba-ssm-cache-dtype float32`): over a 60k-token document, per-token log-likelihood
  matches FP16 state to four digits at every depth (0.41876 vs 0.41887 mean NLL; top-1 differs on 0.2–0.3% of
  positions with no growth with depth), while the doubled state checkpoints cost 17% of KV capacity and a single
  262,144-token request no longer fits in 13 GiB.
- **8192-token prefill chunks**: first token at 33k/126k 1.5%/3% sooner, decode unchanged, 4k tokens less KV.
- **A bigger KV pool** (+0.75 GiB, ~289k → ~305k tokens): same speed, but under concurrent long prompts plus images
  the allocator grew into all remaining memory. The default leaves ~1.2 GiB for that.
- **Greedy drafting under sampling**: at the model's default sampling (T=1.0, top-k 20, top-p 0.95) DFlash2
  acceptance is only 1–2 points below greedy decoding, so a greedy draft has nothing to win back.
- **Tree verification** (verify a tree of drafts instead of one chain): measured offline on DFlash2's top-k
  candidates, a 16/32/64-node tree accepts 3–15% / 4–26% / 5–33% more tokens per step than the 7-token chain
  (prose gains most, edits least). Verifying 2–9× more rows per step eats most of that at depth: estimated net
  0–8%. Neither upstream attempt (vLLM PRs #42910, #29846) fits this setup: one needs the removed Triton
  tree-attention backend and greedy-only verify, the other patches only the Triton GDN ops, while XPU runs the
  fused SYCL GDN op. Not worth a port.
- **CPU KV offload tier** (`--kv-offloading-size`): on vLLM 0.31 with this hybrid model it stores blocks but never
  finds them again. An evicted 64k-token prompt reports 0 cached tokens and re-prefills (44.5 s), with MTP or
  DFlash2. The flag only holds host RAM, so `serve.sh` leaves it out.
- **Clocks and power**: `xe` holds 2800 MHz under load with no throttle reasons; there's no Linux-side overclock or
  power-cap knob that helps.

## Testing kernels without taking the box down

- **Check for the reference fallback.** When vllm-xpu-kernels has no compiled kernel for a shape, it warns once and
  falls back to a PyTorch reference implementation. At long context that fallback allocates host memory until the
  machine runs out. Abort any test the moment `fai._warned_missing_configs` is non-empty (`tests/kcheck.py` does),
  and run GPU experiments under a memory cap (`systemd-run -p MemoryMax=...`).
- **Exactness first.** `tests/kcheck.py` runs prefill, decode and verify on fixed seeds with both every finite fp8
  code and normally distributed KV (the all-codes data makes softmax collapse onto one key, which hides
  differences), and `kcompare.py` reports bit-identity per case.
- **Hold the serving entry during A/Bs** so a real request can't load a second model onto a card that's already full.

## llama.cpp, for comparison

Before vLLM I ran this model on llama.cpp with SYCL:

- **Build with `-DGGML_SYCL_F16=ON`.** Without it, prefill GEMMs run in FP32 and never touch the XMX engines. Q4_K_M
  prefill at ~12k tokens went from 324 to 954 tok/s; decode didn't change.
- **Vulkan vs SYCL**: Vulkan prefills faster (+49 to +88%), but SYCL wins decode, and MTP on Vulkan degrades with two
  streams.

vLLM beat llama.cpp on every decode prompt (+28 to +42% before any of the work above) and on cold prefill.
llama.cpp restarts in ~18 s against vLLM's ~2 minutes.
