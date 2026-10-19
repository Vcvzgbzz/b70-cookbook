# kcheck.py <tag> <outfile>: fixed-seed attention outputs + timings on the B70 for stock vs rebuilt vllm_xpu_kernels.
# Run once per build (PYTHONPATH picks the build); compare the saved outputs with kcompare.py.
# Qwen3.8-27B shapes: Hq 24, Hkv 4, head 256, fp8 e4m3 paged KV, page 896.
import math, sys, torch
import vllm_xpu_kernels.flash_attn_interface as fai

tag, outfile = sys.argv[1], sys.argv[2]
D, HQ, HKV, BS = 256, 24, 4, 896
SCALE = 1 / math.sqrt(D)
dev = "xpu"
res = {}


def no_fallback(what):
    if fai._warned_missing_configs:
        print(f"ABORT {what}: kernel fallback {list(fai._warned_missing_configs)[0][:200]}", flush=True)
        sys.exit(2)


import os
DIST = os.environ.get("KCHECK_DIST", "codes")


def fp8_all_codes(shape):
    if DIST == "normal":
        return (torch.randn(shape, device=dev) * 8).to(torch.float8_e4m3fn)
    # every finite e4m3 code (0x7F/0xFF are NaN and never stored in the KV cache)
    b = torch.randint(0, 254, shape, dtype=torch.int32, device=dev)
    b = torch.where(b >= 0x7F, b + 1, b)
    return b.to(torch.uint8).view(torch.float8_e4m3fn)


def kv(B, L):
    nb = (L + BS - 1) // BS
    k = fp8_all_codes((B * nb + 1, BS, HKV, D))
    v = fp8_all_codes((B * nb + 1, BS, HKV, D))
    if os.environ.get("KCHECK_KV16"):  # fp16 KV, same values (descales applied up front)
        k, v = (k.to(torch.float16) * 0.0371), (v.to(torch.float16) * 0.0213)
    bt = torch.randperm(B * nb + 1, device=dev, generator=None)[: B * nb].view(B, nb).to(torch.int32)
    return k, v, bt


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    st, en = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    st.record()
    for _ in range(n):
        fn()
    en.record()
    torch.xpu.synchronize()
    return st.elapsed_time(en) / n * 1000


ks = torch.full((1,), 0.0371, device=dev, dtype=torch.float32)
vs = torch.full((1,), 0.0213, device=dev, dtype=torch.float32)
if os.environ.get("KCHECK_KV16"):
    ks = vs = None

# 1) chunked prefill: one 3584-token chunk on top of a long cached context (paged, causal)
for ctx in (() if os.environ.get("KCHECK_VERIFY_ONLY") else (8192, 32768, 65536, 110592)):
    torch.manual_seed(1)
    k, v, bt = kv(1, ctx)
    ql = 3584
    q = torch.randn(ql, HQ, D, device=dev, dtype=torch.float16)
    cuq = torch.tensor([0, ql], dtype=torch.int32, device=dev)
    seq = torch.tensor([ctx], dtype=torch.int32, device=dev)
    out = torch.empty_like(q)
    f = lambda: fai.flash_attn_varlen_func(q, k, v, ql, cuq, ctx, seqused_k=seq, causal=True, block_table=bt,
                                           softmax_scale=SCALE, fa_version=2, k_descale=ks, v_descale=vs, out=out)
    f(); torch.xpu.synchronize(); no_fallback(f"prefill ctx={ctx}")
    t = timeit(f, 10)
    res[f"prefill{ctx}"] = out.float().cpu()
    keys = ql * (ctx - ql) + ql * (ql + 1) // 2
    print(f"{tag} prefill chunk 3584 @ctx {ctx:6d}: {t:9.1f} us  {4 * D * HQ * keys / t / 1e6:6.1f} TFLOPS", flush=True)
    del k, v

# 2) q=1 decode and q=6 spec verify (stock per-position path) at depth
for ctx in (() if os.environ.get("KCHECK_PREFILL_ONLY") else tuple(int(c) for c in os.environ.get("KCHECK_VERIFY_CTX", "8192,65536").split(","))):
    torch.manual_seed(2)
    B = 2
    k, v, bt = kv(B, ctx)
    seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    q1 = torch.randn(B, HQ, D, device=dev, dtype=torch.float16)
    cu1 = torch.arange(B + 1, dtype=torch.int32, device=dev)
    o1 = torch.empty_like(q1)
    f1 = lambda: fai.flash_attn_varlen_func(q1, k, v, 1, cu1, ctx, seqused_k=seq, causal=True, block_table=bt,
                                            softmax_scale=SCALE, fa_version=2, k_descale=ks, v_descale=vs, out=o1)
    f1(); torch.xpu.synchronize(); no_fallback(f"decode ctx={ctx}")
    t1 = timeit(f1)
    res[f"decode{ctx}"] = o1.float().cpu()
    q6 = torch.randn(B * 6, HQ, D, device=dev, dtype=torch.float16)
    cu6 = torch.arange(0, B * 6 + 1, 6, dtype=torch.int32, device=dev)
    o6 = torch.empty_like(q6)
    f6 = lambda: fai.flash_attn_varlen_func(q6, k, v, 6, cu6, ctx, seqused_k=seq, causal=True, block_table=bt,
                                            softmax_scale=SCALE, fa_version=2, k_descale=ks, v_descale=vs, out=o6)
    f6(); torch.xpu.synchronize(); no_fallback(f"verify ctx={ctx}")
    t6 = timeit(f6)
    res[f"verify{ctx}"] = o6.float().cpu()
    print(f"{tag} B=2 decode q=1 @ctx {ctx:6d}: {t1:8.1f} us   verify q=6: {t6:8.1f} us", flush=True)
    del k, v

torch.save(res, outfile)
print(f"{tag} saved {outfile}", flush=True)
