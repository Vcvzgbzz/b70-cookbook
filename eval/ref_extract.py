# ref_extract.py <fifo> <out.npz> <n_ctx>: reduce a llama-perplexity --kl-divergence-base stream to per-position top-20 + actual-token logprobs
import sys, numpy as np
src, out, n_ctx = sys.argv[1], sys.argv[2], int(sys.argv[3])
f = open(src, "rb", buffering=1 << 24)
assert f.read(8) == b"_logits_"; hdr_ctx, n_vocab, n_chunk = (int(x) for x in np.frombuffer(f.read(12), np.int32)); assert hdr_ctx == n_ctx, hdr_ctx
toks = np.frombuffer(f.read(4 * n_chunk * n_ctx), np.int32).reshape(n_chunk, n_ctx)
nv = 2 * ((n_vocab + 1) // 2) + 4
first = n_ctx // 2; per = n_ctx - 1 - first
K = 20; N = n_chunk * per
ids = np.zeros((N, K), np.int32); lps = np.zeros((N, K), np.float32); act = np.zeros(N, np.float32)
for c in range(n_chunk):
    for i in range(per):
        rec = np.frombuffer(f.read(2 * nv), np.uint16)
        scale, minlp = rec[:4].view(np.float32)
        q = rec[4:4 + n_vocab].astype(np.float32)
        top = np.argpartition(-q, K)[:K]; top = top[np.argsort(-q[top])]
        r = c * per + i
        ids[r] = top; lps[r] = minlp + scale * q[top]
        act[r] = minlp + scale * q[toks[c, first + i + 1]]
    print("chunk", c + 1, "ppl so far %.4f" % np.exp(-act[: (c + 1) * per].mean()), flush=True)
np.savez(out, tokens=toks, ids=ids, lps=lps, act=act, n_ctx=n_ctx)
print("DONE", out, "PPL %.4f" % np.exp(-act.mean()), "top1 mean p %.3f" % np.exp(lps[:, 0]).mean())
