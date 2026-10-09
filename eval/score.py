# score.py <ref.npz> <out.json> [url] [model]: score a served candidate against the Q8_0 reference on identical token ids
import sys, json, math, urllib.request, numpy as np
ref, out = sys.argv[1], sys.argv[2]; url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8000"; model = sys.argv[4] if len(sys.argv) > 4 else "qwen38"
R = np.load(ref); toks = R["tokens"]; n_ctx = int(R["n_ctx"]); first = n_ctx // 2; per = n_ctx - 1 - first
nll = agree = conf_n = conf_flip = n = 0; kld = []
for c in range(toks.shape[0]):
    chunk = [int(t) for t in toks[c]]
    b = json.dumps({"model": model, "prompt": chunk, "max_tokens": 1, "prompt_logprobs": 20, "temperature": 0}).encode()
    pl = json.load(urllib.request.urlopen(urllib.request.Request(f"{url}/v1/completions", b, {"Content-Type": "application/json"}), timeout=900))["choices"][0]["prompt_logprobs"]
    for i in range(per):
        p = first + i + 1; r = c * per + i; d = pl[p]
        lp = {int(k): v["logprob"] for k, v in d.items()}
        top1 = min(d.items(), key=lambda kv: kv[1]["rank"])[0]
        nll -= lp[chunk[p]]; n += 1
        rid, rlp = R["ids"][r], R["lps"][r]
        same = int(top1) == int(rid[0]); agree += same
        if math.exp(rlp[0]) > 0.5: conf_n += 1; conf_flip += not same
        floor = min(lp.values()); pr = np.exp(rlp)
        kld.append(float(np.sum(pr * (rlp - np.array([lp.get(int(t), floor) for t in rid])))))
    print(f"[{c + 1}] ppl {math.exp(nll / n):.4f} top1 {agree / n:.4f}", flush=True)
ref_ppl = float(np.exp(-R["act"].mean())); k = np.array(kld)
s = {"ref": ref, "n": n, "ppl": math.exp(nll / n), "ref_ppl": ref_ppl, "ppl_delta_pct": 100 * (math.exp(nll / n) / ref_ppl - 1),
     "top1_agree": agree / n, "confident_flip_rate": conf_flip / max(conf_n, 1), "confident_n": conf_n,
     "kld20_mean": float(k.mean()), "kld20_p99": float(np.percentile(k, 99)), "kld20_p999": float(np.percentile(k, 99.9))}
json.dump(s, open(out, "w"), indent=1); print(json.dumps(s))
