# arrivals.py <tag> [rounds]: one long decode (A) while three requests with fresh prompts arrive (prefill + MTP decode
# share steps). Reports A's decode t/s, each arrival's first-token time and decode t/s, and total tokens / wall.
import glob, json, os, sys, sysconfig, threading, time, urllib.request, uuid
URL = os.environ.get("BASE_URL", "http://localhost:8000/v1").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("MODEL", "qwen38")

src = ""
for f in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "**/*.py"), recursive=True)):
    src += open(f, errors="ignore").read()
    if len(src) > 400_000:
        break


def stream(content, n, res, key):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); tf = None; gen = 0
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                       {"Content-Type": "application/json"}), timeout=1200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                gen = d["usage"]["completion_tokens"]
            if tf is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                tf = time.time()
    res[key] = dict(ttft=tf - t0, tps=(gen - 1) / (time.time() - tf), gen=gen, end=time.time())


tag = sys.argv[1]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
for rd in range(rounds):
    res = {}
    A = "Write a complete Python module implementing a B-tree with insert, delete, search and range queries, with tests. Code only."
    jobs = [(0.0, "A", A, 1500)]
    for delay, key, chars, n in ((3.0, "B", 16_000, 300), (8.0, "C", 40_000, 300), (14.0, "D", 80_000, 300)):
        body = f"[{uuid.uuid4()}]\n" + src[:chars] + "\n\nSummarize what the code above does in 5 bullet points."
        jobs.append((delay, key, body, n))
    t0 = time.time(); th = []
    for delay, key, content, n in jobs:
        while time.time() - t0 < delay:
            time.sleep(0.05)
        t = threading.Thread(target=stream, args=(content, n, res, key)); t.start(); th.append(t)
    [t.join() for t in th]
    wall = max(r["end"] for r in res.values()) - t0
    total = sum(r["gen"] for r in res.values())
    print(f"{tag} round {rd + 1}: A decode {res['A']['tps']:.1f} t/s | " +
          " ".join(f"{k} ttft {res[k]['ttft']:.1f}s dec {res[k]['tps']:.1f}" for k in "BCD") +
          f" | total {total} tok / {wall:.1f}s = {total / wall:.1f} t/s", flush=True)
