# deep_decode.py <tag> <approx-tokens...>: decode t/s after a cached N-token prompt on $BASE_URL (first call fills the cache)
import glob, json, os, sys, sysconfig, time, urllib.request
URL = os.environ.get("BASE_URL", "http://localhost:8000/v1").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("MODEL", "qwen38")

src = ""
for f in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "**/*.py"), recursive=True)):
    src += open(f, errors="ignore").read()
    if len(src) > 2_000_000:
        break


def run(n):
    body = {"model": MODEL, "stream": True, "max_tokens": 500, "temperature": 0,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": src[: n * 4] + "\n\nExplain in detail what the last file above does, function by function."}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if first is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                first = time.time()
            if d.get("usage"):
                usage = d["usage"]
    end = time.time()
    return usage["prompt_tokens"], usage["completion_tokens"], first - t0, (usage["completion_tokens"] - 1) / (end - first)


tag = sys.argv[1]
for n in map(int, sys.argv[2:]):
    run(n)  # fill the prefix cache
    p, c, ttft, tps = run(n)
    print(f"{tag} deep {p:6d} tok: decode {tps:5.1f} t/s ({c} out, ttft {ttft:.1f}s)", flush=True)
