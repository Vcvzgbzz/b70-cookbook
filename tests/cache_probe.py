# cache_probe.py <tag> <approx-tokens...>: send the same prompt twice; report how many prompt tokens the second call
# reused from the prefix cache (usage.prompt_tokens_details.cached_tokens) and its time to first token
import glob, json, os, sys, sysconfig, time, urllib.request, uuid

URL = os.environ.get("BASE_URL", "http://localhost:8000/v1").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("MODEL", "qwen38")
src = ""
for f in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "**/*.py"), recursive=True)):
    src += open(f, errors="ignore").read()
    if len(src) > 2_000_000:
        break


def call(content):
    body = {"model": MODEL, "stream": True, "max_tokens": 32, "temperature": 0,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time(); ttft = None; usage = {}; text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if ttft is None and d.get("choices"):
                ttft = time.time() - t0
            for c in d.get("choices", []):
                text.append(c.get("delta", {}).get("content") or "")
            if d.get("usage"):
                usage = d["usage"]
    return ttft, usage.get("prompt_tokens", 0), (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0), "".join(text).strip()


tag = sys.argv[1]
for n in map(int, sys.argv[2:]):
    content = f"[{uuid.uuid4()}]\n" + src[: n * 4] + "\n\nName the first function defined above."
    t1, p, c1, a1 = call(content)
    t2, p, c2, a2 = call(content)
    print(f"{tag} cache probe {p:6d} tok: cold ttft {t1:6.1f}s | repeat ttft {t2:5.1f}s, reused {c2}/{p} "
          f"({100 * c2 / max(p, 1):.0f}%), recomputed {p - c2} | answers {'SAME' if a1 == a2 else 'DIFFER'}: {a1[:60]!r} / {a2[:60]!r}", flush=True)
