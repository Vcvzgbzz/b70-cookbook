# prefill_ab.py <tag> <approx-tokens...>: cold (nonce-busted) streamed prefill on $BASE_URL, prints TTFT and prefill tok/s
import glob, json, os, sys, sysconfig, time, urllib.request, uuid
URL = os.environ.get("BASE_URL", "http://localhost:8000/v1").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("MODEL", "qwen38")

src = ""
for f in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "**/*.py"), recursive=True)):
    src += open(f, errors="ignore").read()
    if len(src) > 2_000_000:
        break

tag = sys.argv[1]
for n in map(int, sys.argv[2:]):
    body = {"model": MODEL, "stream": True, "max_tokens": 1, "temperature": 0,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"[{uuid.uuid4()}]\n" + src[: n * 4] + "\n\nSay ok."}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t0 = time.time(); ttft = None; ptok = None
    with urllib.request.urlopen(req, timeout=1500) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if ttft is None and d.get("choices"):
                ttft = time.time() - t0
            if d.get("usage"):
                ptok = d["usage"]["prompt_tokens"]
    print(f"{tag} prompt {ptok:6d} tok  ttft {ttft:6.1f} s  {ptok / ttft:6.0f} tok/s", flush=True)
