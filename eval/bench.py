# bench.py <llama.cpp checkout> [url] [model...]: streaming single-stream decode t/s on prose (~24k prompt), code, and a whole-file edit.
# Prompts come from a llama.cpp checkout so they're reproducible; tok/s varies a few % run to run, so compare several runs.
import json, sys, time, urllib.request
LCPP = sys.argv[1]
URL = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000") + "/v1/chat/completions"
src = open(f"{LCPP}/tools/server/server-context.cpp").read()[:100000]
PROSE = ("Here is the source of a C++ inference server:\n\n" + src + "\n\nExplain in detail, in plain prose paragraphs "
         "with no code, how this server manages slots, the prompt cache, and speculative decoding. Be thorough.")
CODE = "Write a Python script that walks a directory tree, hashes every file with sha256, and prints groups of duplicate files. Include argparse and a main()."
f = open(f"{LCPP}/tools/server/tests/unit/test_stream.py").read()
EDIT = ("Here is a Python test file:\n\n```python\n" + f + "\n```\n\nRename the module-level variable `server` to `srv` everywhere "
        "in this file. Return the COMPLETE updated file in one code block and nothing else.")
def run(model, prompt, mt):
    body = json.dumps({"model": model, "max_tokens": mt, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    t0 = time.monotonic(); t1 = None; n = 0; usage = None
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    if t1 is None:
                        t1 = time.monotonic()
                    n += 1
    t2 = time.monotonic()
    ntok = (usage or {}).get("completion_tokens", n)
    ptok = (usage or {}).get("prompt_tokens", 0)
    return ptok, t1 - t0, (ntok - 1) / (t2 - t1), ntok
fmt = lambda r: "%5.1f t/s (ttft %4.1fs @%dtok, %d out)" % (r[2], r[1], r[0], r[3])
for m in sys.argv[3:] or ["qwen38"]:
    try:
        run(m, "hi", 3)
        p = [run(m, PROSE, 900) for _ in range(2)]
        c = run(m, CODE, 700)
        e = run(m, EDIT, 3000)
        print("%-9s prose: %s | %s\n          code: %s   edit: %s" % (m, fmt(p[0]), fmt(p[1]), fmt(c), fmt(e)), flush=True)
    except Exception as ex:
        print("%-9s FAILED: %s" % (m, str(ex)[:200]), flush=True)
