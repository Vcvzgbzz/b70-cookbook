# short_code.py [runs]: greedy decode t/s on 3 short code prompts (thinking off, 600 tokens) against $BASE_URL
import json, os, sys, time, urllib.request
URL = os.environ.get("BASE_URL", "http://localhost:8000/v1").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("MODEL", "qwen38")

PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H12M' into total seconds, with docstring and tests.",
    "Implement an LRU cache class in TypeScript with get, set, delete and a max size, using a Map. Include usage examples.",
    "Write a Rust function that reads a CSV file of (name, score) rows and prints the top 5 scores, handling errors with Result.",
]


def run(prompt):
    body = {"model": MODEL, "stream": True, "max_tokens": 600, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    first = None; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if first is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                first = time.time()
            if d.get("usage"):
                usage = d["usage"]
    return (usage["completion_tokens"] - 1) / (time.time() - first), usage["completion_tokens"]


runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
allv = []
for i, p in enumerate(PROMPTS):
    vals = [run(p) for _ in range(runs)]
    allv += [v for v, _ in vals]
    print(f"prompt {i + 1}: " + " ".join(f"{v:.1f}" for v, _ in vals) + f"  ({vals[0][1]} tok)", flush=True)
print(f"avg of {len(allv)}: {sum(allv) / len(allv):.1f} t/s")
