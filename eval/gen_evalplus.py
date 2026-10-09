# gen_evalplus.py <out-dir> [url] [model]: greedy non-thinking completions for HumanEval+ and MBPP+ -> <out-dir>/<ds>.jsonl
import sys, os, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from evalplus.data import get_human_eval_plus, get_mbpp_plus
out = sys.argv[1]; url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"; model = sys.argv[3] if len(sys.argv) > 3 else "qwen38"; os.makedirs(out, exist_ok=True)
PRE = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
def ask(prompt):
    b = json.dumps({"model": model, "temperature": 0, "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": f"{PRE}\n```\n{prompt.strip()}\n```"}]}).encode()
    for a in range(3):
        try:
            j = json.load(urllib.request.urlopen(urllib.request.Request(f"{url}/v1/chat/completions", b, {"Content-Type": "application/json"}), timeout=900))
            return j["choices"][0]["message"]["content"], j["usage"]["completion_tokens"]
        except Exception as e:
            err = e; time.sleep(5)
    return f"# error: {err}", 0
for ds, probs in (("humaneval", get_human_eval_plus()), ("mbpp", get_mbpp_plus())):
    t0 = time.time(); items = list(probs.items())
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda kv: ask(kv[1]["prompt"]), items))
    with open(f"{out}/{ds}.jsonl", "w") as f:
        for (tid, _), (txt, _) in zip(items, res): f.write(json.dumps({"task_id": tid, "solution": txt}) + "\n")
    print(ds, len(items), "done in %.0fs, %d completion tokens" % (time.time() - t0, sum(r[1] for r in res)), flush=True)
