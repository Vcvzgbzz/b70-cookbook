# imgcheck.py <tag>: 12 fixed codes rendered as an image; greedy once + sampled 3x each on $BASE_URL
import base64, io, json, os, random, sys, urllib.request
from PIL import Image, ImageDraw, ImageFont
URL = os.environ.get("BASE_URL", "http://localhost:8000/v1").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("MODEL", "qwen38")

tag = sys.argv[1]
rng = random.Random(7)
words = ["KITE", "OPAL", "RUBY", "MINT", "JADE", "NOVA", "LYNX", "ONYX", "PLUM", "SAGE", "TEAL", "ZINC"]
codes = [f"{rng.randint(1000, 9999)}-{w}" for w in words]
font = ImageFont.load_default(size=56)


def ask(code, temp):
    img = Image.new("RGB", (520, 160), "white")
    ImageDraw.Draw(img).text((30, 45), code, fill="black", font=font)
    buf = io.BytesIO(); img.save(buf, "PNG")
    body = {"model": MODEL, "max_tokens": 40, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What code is written in this image? Reply with the code only."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}]}]}
    if temp is not None:
        body["temperature"] = temp
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"].strip()


greedy_bad, sampled_bad, misses = 0, 0, []
for c in codes:
    a = ask(c, 0)
    if c not in a:
        greedy_bad += 1; misses.append(f"greedy {c}->{a!r}")
    for _ in range(3):
        a = ask(c, None)
        if c not in a:
            sampled_bad += 1; misses.append(f"sampled {c}->{a!r}")
print(f"{tag} image: greedy misses {greedy_bad}/12, sampled misses {sampled_bad}/36", *misses, sep="\n  ", flush=True)
