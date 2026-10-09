# merge_layers.py <base-dir> <donor-dir> <out-dir> <donor layers, e.g. 24-31,56-63>: a compressed-tensors checkpoint with whole decoder
# layers swapped in from a second one (written for RedHat G128 base + cyankiwi AWQ-G32 donor). Whole-layer swaps keep AWQ smoothing consistent.
import sys, os, json, re
from safetensors import safe_open
from safetensors.torch import save_file
RH, AWQ, out = sys.argv[1], sys.argv[2], sys.argv[3]
spec = sys.argv[4] if len(sys.argv) > 4 else ""
awq = sorted({i for part in spec.split(",") if part for i in (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))})
os.makedirs(out, exist_ok=True)
for f in os.listdir(out):
    if f.endswith(".safetensors") or f.endswith(".json"): os.remove(os.path.join(out, f))
LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
def is_awq(k): m = LAYER.match(k); return bool(m) and int(m.group(1)) in awq
rh_map = json.load(open(f"{RH}/model.safetensors.index.json"))["weight_map"]
aw_map = json.load(open(f"{AWQ}/model.safetensors.index.json"))["weight_map"]
plan = [(k, RH, f) for k, f in rh_map.items() if not is_awq(k)] + [(k, AWQ, f) for k, f in aw_map.items() if is_awq(k)]
plan.sort(key=lambda x: (x[1], x[2], x[0]))
wm, shard, size, n, handles = {}, {}, 0, 0, {}
def flush():
    global shard, size, n
    if not shard: return
    n += 1; name = f"model-{n:05d}.safetensors"; save_file(shard, os.path.join(out, name), metadata={"format": "pt"})
    for k in shard: wm[k] = name
    shard, size = {}, 0
for k, d, f in plan:
    h = handles.get((d, f)) or handles.setdefault((d, f), safe_open(os.path.join(d, f), "pt"))
    t = h.get_tensor(k); shard[k] = t; size += t.numel() * t.element_size()
    if size > 2e9: flush()
flush()
json.dump({"metadata": {}, "weight_map": wm}, open(f"{out}/model.safetensors.index.json", "w"))
cfg = json.load(open(f"{RH}/config.json")); q = cfg["quantization_config"]
aq = json.load(open(f"{AWQ}/config.json"))["quantization_config"]
rh_w, aw_w = q["config_groups"]["group_0"]["weights"], aq["config_groups"]["group_0"]["weights"]
alt = lambda ls: "|".join(map(str, ls))
tgt = lambda ls: [rf"re:.*layers\.({alt(ls)})\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)$"]
rh_layers = [i for i in range(64) if i not in awq]
groups = {}
if rh_layers: groups["group_0"] = {"weights": rh_w, "input_activations": None, "output_activations": None, "targets": tgt(rh_layers)}
if awq: groups["group_1"] = {"weights": aw_w, "input_activations": None, "output_activations": None, "targets": tgt(awq)}
q["config_groups"] = groups
if 0 in awq: q["ignore"] = sorted(set(q["ignore"]) | {"model.language_model.layers.0.linear_attn.out_proj"})
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
for f in os.listdir(RH):
    if f.endswith(".json") and f not in ("config.json", "model.safetensors.index.json") or f.endswith(".jinja"):
        dst = os.path.join(out, f); os.path.lexists(dst) or os.symlink(os.path.join(RH, f), dst)
print("awq layers:", awq, "| tensors:", len(wm), "| shards:", n)
