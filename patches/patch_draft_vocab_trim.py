#!/usr/bin/env python3
"""Trim the int4 draft LM head to the first B70_DRAFT_VOCAB token ids plus specials.

Applies on top of SergiioB's patch_draft_lmhead_int4.py (edits the helper it writes).
The draft proposes from the trimmed vocab; trimmed ids get -inf, and the full-vocab
target still verifies every token, so output is unchanged. Unset or 0 = original path.
"""
import os
import sys

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    import vllm
    path = os.path.join(os.path.dirname(vllm.__file__), "model_executor", "models", "b70_draft_lmhead_int4.py")
src = open(path).read()
if "B70_DRAFT_VOCAB" in src:
    print("already applied:", path)
    sys.exit(0)

QUANT = """    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(
        weight.detach()
    )
"""
QUANT_NEW = """    w = weight.detach()
    keep = int(os.environ.get("B70_DRAFT_VOCAB", "0"))
    if 0 < keep < w.shape[0]:
        # First `keep` ids plus the added/special block (Qwen3.x: 248044+), padded to a multiple of 256.
        org = getattr(getattr(model, "logits_processor", None), "org_vocab_size", w.shape[0])
        hi_start = int(os.environ.get("B70_DRAFT_SPECIALS_START", "248044"))
        hi_start = min(hi_start, org)
        lo = keep + (-(keep + org - hi_start)) % 256
        idx = torch.cat([torch.arange(lo), torch.arange(hi_start, org)]).to(w.device)
        model._b70_lmhead_idx = (idx, org)
        w = w.index_select(0, idx)
        print(f"[B70] draft LM head vocab trim: {w.shape[0]} of {org} rows", flush=True)
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(w)
"""
LOGITS = """    logits = int4_lmhead_logits(
        hidden_states, qweight, scales, qzeros, group_size
    )
"""
LOGITS_NEW = LOGITS + """    trim = getattr(model, "_b70_lmhead_idx", None)
    if trim is not None:
        idx, org = trim
        full = logits.new_full((*logits.shape[:-1], org), float("-inf"))
        full[..., idx] = logits
        return full
"""
for old, new in ((QUANT, QUANT_NEW), (LOGITS, LOGITS_NEW)):
    if src.count(old) != 1:
        sys.exit(f"anchor not found once in {path}; is patch_draft_lmhead_int4.py applied?")
    src = src.replace(old, new)
open(path, "w").write(src)
print("patched:", path)
