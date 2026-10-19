"""B70: DFlash drafter linears as int4 g128 sym (round-to-nearest) through int4_gemm_w4a16 (gate B70_DFLASH_DRAFT_INT4=1).

The drafter's weights are read once per decode step, so halving them again (bf16 3.85 GB -> ~1 GB) cuts per-step
bytes on a memory-bound card. Activations and scales stay in the model dtype: the drafter is BF16-trained and its
activations overflow FP16. Output stays lossless; only draft acceptance can move.
"""
from __future__ import annotations

import os

import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.b70_draft_mtp_int4 import quantize_to_int4

_GROUP = 128


class _Int4LinearMethod:
    """Stands in for an unquantized linear's quant method; apply() runs int4_gemm_w4a16."""

    def __init__(self, qweight, scales, qzeros):
        self.qweight, self.scales, self.qzeros = qweight, scales, qzeros

    def create_weights(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, *args, **kwargs):
        pass

    def apply(self, layer, x, bias=None):
        flat = x.reshape(-1, x.shape[-1])
        out = torch.ops._xpu_C.int4_gemm_w4a16(flat, self.qweight, None, self.scales, self.qzeros, _GROUP, None)
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], self.qweight.shape[1])


@torch.no_grad()
def quantize_dflash_drafter_int4(model) -> None:
    """Swap every eligible linear in the drafter body to int4; no-op without the gate."""
    if os.environ.get("B70_DFLASH_DRAFT_INT4") != "1" or getattr(model, "_b70_int4_done", False):
        return
    model._b70_int4_done = True
    body = getattr(model, "model", model)
    before = after = count = 0
    for name, mod in body.named_modules():
        w = getattr(mod, "weight", None)
        if not isinstance(mod, LinearBase) or w is None or w.dim() != 2 or w.shape[1] % _GROUP:
            continue
        if type(mod.quant_method).__name__ != "UnquantizedLinearMethod":
            continue
        qweight, scales, qzeros, _ = quantize_to_int4(w.data, _GROUP)
        scales = scales.to(w.dtype)
        mod.quant_method = _Int4LinearMethod(qweight, scales, qzeros)
        before += w.numel() * w.element_size()
        after += qweight.numel() * qweight.element_size() + scales.numel() * scales.element_size()
        w.data = torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device)
        count += 1
    torch.xpu.empty_cache()
    print(f"[B70] DFlash drafter int4: {count} linears, {before / 1e9:.2f} -> {after / 1e9:.2f} GB per step", flush=True)
