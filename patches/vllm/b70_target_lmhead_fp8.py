"""B70: target lm_head as per-row fp8 e4m3 through fp8_gemm_w8a16 (gate B70_TARGET_LMHEAD_FP8=1).

Halves the per-step head read (fp16 2.5 GB -> fp8 1.27 GB). The MTP draft shares this lm_head module and
builds its int4 head from the same fp16 weight, so the fp16 copy is released only once both copies exist.
"""
from __future__ import annotations

import os

import torch

_FP8_MAX = 448.0


class _Fp8HeadMethod:
    """Stands in for the lm_head's quant method; only apply() changes."""

    def __init__(self, base, qweight_t: torch.Tensor, scale_t: torch.Tensor):
        self.base = base
        self.qweight_t = qweight_t  # [hidden, vocab] view of a contiguous [vocab, hidden] fp8 tensor
        self.scale_t = scale_t  # [1, vocab] fp32

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        out = torch.ops._xpu_C.fp8_gemm_w8a16(flat, self.qweight_t, self.scale_t, bias)
        return out.reshape(*x.shape[:-1], out.shape[-1])

    def __getattr__(self, name):
        return getattr(self.base, name)


@torch.no_grad()
def use_fp8_target_lmhead(model) -> None:
    """One-time swap of model.lm_head's matmul to fp8; no-op without the env gate or once done."""
    if os.environ.get("B70_TARGET_LMHEAD_FP8") != "1" or getattr(model, "_b70_lmhead_fp8_done", False):
        return
    model._b70_lmhead_fp8_done = True
    head = getattr(model, "lm_head", None)
    w = getattr(head, "weight", None)
    if w is None or getattr(head, "tp_size", 1) != 1 or not hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):
        print("[B70] target lm_head fp8: unavailable here; staying fp16", flush=True)
        return
    vocab = w.shape[0]
    q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty((vocab, 1), dtype=torch.float32, device=w.device)
    for i in range(0, vocab, 8192):
        wc = w[i : i + 8192].float()
        s = wc.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / _FP8_MAX
        q[i : i + 8192] = (wc / s).to(torch.float8_e4m3fn)
        scale[i : i + 8192] = s
    head.quant_method = _Fp8HeadMethod(head.quant_method, q.t(), scale.t().contiguous())
    print(f"[B70] target lm_head fp8: {tuple(w.shape)} per-row e4m3, "
          f"{w.numel() * w.element_size() / 1e9:.2f} -> {q.numel() / 1e9:.2f} GB per step", flush=True)
    head._b70_fp8_built = True
    release_fp16_head_if_unused(head)


@torch.no_grad()
def release_fp16_head_if_unused(head) -> None:
    """Drop the fp16 lm_head weight once the fp8 target head and (if enabled) the int4 draft head are built."""
    if not getattr(head, "_b70_fp8_built", False) or getattr(head, "_b70_fp16_released", False):
        return
    if os.environ.get("B70_DRAFT_LMHEAD_INT4") == "1" and not getattr(head, "_b70_int4_built", False):
        return
    w = head.weight
    freed = w.numel() * w.element_size()
    w.data = torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device)
    head._b70_fp16_released = True
    torch.xpu.empty_cache()
    print(f"[B70] target lm_head fp8: released the fp16 weight ({freed / 1e9:.2f} GB)", flush=True)
