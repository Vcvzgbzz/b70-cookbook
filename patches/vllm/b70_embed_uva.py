"""B70: keep the input embedding table in pinned host RAM, read by the GPU in place (gate B70_EMBED_UVA=1).

Each step looks up a handful of 10 KB rows, so a UVA view costs ~nothing and frees ~2.4 GiB of VRAM before the
KV cache is sized. The MTP draft shares this module, so one offload covers both. Pair it with an offloaded
tower (`--cpu-offload-gb N --cpu-offload-params visual`), which also turns off Inductor's static launcher that
rejects host pointers.
"""
from __future__ import annotations

import os

import torch


@torch.no_grad()
def offload_embed_tokens(model) -> None:
    """Move model.embed_tokens.weight to pinned host memory behind a device view; no-op without the gate."""
    if os.environ.get("B70_EMBED_UVA") != "1":
        return
    emb = getattr(model, "embed_tokens", None)
    w = getattr(emb, "weight", None)
    if w is None or getattr(w, "_vllm_is_uva_offloaded", False) or w.device.type == "cpu":
        return
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    host = w.data.to("cpu").pin_memory()
    emb._b70_host_weight = host  # the device view does not own the host buffer
    w.data = get_accelerator_view_from_cpu_tensor(host)
    w._vllm_is_uva_offloaded = True
    torch.xpu.empty_cache()
    print(f"[B70] embed_tokens -> pinned host RAM via UVA ({host.numel() * host.element_size() / 2**30:.2f} GiB)",
          flush=True)
