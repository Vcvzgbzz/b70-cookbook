

# B70 packed MTP verify with an in-kernel per-row causal limit (gate B70_PACKED_ROWMASK=1; needs a
# B70_SPEC_ROW_MASK kernel build): all draft positions x GQA heads of a sequence ride one decode query per KV head;
# the kernel reads B70_SPEC_QLEN at launch (baked in at XPU-graph capture) and masks each row's newest keys.
if os.environ.get("B70_PACKED_ROWMASK") == "1":
    _b70_rowmask_stock_spec = _spec_decode_varlen_fwd

    def _spec_decode_varlen_fwd(q, k, v, out, cu_seqlens_q, seqused_k, block_table, max_seqlen_k,
                                k_descale, v_descale, softmax_scale, s_aux, window_size, softcap):
        B = cu_seqlens_q.numel() - 1
        ql = q.shape[0] // B
        hkv, D = k.shape[2], q.shape[2]
        g = q.shape[1] // hkv
        if s_aux is not None or softcap != 0.0 or tuple(window_size) != (-1, -1) or ql < 2:
            return _b70_rowmask_stock_spec(q, k, v, out, cu_seqlens_q, seqused_k, block_table, max_seqlen_k,
                                           k_descale, v_descale, softmax_scale, s_aux, window_size, softcap)
        qp = q.view(B, ql, hkv, g, D).permute(0, 2, 1, 3, 4).reshape(B, hkv * ql * g, D)
        cu = torch.arange(B + 1, dtype=torch.int32, device=q.device)
        os.environ["B70_SPEC_QLEN"] = str(ql)
        try:
            op = flash_attn_varlen_func(qp, k, v, 1, cu, max_seqlen_k, seqused_k=seqused_k, causal=False,
                                        block_table=block_table, softmax_scale=softmax_scale, fa_version=2,
                                        k_descale=k_descale, v_descale=v_descale)
        finally:
            os.environ["B70_SPEC_QLEN"] = "1"
        res = op.view(B, hkv, ql, g, D).permute(0, 2, 1, 3, 4).reshape(B * ql, hkv * g, D)
        if out is None:
            return res
        out.copy_(res)
        return out
