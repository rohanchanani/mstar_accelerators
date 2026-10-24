"""Qwen3-Omni's fused kernels against the op sequences they replace (GPU only).

* ``cp_qk_norm_rope_kv_write``: the code predictor's q/k norm + RoPE and its
  dense-KV write, bit-for-bit what two ``fused_qk_norm_rope`` calls and two
  strided copies produce.
* ``softmax_topk``: ``TopKRouter``'s softmax -> topk -> renormalize.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.parametrize("bs,seq_len,cache_pos", [(1, 1, 0), (5, 1, 7), (32, 1, 15), (3, 2, 4)])
def test_cp_prologue_matches_the_unfused_ops(bs, seq_len, cache_pos):
    from mstar_accelerators.implementations.qwen3_omni.amd_b.cp_kernels import cp_qk_norm_rope_kv_write
    from mstar.utils.attention import fused_qk_norm_rope

    torch.manual_seed(0)
    hq, hkv, d, layers, max_bs, max_s = 16, 8, 128, 5, 32, 16
    dev, dt = "cuda", torch.bfloat16
    qkv = torch.randn(bs, seq_len, (hq + 2 * hkv) * d, device=dev, dtype=dt)
    wq = torch.randn(d, device=dev, dtype=dt)
    wk = torch.randn(d, device=dev, dtype=dt)
    pos = torch.randint(0, 16, (bs, seq_len), device=dev)
    cache = torch.randn(layers, max_bs, 2, max_s, hkv, d, device=dev, dtype=dt)
    ref_cache, ref_qkv = cache.clone(), qkv.clone()
    flat_pos = pos.reshape(-1)

    q, k, v = ref_qkv.split((hq * d, hkv * d, hkv * d), dim=-1)
    q = fused_qk_norm_rope(q.view(bs * seq_len, hq, d), wq, pos=flat_pos, eps=1e-6, rope_theta=1e6)
    k = fused_qk_norm_rope(k.view(bs * seq_len, hkv, d), wk, pos=flat_pos, eps=1e-5, rope_theta=1e6)
    view = ref_cache[:, :bs]
    view[2, :, 0, cache_pos:cache_pos + seq_len] = k.view(bs, seq_len, hkv, d)
    view[2, :, 1, cache_pos:cache_pos + seq_len] = v.view(bs, seq_len, hkv, d)

    cp_qk_norm_rope_kv_write(
        qkv.view(bs * seq_len, -1), wq, wk, flat_pos, cache[:, :bs][2], cache_pos, seq_len,
        1e-6, 1e-5, 1e6, hq, hkv, d,
    )
    torch.testing.assert_close(qkv[..., :hq * d], ref_qkv[..., :hq * d], rtol=0, atol=0)
    torch.testing.assert_close(cache, ref_cache, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 7, 32, 300, 4096])
@pytest.mark.parametrize("experts,top_k,norm", [(128, 8, True), (128, 6, True), (60, 4, False)])
def test_softmax_topk_matches_the_torch_router(tokens, experts, top_k, norm):
    from mstar.utils.fused_moe.router import softmax_topk

    torch.manual_seed(1)
    logits = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16) * 3
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    ref_w, ref_ids = torch.topk(probs, top_k, dim=-1)
    if norm:
        ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)

    w, ids = softmax_topk(logits, top_k, norm)
    assert ids.dtype == torch.int64 and w.dtype == torch.float32
    # bf16 logits tie often and torch.topk orders ties arbitrarily: compare the
    # selected probabilities, and that every pick is a distinct expert holding them
    torch.testing.assert_close(w, ref_w, rtol=1e-6, atol=1e-7)
    assert (ids.sort(dim=-1).values.diff(dim=-1) > 0).all()
    picked = probs.gather(1, ids)
    if norm:
        picked = picked / picked.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(picked, w, rtol=1e-6, atol=1e-7)
    # with no ties the choice and its order are torch's
    distinct = (ref_w[:, :-1] != ref_w[:, 1:]).all(dim=-1)
    assert torch.equal(ids[distinct], ref_ids[distinct])


def test_router_compiles_with_the_fused_topk():
    from mstar.model.components.moe import TopKRouter

    router = TopKRouter(64, 128, 8).cuda().to(torch.bfloat16).requires_grad_(False)
    torch.nn.init.normal_(router.weight)
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        eager = router(x)
        compiled = torch.compile(router, fullgraph=True)(x)
    assert torch.equal(eager[1], compiled[1])
    torch.testing.assert_close(eager[0], compiled[0], rtol=0, atol=0)


@pytest.mark.parametrize("bs", [1, 8, 32])
def test_cp_embed_accumulate_matches_the_unfused_ops(bs):
    from mstar_accelerators.implementations.qwen3_omni.amd_b.cp_kernels import cp_embed_accumulate

    torch.manual_seed(2)
    weight = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    tokens = torch.randint(0, 2048, (bs,), device="cuda")
    emb_sum = torch.randn(bs, 1024, device="cuda")
    codes = torch.randint(0, 2048, (bs, 16), device="cuda")
    ref_sum, ref_codes = emb_sum.clone(), codes.clone()

    ref_codes[:, 5] = tokens
    ref_emb = torch.nn.functional.embedding(tokens, weight)
    ref_sum.add_(ref_emb)

    emb = cp_embed_accumulate(tokens, weight, emb_sum, codes, 5)
    assert torch.equal(emb, ref_emb)
    assert torch.equal(emb_sum, ref_sum)
    assert torch.equal(codes, ref_codes)
