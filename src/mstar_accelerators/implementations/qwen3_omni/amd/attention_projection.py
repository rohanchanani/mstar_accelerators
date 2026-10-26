import torch
import triton
import triton.language as tl


@triton.jit
def _attention_projection(
    Q,
    K,
    V,
    W,
    P,
    SQ: tl.constexpr,
    SKB: tl.constexpr,
    SKN: tl.constexpr,
    SVB: tl.constexpr,
    SVN: tl.constexpr,
    L: tl.constexpr,
    BN: tl.constexpr,
):
    b = tl.program_id(0)
    head = tl.program_id(1)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    d = tl.arange(0, 128)
    t = tl.arange(0, 16)
    q = tl.load(Q + b * SQ + head * 128 + d).to(tl.float32) * (128.0**-0.5)
    k = tl.load(
        K + b * SKB + t[:, None] * SKN + (head // 2) * 128 + d[None, :],
        t[:, None] < L,
        other=0,
    ).to(tl.float32)
    logits = tl.sum(k * q[None, :], 1)
    logits = tl.where(t < L, logits, -float("inf"))
    prob = tl.exp(logits - tl.max(logits, 0))
    v = tl.load(
        V + b * SVB + t[:, None] * SVN + (head // 2) * 128 + d[None, :],
        t[:, None] < L,
        other=0,
    ).to(tl.float32)
    a = (tl.sum(prob[:, None] * v, 0) / tl.sum(prob, 0)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(
        W + n[:, None] * 2048 + head * 128 + d[None, :], n[:, None] < 1024, other=0
    ).to(tl.float32)
    partial = tl.sum(w * a[None, :], 1)
    tl.store(P + (b * 16 + head) * 1024 + n, partial, n < 1024)


@triton.jit
def _finish(P, X, N, Y, R, EPS: tl.constexpr):
    b = tl.program_id(0)
    n = tl.arange(0, 1024)
    heads = tl.arange(0, 16)
    partial = tl.load(P + (b * 16 + heads[:, None]) * 1024 + n[None, :])
    proj = tl.sum(partial, 0).to(tl.bfloat16).to(tl.float32)
    x = tl.load(X + b * 1024 + n).to(tl.float32)
    r = (x + proj).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(r * r, 0) / 1024 + EPS)
    norm = tl.load(N + n).to(tl.float32)
    tl.store(R + b * 1024 + n, r.to(tl.bfloat16))
    tl.store(Y + b * 1024 + n, ((r * inv) * norm).to(tl.bfloat16))


@torch.compiler.disable
def fused(q, k, v, w, x, n, L, bn=128, eps=1e-6):
    b = q.shape[0]
    p = torch.empty((b, 16, 1024), device=q.device, dtype=torch.float32)
    y = torch.empty_like(x)
    r = torch.empty_like(x)
    _attention_projection[(b, 16, triton.cdiv(1024, bn))](
        q,
        k,
        v,
        w,
        p,
        q.stride(0),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        L,
        bn,
        num_warps=4,
    )
    _finish[(b,)](p, x, n, y, r, eps, num_warps=8, enable_fp_fusion=False)
    return y, r


@torch.compiler.disable
def attention_projection_norm(q, k, v, weight, residual, norm_weight, cache_len, eps):
    """Preserve BF16 attention/projection/sum boundaries and the compiler break.

    Only short single-token gfx942 predictor shapes are specialized. The caller
    owns Q/K normalization, RoPE and KV writes; this function reads those states.
    """
    from .short_attention import _gfx942

    eligible = (
        q.is_cuda
        and torch.version.hip
        and _gfx942(q.device.index)
        and q.ndim == 4
        and q.shape[1:] == (1, 16, 128)
        and q.shape[0] in (1, 2)
        and q.dtype == torch.bfloat16
        and q.stride(-1) == 1
        and q.stride(-2) == 128
        and k.ndim == v.ndim == 4
        and k.shape == v.shape
        and k.shape[0] == q.shape[0]
        and k.shape[2:] == (8, 128)
        and k.shape[1] >= cache_len
        and 1 <= cache_len <= 16
        and k.stride(-1) == v.stride(-1) == 1
        and k.stride(-2) == v.stride(-2) == 128
        and all(
            t.dtype == torch.bfloat16 for t in (k, v, weight, residual, norm_weight)
        )
        and tuple(weight.shape) == (1024, 2048)
        and tuple(residual.shape) == (q.shape[0], 1, 1024)
        and tuple(norm_weight.shape) == (1024,)
        and all(t.is_contiguous() for t in (weight, residual, norm_weight))
        and all(t.device == q.device for t in (k, v, weight, residual, norm_weight))
    )
    if eligible:
        return fused(
            q,
            k,
            v,
            weight,
            residual,
            norm_weight,
            cache_len,
            64 if q.shape[0] == 1 else 128,
            eps,
        )
    from .rounded_norm import rounded_add_norm
    from .short_attention import short_decode_attn_nhd

    attention = short_decode_attn_nhd(q, k, v, cache_len)
    projected = torch.nn.functional.linear(
        attention.reshape(*residual.shape[:-1], -1), weight
    )
    return rounded_add_norm(residual, projected, norm_weight, eps)
