"""Fused Thinker Q/K RMSNorm and prepared MRoPE with BF16 boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qknorm_rope(
    Q,
    K,
    QW,
    KW,
    C,
    S,
    OQ,
    OK,
    QH: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    EPS: tl.constexpr,
    ROUND_PRODUCTS: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // (QH + KH)
    head = row % (QH + KH)
    d = tl.arange(0, D)
    if head < QH:
        x = tl.load(Q + token * QS0 + head * QS1 + d).to(tl.float32)
        w = tl.load(QW + d).to(tl.float32)
    else:
        x = tl.load(K + token * KS0 + (head - QH) * KS1 + d).to(tl.float32)
        w = tl.load(KW + d).to(tl.float32)
    norm = (x * tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * w).to(tl.bfloat16).to(tl.float32)
    rot = tl.gather(norm, (d + D // 2) % D, 0) * tl.where(d < D // 2, -1.0, 1.0)
    c = tl.load(C + token * CS + d).to(tl.float32)
    s = tl.load(S + token * SS + d).to(tl.float32)
    a = norm * c
    b = rot * s
    if ROUND_PRODUCTS:
        a = a.to(tl.bfloat16).to(tl.float32)
        b = b.to(tl.bfloat16).to(tl.float32)
    y = (a + b).to(tl.bfloat16)
    if head < QH:
        tl.store(OQ + token * QH * D + head * D + d, y)
    else:
        tl.store(OK + token * KH * D + (head - QH) * D + d, y)


def fused_qknorm_rope(q, k, qw, kw, cos, sin, eps=1e-6, round_products=True):
    m, qh, d = q.shape
    kh = k.shape[1]
    oq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ok = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    _qknorm_rope[(m * (qh + kh),)](
        q,
        k,
        qw,
        kw,
        cos,
        sin,
        oq,
        ok,
        qh,
        kh,
        d,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        cos.stride(0),
        sin.stride(0),
        eps,
        round_products,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return oq, ok


def supported(q, k, qw, kw, cos, sin):
    tensors = (q, k, qw, kw, cos, sin)
    return (
        q.is_cuda
        and all(t.dtype == torch.bfloat16 and t.device == q.device for t in tensors)
        and q.ndim == 3
        and k.ndim == 3
        and q.shape[0] in (1, 2, 4, 8, 128, 256, 512, 1024, 2048)
        and q.shape[1:] == (32, 128)
        and k.shape == (q.shape[0], 4, 128)
        and qw.shape == kw.shape == (128,)
        and cos.shape == sin.shape == (q.shape[0], 128)
        and q.stride(-1) == k.stride(-1) == 1
        and all(t.is_contiguous() for t in (qw, kw, cos, sin))
    )
