"""Shared-expert MFMA for qualified 24/48-token TP1 BF16 prefill.

Keep BF16 materialization after each linear, SiLU, product, sigmoid and add.
Routing remains in SharedRegionMoe and uses the original prefill router.
"""

import triton as tr
import triton.language as tl


@tr.jit
def _prepare_mfma(X, W, SG, A, G, M: tl.constexpr):
    m = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.program_id(0) * 16 + tl.arange(0, 16)
    k = tl.arange(0, 128)
    g = tl.full((16, 16), 0, tl.float32)
    u = tl.full((16, 16), 0, tl.float32)
    sg = tl.full((16,), 0, tl.float32)
    for off in range(8):
        kk = off * 128 + k
        x = tl.load(X + m[:, None] * 1024 + kk[None, :], m[:, None] < M, other=0)
        wg = tl.load(W + n[None, :] * 1024 + kk[:, None])
        wu = tl.load(W + (n[None, :] + 768) * 1024 + kk[:, None])
        g += tl.dot(x, wg)
        u += tl.dot(x, wu)
        if tl.program_id(0) == 0:
            sw = tl.load(SG + kk).to(tl.float32)
            sg += tl.sum(x.to(tl.float32) * sw[None, :], 1)
    gb = g.to(tl.bfloat16).to(tl.float32)
    ub = u.to(tl.bfloat16).to(tl.float32)
    a = ((gb / (1 + tl.exp(-gb))).to(tl.bfloat16).to(tl.float32) * ub).to(tl.bfloat16)
    tl.store(A + m[:, None] * 768 + n[None, :], a, m[:, None] < M)
    if tl.program_id(0) == 0:
        tl.store(
            G + m, tl.sigmoid(sg.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), m < M
        )


@tr.jit
def _finish_mfma(A, W, G, ROUTED, O, M: tl.constexpr):
    m = tl.program_id(1) * 16 + tl.arange(0, 16)
    n = tl.program_id(0) * 16 + tl.arange(0, 16)
    k = tl.arange(0, 128)
    acc = tl.full((16, 16), 0, tl.float32)
    for off in range(6):
        kk = off * 128 + k
        a = tl.load(A + m[:, None] * 768 + kk[None, :], m[:, None] < M, other=0)
        w = tl.load(W + n[None, :] * 768 + kk[:, None])
        acc += tl.dot(a, w)
    down = acc.to(tl.bfloat16).to(tl.float32)
    gate = tl.load(G + m, m < M, other=0).to(tl.float32)
    shared = (down * gate[:, None]).to(tl.bfloat16).to(tl.float32)
    routed = tl.load(
        ROUTED + m[:, None] * 1024 + n[None, :], m[:, None] < M, other=0
    ).to(tl.float32)
    tl.store(
        O + m[:, None] * 1024 + n[None, :],
        (routed + shared).to(tl.bfloat16),
        m[:, None] < M,
    )
