"""Small-token BF16 routed experts for the MI300X Talker geometry.

Routing decisions stay outside this backend. The kernels preserve BF16
projection, SiLU, product and weighted-output rounding before the FP32
route sum. Other shapes use the inherited grouped-GEMM backend.
"""

import torch
import triton
import triton.language as tl

from mstar.utils.fused_moe.backend import TritonFusedMoE
from mstar.utils.fused_moe.kernels import moe_sum_reduce_triton


@triton.jit
def _gate_vector(X, W, IDS, Y, H: tl.constexpr, I: tl.constexpr, TOP: tl.constexpr, BN: tl.constexpr):
    s = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    e = tl.load(IDS + s).to(tl.int64)
    k = tl.arange(0, triton.next_power_of_2(H))
    x = tl.load(X + (s // TOP) * H + k, mask=k < H, other=0).to(tl.float32)
    g = tl.load(W + e * 2 * I * H + n[:, None] * H + k[None, :], mask=(n[:, None] < I) & (k[None, :] < H), other=0).to(
        tl.float32
    )
    u = tl.load(
        W + e * 2 * I * H + (n[:, None] + I) * H + k[None, :], mask=(n[:, None] < I) & (k[None, :] < H), other=0
    ).to(tl.float32)
    g = tl.sum(g * x[None, :], 1).to(tl.bfloat16).to(tl.float32)
    u = tl.sum(u * x[None, :], 1).to(tl.bfloat16).to(tl.float32)
    act = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + s * I + n, (act * u).to(tl.bfloat16), mask=n < I)


@triton.jit
def _down_vector(X, W, IDS, P, Y, H: tl.constexpr, I: tl.constexpr, TOP: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, triton.next_power_of_2(I))
    out = tl.zeros((BN,), tl.float32)
    for r in range(TOP):
        e = tl.load(IDS + t * TOP + r).to(tl.int64)
        x = tl.load(X + (t * TOP + r) * I + k, mask=k < I, other=0).to(tl.float32)
        w = tl.load(W + e * H * I + n[:, None] * I + k[None, :], mask=(n[:, None] < H) & (k[None, :] < I), other=0).to(
            tl.float32
        )
        y = tl.sum(w * x[None, :], 1) * tl.load(P + t * TOP + r)
        out += y.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + t * H + n, out.to(tl.bfloat16), mask=n < H)


def vector_experts(x, w1, w2, p, ids, config=(4, 4, 4)):
    M, H = x.shape
    I = w2.shape[2]
    TOP = ids.shape[1]
    bn, bnd, nw = config
    y = torch.empty((M * TOP, I), device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    _gate_vector[(M * TOP, triton.cdiv(I, bn))](x, w1, ids, y, H, I, TOP, bn, num_warps=nw, enable_fp_fusion=False)
    _down_vector[(M, triton.cdiv(H, bnd))](y, w2, ids, p, out, H, I, TOP, bnd, num_warps=nw, enable_fp_fusion=False)
    return out


@triton.jit
def _grouped_gate(
    X,
    W,
    IDS,
    Y,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOP: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    s = tl.program_id(0)
    e = tl.load(IDS + s).to(tl.int64)
    all_s = tl.arange(0, triton.next_power_of_2(M * TOP))
    all_ids = tl.load(IDS + all_s, mask=all_s < M * TOP, other=-1)
    if tl.sum(((all_ids == e) & (all_s < s)).to(tl.int32), 0) > 0:
        return
    m = tl.arange(0, 16)
    r = tl.arange(0, triton.next_power_of_2(TOP))
    row_ids = tl.load(IDS + m[:, None] * TOP + r[None, :], mask=(m[:, None] < M) & (r[None, :] < TOP), other=-1)
    hit = row_ids == e
    valid = tl.sum(hit.to(tl.int32), 1) > 0
    route = tl.sum(tl.where(hit, r[None, :], 0), 1)
    slot = m * TOP + route
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    ag = tl.zeros((16, BN), tl.float32)
    au = tl.zeros((16, BN), tl.float32)
    for start in range(tl.cdiv(H, BK)):
        kk = start * BK + k
        a = tl.load(X + m[:, None] * H + kk[None, :], mask=valid[:, None] & (kk[None, :] < H), other=0)
        bg = tl.load(
            W + e * 2 * I * H + n[None, :] * H + kk[:, None], mask=(n[None, :] < I) & (kk[:, None] < H), other=0
        )
        bu = tl.load(
            W + e * 2 * I * H + (n[None, :] + I) * H + kk[:, None], mask=(n[None, :] < I) & (kk[:, None] < H), other=0
        )
        ag += tl.dot(a, bg)
        au += tl.dot(a, bu)
    g = ag.to(tl.bfloat16).to(tl.float32)
    u = au.to(tl.bfloat16).to(tl.float32)
    act = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + slot[:, None] * I + n[None, :], (act * u).to(tl.bfloat16), mask=valid[:, None] & (n[None, :] < I))


@triton.jit
def _grouped_down(
    X,
    W,
    IDS,
    P,
    Y,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOP: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    s = tl.program_id(0)
    e = tl.load(IDS + s).to(tl.int64)
    all_s = tl.arange(0, triton.next_power_of_2(M * TOP))
    all_ids = tl.load(IDS + all_s, mask=all_s < M * TOP, other=-1)
    if tl.sum(((all_ids == e) & (all_s < s)).to(tl.int32), 0) > 0:
        return
    m = tl.arange(0, 16)
    r = tl.arange(0, triton.next_power_of_2(TOP))
    row_ids = tl.load(IDS + m[:, None] * TOP + r[None, :], mask=(m[:, None] < M) & (r[None, :] < TOP), other=-1)
    hit = row_ids == e
    valid = tl.sum(hit.to(tl.int32), 1) > 0
    route = tl.sum(tl.where(hit, r[None, :], 0), 1)
    slot = m * TOP + route
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(tl.cdiv(I, BK)):
        kk = start * BK + k
        a = tl.load(X + slot[:, None] * I + kk[None, :], mask=valid[:, None] & (kk[None, :] < I), other=0)
        b = tl.load(W + e * H * I + n[None, :] * I + kk[:, None], mask=(n[None, :] < H) & (kk[:, None] < I), other=0)
        acc += tl.dot(a, b)
    prob = tl.load(P + slot, mask=valid, other=0)
    tl.store(
        Y + slot[:, None] * H + n[None, :],
        (acc * prob[:, None]).to(tl.bfloat16),
        mask=valid[:, None] & (n[None, :] < H),
    )


def grouped_experts(x, w1, w2, p, ids, config=(32, 128, 2, 1)):
    M, H = x.shape
    I = w2.shape[2]
    TOP = ids.shape[1]
    bn, bk, nw, kp = config
    y = torch.empty((M * TOP, I), device=x.device, dtype=x.dtype)
    z = torch.empty((M, TOP, H), device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    _grouped_gate[(M * TOP, triton.cdiv(I, bn))](
        x,
        w1,
        ids,
        y,
        M,
        H,
        I,
        TOP,
        bn,
        bk,
        num_warps=nw,
        num_stages=2,
        waves_per_eu=2,
        matrix_instr_nonkdim=16,
        kpack=kp,
    )
    _grouped_down[(M * TOP, triton.cdiv(H, bn))](
        y,
        w2,
        ids,
        p,
        z,
        M,
        H,
        I,
        TOP,
        bn,
        bk,
        num_warps=nw,
        num_stages=2,
        waves_per_eu=2,
        matrix_instr_nonkdim=16,
        kpack=kp,
    )
    moe_sum_reduce_triton(z, out)
    return out


class SmallTokenTalkerMoE(TritonFusedMoE):
    """Compute router-selected unique top-k experts on dense BF16 operands.

    The inherited runner requires contiguous hidden states and expert weights.
    Unqualified operands retain its validation/behavior through the fallback.
    """

    def __call__(self, x, w1, w2, weights, ids):
        if (
            x.dtype == torch.bfloat16
            and w1.dtype == torch.bfloat16
            and w2.dtype == torch.bfloat16
            and x.is_contiguous()
            and w1.is_contiguous()
            and w2.is_contiguous()
            and tuple(w1.shape) == (128, 768, 1024)
            and tuple(w2.shape) == (128, 1024, 384)
            and ids.shape[1] == 6
            and weights.is_contiguous()
            and ids.is_contiguous()
        ):
            if x.shape[0] in (1, 2, 4):
                return vector_experts(x, w1, w2, weights, ids, (4, 4, 4))
            if x.shape[0] == 8:
                return grouped_experts(x, w1, w2, weights, ids, (64, 64, 4, 2))
        return super().__call__(x, w1, w2, weights, ids)
