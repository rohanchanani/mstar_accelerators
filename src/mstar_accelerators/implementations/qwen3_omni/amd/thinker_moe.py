"""Small Thinker decode experts, preserving materialized BF16 boundaries.

The first kernel fuses gate/up projection and rounded SwiGLU. At one, two,
and four rows, the second fuses down projection and routing reduction. At
eight rows, a wider down tile and the original reduction perform better.
Both preserve the weighted-expert BF16 boundary and serial FP32 route sum.
"""

import torch
import triton
import triton.language as tl

from mstar.utils.fused_moe.backend import TritonFusedMoE
from mstar.utils.fused_moe.kernels import moe_sum_reduce_triton


@triton.jit
def gate_up_vector(
    X,
    W,
    IDS,
    Y,
    H: tl.constexpr,
    I: tl.constexpr,
    TOP: tl.constexpr,
    E: tl.constexpr,
    BN: tl.constexpr,
    KH: tl.constexpr,
):
    slot = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, KH)
    expert = tl.load(IDS + slot).to(tl.int64)
    x = tl.load(X + (slot // TOP) * H + k, mask=k < H, other=0).to(tl.float32)
    g = tl.load(
        W + expert * 2 * I * H + n[:, None] * H + k[None, :],
        mask=(n[:, None] < I) & (k[None, :] < H) & (expert < E),
        other=0,
    ).to(tl.float32)
    u = tl.load(
        W + expert * 2 * I * H + (n[:, None] + I) * H + k[None, :],
        mask=(n[:, None] < I) & (k[None, :] < H) & (expert < E),
        other=0,
    ).to(tl.float32)
    gv = tl.sum(g * x[None, :], 1).to(tl.bfloat16).to(tl.float32)
    uv = tl.sum(u * x[None, :], 1).to(tl.bfloat16)
    act = (gv * tl.sigmoid(gv)).to(tl.bfloat16)
    tl.store(Y + slot * I + n, (act * uv).to(tl.bfloat16), mask=n < I)


@triton.jit
def down_vector_reduce(
    X,
    W,
    IDS,
    RW,
    Y,
    H: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    TOP: tl.constexpr,
    BN: tl.constexpr,
    KI: tl.constexpr,
):
    token = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, KI)
    s = tl.arange(0, TOP)
    slot = token * TOP + s
    expert = tl.load(IDS + slot).to(tl.int64)
    x = tl.load(X + slot[:, None] * I + k[None, :], mask=k[None, :] < I, other=0).to(tl.float32)
    w = tl.load(
        W + expert[:, None, None] * H * I + n[None, :, None] * I + k[None, None, :],
        mask=(n[None, :, None] < H) & (k[None, None, :] < I) & (expert[:, None, None] < E),
        other=0,
    ).to(tl.float32)
    y = tl.sum(w * x[:, None, :], 2) * tl.load(RW + slot)[:, None]
    y = y.to(tl.bfloat16).to(tl.float32)
    # Preserve the upstream sequential expert-slot accumulation order.
    result = tl.full((BN,), 0, tl.float32)
    for t in tl.static_range(TOP):
        result += tl.sum(tl.where(s[:, None] == t, y, 0.0), 0)
    tl.store(Y + token * H + n, result.to(tl.bfloat16), mask=n < H)


@triton.jit
def down_vector(
    X, W, IDS, RW, Y, H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, BN: tl.constexpr, KI: tl.constexpr
):
    slot = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, KI)
    expert = tl.load(IDS + slot).to(tl.int64)
    x = tl.load(X + slot * I + k, mask=k < I, other=0).to(tl.float32)
    w = tl.load(
        W + expert * H * I + n[:, None] * I + k[None, :],
        mask=(n[:, None] < H) & (k[None, :] < I) & (expert < E),
        other=0,
    ).to(tl.float32)
    y = tl.sum(w * x[None, :], 1) * tl.load(RW + slot)
    tl.store(Y + slot * H + n, y.to(tl.bfloat16), mask=n < H)


class SmallTokenThinkerMoE(TritonFusedMoE):
    """gfx942 Thinker specialization; all other invocations retain the backend."""

    def __call__(self, hidden_states, w1, w2, topk_weights, topk_ids, *, skip_invalid=False):
        if skip_invalid:
            return super().__call__(hidden_states, w1, w2, topk_weights, topk_ids, skip_invalid=True)
        if (
            hidden_states.dtype != torch.bfloat16
            or w1.dtype != torch.bfloat16
            or w2.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
            or not w1.is_contiguous()
            or not w2.is_contiguous()
            or hidden_states.shape[0] not in (1, 2, 4, 8)
            or not topk_weights.is_contiguous()
            or not topk_ids.is_contiguous()
        ):
            return super().__call__(hidden_states, w1, w2, topk_weights, topk_ids)
        m, h = hidden_states.shape
        e, _, i = w2.shape
        top = topk_ids.shape[1]
        mid = torch.empty((m * top, i), dtype=hidden_states.dtype, device=hidden_states.device)
        output = torch.empty_like(hidden_states)
        if m == 8:
            expert_outputs = torch.empty((m, top, h), dtype=hidden_states.dtype, device=hidden_states.device)
            gate_up_vector[(m * top, triton.cdiv(i, 8))](
                hidden_states,
                w1,
                topk_ids,
                mid,
                h,
                i,
                top,
                e,
                BN=8,
                KH=triton.next_power_of_2(h),
                num_warps=4,
                enable_fp_fusion=False,
            )
            down_vector[(m * top, triton.cdiv(h, 8))](
                mid,
                w2,
                topk_ids,
                topk_weights,
                expert_outputs,
                h,
                i,
                e,
                BN=8,
                KI=triton.next_power_of_2(i),
                num_warps=4,
                enable_fp_fusion=False,
            )
            moe_sum_reduce_triton(expert_outputs, output)
            return output
        gate_up_vector[(m * top, i)](
            hidden_states,
            w1,
            topk_ids,
            mid,
            h,
            i,
            top,
            e,
            BN=1,
            KH=triton.next_power_of_2(h),
            num_warps=4,
            enable_fp_fusion=False,
        )
        down_vector_reduce[(m, h)](
            mid,
            w2,
            topk_ids,
            topk_weights,
            output,
            h,
            i,
            e,
            top,
            BN=1,
            KI=triton.next_power_of_2(i),
            num_warps=4,
            enable_fp_fusion=False,
        )
        return output
