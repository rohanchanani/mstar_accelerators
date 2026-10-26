"""TP1 Thinker output projection, rounded residual and post-attention norm."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def project(X, W, P, M: tl.constexpr):
    m = tl.arange(0, 16)
    n = tl.program_id(0) * 32 + tl.arange(0, 32)
    k = tl.arange(0, 128)
    acc = tl.zeros((16, 32), tl.float32)
    for off in range(4):
        kk = (off * 8 + tl.program_id(1)) * 128 + k
        x = tl.load(X + m[:, None] * 4096 + kk[None, :], m[:, None] < M, other=0)
        w = tl.load(W + n[None, :] * 4096 + kk[:, None])
        acc += tl.dot(x, w)
    tl.store(P + tl.program_id(1) * M * 2048 + m[:, None] * 2048 + n[None, :], acc, m[:, None] < M)


@tr.jit
def finish(P, R, W, Y, S, M: tl.constexpr):
    row = tl.program_id(0)
    n = tl.arange(0, 2048)
    s = tl.arange(0, 8)
    p = tl.load(P + s[:, None] * M * 2048 + row * 2048 + n[None, :])
    delta = tl.sum(p, 0).to(tl.bfloat16).to(tl.float32)
    res = tl.load(R + row * 2048 + n).to(tl.float32)
    rounded = (res + delta).to(tl.bfloat16)
    z = rounded.to(tl.float32)
    norm = z * tl.rsqrt(tl.sum(z * z, 0) / 2048 + 1e-6) * tl.load(W + n).to(tl.float32)
    tl.store(Y + row * 2048 + n, norm)
    tl.store(S + row * 2048 + n, rounded)


def run(x, w, r, nw):
    m = x.shape[0]
    p = torch.empty((8, m, 2048), device=x.device, dtype=torch.float32)
    y = torch.empty_like(r)
    s = torch.empty_like(r)
    project[(64, 8)](x, w, p, m, num_warps=2, num_stages=2, matrix_instr_nonkdim=16, enable_fp_fusion=False)
    finish[(m,)](p, r, nw, y, s, m, num_warps=4, enable_fp_fusion=False)
    return y, s


def supported(attn, norm, residual):
    from mstar.model.components.distributed.linear import RowParallelLinear
    from mstar.model.components.norm import RMSNorm
    from .thinker_attention import Qwen3OmniAttention

    if type(attn) is not Qwen3OmniAttention or type(norm) is not RMSNorm or type(attn.o_proj) is not RowParallelLinear:
        return False
    proj = attn.o_proj
    if (
        attn.training
        or norm.training
        or torch.is_grad_enabled()
        or not attn._fused_mrope_gfx942
        or not attn.use_mrope
        or proj.comm_group.world_size != 1
        or proj.bias is not None
        or norm.gemma_mode
        or norm.variance_epsilon != 1e-6
    ):
        return False
    if residual.ndim != 2 or residual.shape[0] not in (1, 4, 8) or residual.shape[1] != 2048:
        return False
    if proj.weight.shape != (2048, 4096) or norm.weight.shape != (2048,):
        return False
    return (
        all(
            t.dtype == torch.bfloat16 and t.device == residual.device and t.is_contiguous()
            for t in (residual, proj.weight, norm.weight)
        )
        and residual.is_cuda
    )
