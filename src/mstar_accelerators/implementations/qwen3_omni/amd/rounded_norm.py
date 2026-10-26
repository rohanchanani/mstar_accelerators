"""Predictor-local residual add then norm, retaining the BF16 materialization."""
import torch
import triton
import triton.language as tl

NORM_NUM_WARPS = 2


@triton.jit
def _rounded_add_norm(X, Delta, W, Y, R, sx: tl.constexpr, sd: tl.constexpr,
                      sy: tl.constexpr, sr: tl.constexpr, EPS: tl.constexpr,
                      H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * sx + col, col < H, other=0).to(tl.float32)
    delta = tl.load(Delta + row * sd + col, col < H, other=0).to(tl.float32)
    # This conversion is semantically required: norm consumes the rounded sum.
    rounded = (x + delta).to(tl.bfloat16, fp_downcast_rounding='rtne')
    r = rounded.to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(r * r, 0) / H + EPS)
    weight = tl.load(W + col, col < H, other=0).to(tl.float32)
    normalized = (r * inv_rms) * weight
    tl.store(R + row * sr + col, rounded, col < H)
    tl.store(Y + row * sy + col,
             normalized.to(tl.bfloat16, fp_downcast_rounding='rtne'), col < H)


def rounded_add_norm(x: torch.Tensor, delta: torch.Tensor,
                     weight: torch.Tensor, eps: float):
    """Return (norm(BF16(x+delta)), BF16(x+delta)); neither input is mutated."""
    assert x.dtype == delta.dtype == weight.dtype == torch.bfloat16
    assert x.shape == delta.shape and x.stride(-1) == delta.stride(-1) == 1
    h = x.shape[-1]
    assert weight.shape == (h,)
    xf, df = x.reshape(-1, h), delta.reshape(-1, h)
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    r = torch.empty_like(y)
    yf, rf = y.reshape(-1, h), r.reshape(-1, h)
    _rounded_add_norm[(xf.shape[0],)](
        xf, df, weight, yf, rf, xf.stride(0), df.stride(0),
        yf.stride(0), rf.stride(0), eps, h, triton.next_power_of_2(h),
        num_warps=NORM_NUM_WARPS, enable_fp_fusion=False,
    )
    return y, r
