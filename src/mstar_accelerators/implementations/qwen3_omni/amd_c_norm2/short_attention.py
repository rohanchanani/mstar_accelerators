"""gfx942 launch schedule for the predictor's existing short NHD attention."""
from functools import cache

import torch


@cache
def _gfx942(device_index):
    return torch.cuda.get_device_properties(device_index).gcnArchName.split(':')[0] == 'gfx942'


@torch.compiler.disable
def short_decode_attn_nhd(q, k_cache, v_cache, cache_len, sm_scale=None):
    from mstar.utils.attention import _decode_attn_nhd_kernel, decode_attn_nhd

    squeeze = q.ndim == 4 and q.shape[1] == 1
    qq = q.squeeze(1) if squeeze else q
    if not (
        qq.ndim == 3 and qq.shape[0] in (1, 2, 4, 8)
        and qq.shape[1:] == (16, 128) and qq.dtype == torch.bfloat16
        and qq.is_cuda and torch.version.hip and _gfx942(qq.device.index)
        and k_cache.ndim == v_cache.ndim == 4
        and k_cache.shape == v_cache.shape
        and k_cache.shape[0] == qq.shape[0]
        and k_cache.shape[2:] == (8, 128)
        and k_cache.dtype == v_cache.dtype == qq.dtype
        and k_cache.device == v_cache.device == qq.device
        and 1 <= cache_len <= min(16, k_cache.shape[1])
        and qq.stride(-1) == 1
    ):
        return decode_attn_nhd(q, k_cache, v_cache, cache_len, sm_scale)
    out = torch.empty_like(qq)
    _decode_attn_nhd_kernel.fn[(qq.shape[0], 16)](
        qq, k_cache, v_cache, out, cache_len,
        128 ** -0.5 if sm_scale is None else sm_scale,
        *qq.stride(), *k_cache.stride(), *v_cache.stride(), *out.stride(),
        H_q=16, H_kv=8, GROUP=2, D=128, BLOCK_N=16,
        num_warps=1, num_stages=2,
    )
    return out.unsqueeze(1) if squeeze else out
