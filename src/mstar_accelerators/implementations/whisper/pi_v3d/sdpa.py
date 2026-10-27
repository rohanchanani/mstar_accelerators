"""Optional Q1 F32 attention over views of the original runtime K/V storage."""
from __future__ import annotations

from .graph import _attention


def decode_attention(config):
    """Select a native physical path before preparing either decoder program."""
    heads = config.decoder_attention_heads
    target, source = config.max_target_positions, config.max_source_positions
    if (heads <= 0 or config.d_model != 64 * heads or
            not 1 <= target <= 1536 or not 1 <= source <= 1536):
        return None
    from mstar_pi import ggml
    supported = getattr(ggml.Graph, "supports_sdpa_decode_f32", None)
    if supported is None or not hasattr(ggml.Graph, "sdpa_decode_f32"):
        return None
    if not all(supported(heads, tokens, 1) for tokens in (target, source)):
        return None
    return q1_attention


def q1_attention(g, q, k, v, *, heads, width, queries, keys, mask=None, scale_queries=False):
    """Borrow SET_ROWS-returned aliases directly; retain the full mask domain."""
    if queries != 1 or width != 64 * heads or mask is None or scale_queries or not 1 <= keys <= 1536:
        return _attention(g, q, k, v, heads=heads, width=width, queries=queries,
                          keys=keys, mask=mask, scale_queries=scale_queries)
    # reshape preserves the original producer edges, including actual SET_ROWS
    # outputs for self attention. No second cache input or private cache exists.
    q = g.reshape(q, (64, heads, 1, 1))
    k = g.reshape(k, (64, heads, keys, 1))
    v = g.reshape(v, (64, heads, keys, 1))
    mask = g.reshape(mask, (keys, 1, 1, 1))
    output = g.sdpa_decode_f32(q, k, v, mask, 64 ** -0.5)
    return g.reshape(output, (width, 1))
