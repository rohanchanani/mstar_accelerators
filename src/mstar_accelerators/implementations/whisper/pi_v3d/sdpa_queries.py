"""Optional bounded multi-query F32 attention over existing paged read values."""
from __future__ import annotations

from .graph import _attention


def prefill_attention(config):
    """Select before preparation; unsupported physical paths keep GPU attention."""
    heads = config.decoder_attention_heads
    target, source = config.max_target_positions, config.max_source_positions
    if (heads <= 0 or config.d_model != 64 * heads or
            not 1 <= target <= 1536 or not 1 <= source <= 1536):
        return None
    from mstar_pi import ggml
    supported = getattr(ggml.Graph, "supports_sdpa_queries_f32", None)
    if supported is None or not hasattr(ggml.Graph, "sdpa_queries_f32"):
        return None
    if not all(supported(heads, tokens, 4, 1) for tokens in (target, source)):
        return None
    return small_query_attention


def small_query_attention(g, q, k, v, *, heads, width, queries, keys, mask=None, scale_queries=False):
    """Preserve producer aliases and every physical mask/key entry; Q1 is unchanged."""
    if (not 2 <= queries <= 4 or width != 64 * heads or mask is None or
            scale_queries or not 1 <= keys <= 1536):
        return _attention(g, q, k, v, heads=heads, width=width, queries=queries,
                          keys=keys, mask=mask, scale_queries=scale_queries)
    q = g.reshape(q, (64, heads, queries, 1))
    k = g.reshape(k, (64, heads, keys, 1))
    v = g.reshape(v, (64, heads, keys, 1))
    mask = g.reshape(mask, (keys, queries, 1, 1))
    output = g.sdpa_queries_f32(q, k, v, mask, 64 ** -0.5)
    return g.reshape(output, (width, queries))
