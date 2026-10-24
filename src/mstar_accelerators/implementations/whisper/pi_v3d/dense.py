"""Whisper decode over contiguous views of runtime-owned sequence storage."""
from __future__ import annotations

from mstar.model.whisper.config import ATTN, CROSS_ATTN, KV_CACHE, POS, SAMPLER, WhisperModelConfig

from .graph import _Weights, _attention, _ffn, _linear, _new_graph, _norm


def dense_decoder_graph(config: WhisperModelConfig, *, graph=None, sample_tokens=None):
    """One-token decode with direct cache reads and explicit append dependencies."""
    g = graph if graph is not None else _new_graph()
    w = _Weights(g, config)
    h = config.d_model
    target, source = config.max_target_positions, config.max_source_positions
    ids = g.cast(g.tensor("input_ids", (1,), dtype="i64"), "i32")
    positions = g.tensor(f"{POS}.main.ids", (1,), dtype="i32")
    x = g.get_rows(w("decoder.embed_tokens.weight"), ids)
    if config.scale_embedding:
        x = g.scale(x, h ** 0.5)
    x = g.add(x, g.get_rows(w("decoder.embed_positions.weight"), positions))
    append = g.tensor(f"{KV_CACHE}.main.write.positions", (1,), dtype="i64")
    self_mask = g.tensor(f"{ATTN}.main.read.mask", (target, 1))
    cross_mask = g.tensor(f"{CROSS_ATTN}.main.read.mask", (source, 1))

    for i in range(config.decoder_layers):
        prefix = f"decoder.layers.{i}"
        keys = g.tensor(f"{KV_CACHE}.main.keys.{i}", (h, target))
        values = g.tensor(f"{KV_CACHE}.main.values.{i}", (h, target))
        normalized = _norm(g, w, prefix + ".self_attn_layer_norm", x)
        sq, sk, sv = (_linear(g, w, prefix + ".self_attn." + projection, normalized)
                      for projection in ("q_proj", "k_proj", "v_proj"))
        # Reading the returned aliases makes each append a graph predecessor.
        # A second input alias of the same allocation would not encode that edge.
        sk = g.set_rows(sk, append, keys)
        sv = g.set_rows(sv, append, values)
        attended = _attention(g, sq, sk, sv, heads=config.decoder_attention_heads,
                              width=h, queries=1, keys=target, mask=self_mask)
        x = g.add(x, _linear(g, w, prefix + ".self_attn.out_proj", attended))

        normalized = _norm(g, w, prefix + ".encoder_attn_layer_norm", x)
        cq = _linear(g, w, prefix + ".encoder_attn.q_proj", normalized)
        ck = g.tensor(f"{CROSS_ATTN}.main.read.keys.{i}", (h, source))
        cv = g.tensor(f"{CROSS_ATTN}.main.read.values.{i}", (h, source))
        attended = _attention(g, cq, ck, cv, heads=config.decoder_attention_heads,
                              width=h, queries=1, keys=source, mask=cross_mask)
        x = g.add(x, _linear(g, w, prefix + ".encoder_attn.out_proj", attended))
        x = g.add(x, _ffn(g, w, prefix, _norm(g, w, prefix + ".final_layer_norm", x)))

    x = _norm(g, w, "decoder.layer_norm", x)
    last = g.view(x, (h, 1), (4, 4 * h), offset=0)
    logits = g.mul_mat(w("decoder.proj_out.weight"), last)
    logits = g.add(logits, g.tensor(f"{SAMPLER}.logit_mask", (config.vocab_size, 1)))
    tokens = g.cast(g.argmax(logits), "i64") if sample_tokens is None else sample_tokens(g, logits, SAMPLER)
    g.output("tokens", tokens)
    return g
