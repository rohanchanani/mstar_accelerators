"""The complete FP32 Whisper programs expressed in the actual ggml API.

Tensor dimensions in this file are ggml's fastest-first ``ne``.  There is no
intermediate graph language: ``g`` is the provider's native Graph.  The optional
argument exists to run this exact construction through the host test evaluator.
"""
from __future__ import annotations

from typing import Any

from mstar.model.whisper.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, POS, SAMPLER, WhisperModelConfig
from mstar_accelerators.backends.graph import PagedLayout


class _Weights:
    def __init__(self, graph: Any, config: WhisperModelConfig):
        from mstar.model.whisper.weights import weight_specs
        self.graph = graph
        self.specs = {spec.name: spec for spec in weight_specs(config)}
        self.handles: dict[str, Any] = {}

    def __call__(self, name: str):
        spec = self.specs[name]
        if spec.alias_of:
            return self(spec.alias_of)
        if name not in self.handles:
            self.handles[name] = self.graph.tensor(name, tuple(reversed(spec.shape)))
        return self.handles[name]


def _linear(g, w, prefix, x):
    return g.add(g.mul_mat(w(prefix + ".weight"), x), w(prefix + ".bias"))


def _norm(g, w, prefix, x):
    return g.add(g.mul(g.norm(x, eps=1.0e-5), w(prefix + ".weight")), w(prefix + ".bias"))


def _attention(g, q, k, v, *, heads: int, width: int, queries: int, keys: int, mask=None,
               scale_queries=False):
    head_dim = width // heads
    if scale_queries:
        # HF Whisper explicitly preserves this order before SDPA(scale=1).
        q = g.scale(q, head_dim ** -0.5)
    q = g.cont(g.permute(g.reshape(q, (head_dim, heads, queries)), (0, 2, 1, 3)))
    k = g.cont(g.permute(g.reshape(k, (head_dim, heads, keys)), (0, 2, 1, 3)))
    scores = g.mul_mat(k, q)
    probabilities = g.softmax(scores, mask=mask, scale=1.0 if scale_queries else head_dim ** -0.5)
    v = g.cont(g.permute(g.reshape(v, (head_dim, heads, keys)), (1, 2, 0, 3)))
    attended = g.mul_mat(v, probabilities)
    return g.reshape(g.cont(g.permute(attended, (0, 2, 1, 3))), (width, queries))


def _ffn(g, w, prefix, x):
    return _linear(g, w, prefix + ".fc2", g.gelu_erf(_linear(g, w, prefix + ".fc1", x)))


def _new_graph():
    from mstar_pi import ggml
    return ggml.Graph(max_nodes=4096)


def encoder_graph(config: WhisperModelConfig, *, graph=None):
    """One fixed-duration, single-request encoder, without FP16 im2col."""
    g = graph if graph is not None else _new_graph()
    w = _Weights(g, config)
    h, length = config.d_model, config.max_source_positions
    x = g.tensor("audio_features", (2 * length, config.num_mel_bins))
    for layer, channels, stride in (("conv1", config.num_mel_bins, 1), ("conv2", h, 2)):
        kernel = w("encoder." + layer + ".weight")
        columns = g.im2col(kernel, x, stride=stride, padding=1, dilation=1)
        # GGML's convenient conv1d helper uses F16 im2col.  Explicit F32
        # im2col and mul_mat preserve this implementation's FP32 policy.
        y = g.mul_mat(g.reshape(kernel, (3 * channels, h)), columns)
        y = g.gelu_erf(g.add(y, w("encoder." + layer + ".bias")))
        if layer == "conv1":
            x = g.cont(g.permute(y, (1, 0, 2, 3)))
    x = g.add(y, w("encoder.embed_positions.weight"))
    for i in range(config.encoder_layers):
        prefix = f"encoder.layers.{i}"
        normalized = _norm(g, w, prefix + ".self_attn_layer_norm", x)
        q, k, v = (_linear(g, w, prefix + ".self_attn." + projection, normalized)
                   for projection in ("q_proj", "k_proj", "v_proj"))
        attended = _attention(g, q, k, v, heads=config.encoder_attention_heads,
                              width=h, queries=length, keys=length, scale_queries=True)
        x = g.add(x, _linear(g, w, prefix + ".self_attn.out_proj", attended))
        x = g.add(x, _ffn(g, w, prefix, _norm(g, w, prefix + ".final_layer_norm", x)))
    g.output("encoder_states", _norm(g, w, "encoder.layer_norm", x))
    return g


def decoder_graph(config: WhisperModelConfig, layout: PagedLayout, *, prefill: bool,
                  graph=None, diagnostics: bool = False, sample_tokens=None, readout=None, attention=None):
    """A persistent Q=4 prefill or Q=1 decode template over runtime-owned KV.

    Maps and masks are invocation inputs derived from M*'s immutable plans.
    Fixed capacities make graph construction independent of request history.
    """
    g = graph if graph is not None else _new_graph()
    attend = _attention if attention is None else attention
    w = _Weights(g, config)
    h, q = config.d_model, 4 if prefill else 1
    target, source = config.max_target_positions, config.max_source_positions
    ids = g.tensor("input_ids", (q,), dtype="i64")
    positions = g.tensor(f"{POS}.main.ids", (q,), dtype="i32")
    ids = g.cast(ids, "i32")
    x = g.get_rows(w("decoder.embed_tokens.weight"), ids)
    if config.scale_embedding:
        x = g.scale(x, h ** 0.5)
    x = g.add(x, g.get_rows(w("decoder.embed_positions.weight"), positions))
    self_write_k = g.tensor(f"{KV_CACHE}.main.write.keys", (q,), dtype="i64")
    self_write_v = g.tensor(f"{KV_CACHE}.main.write.values", (q,), dtype="i64")
    self_read_k = g.tensor(f"{ATTN}.main.read.keys", (target,), dtype="i32")
    self_read_v = g.tensor(f"{ATTN}.main.read.values", (target,), dtype="i32")
    self_mask = g.tensor(f"{ATTN}.main.read.mask", (target, q))
    cross_read_k = g.tensor(f"{CROSS_ATTN}.main.read.keys", (source,), dtype="i32")
    cross_read_v = g.tensor(f"{CROSS_ATTN}.main.read.values", (source,), dtype="i32")
    cross_mask = g.tensor(f"{CROSS_ATTN}.main.read.mask", (source, q))
    if prefill:
        encoder = g.tensor("encoder_states", (h, source))
        cross_write_k = g.tensor(f"{CROSS_KV_CACHE}.main.write.keys", (source,), dtype="i64")
        cross_write_v = g.tensor(f"{CROSS_KV_CACHE}.main.write.values", (source,), dtype="i64")

    for i in range(config.decoder_layers):
        prefix = f"decoder.layers.{i}"
        self_cache = g.tensor(f"{KV_CACHE}.layer.{i}", (h, layout.rows(KV_CACHE)))
        cross_cache = g.tensor(f"{CROSS_KV_CACHE}.layer.{i}", (h, layout.rows(CROSS_KV_CACHE)))
        if prefill:
            ck = _linear(g, w, prefix + ".encoder_attn.k_proj", encoder)
            cv = _linear(g, w, prefix + ".encoder_attn.v_proj", encoder)
            cross_cache = g.set_rows(ck, cross_write_k, cross_cache)
            cross_cache = g.set_rows(cv, cross_write_v, cross_cache)

        normalized = _norm(g, w, prefix + ".self_attn_layer_norm", x)
        sq, sk, sv = (_linear(g, w, prefix + ".self_attn." + projection, normalized)
                      for projection in ("q_proj", "k_proj", "v_proj"))
        self_cache = g.set_rows(sk, self_write_k, self_cache)
        self_cache = g.set_rows(sv, self_write_v, self_cache)
        # The read depends on the aliases returned by both writes, not merely
        # on the shared base allocation.  This is a real ggml graph edge.
        sk = g.get_rows(self_cache, self_read_k)
        sv = g.get_rows(self_cache, self_read_v)
        attended = attend(g, sq, sk, sv, heads=config.decoder_attention_heads,
                              width=h, queries=q, keys=target, mask=self_mask)
        x = g.add(x, _linear(g, w, prefix + ".self_attn.out_proj", attended))

        normalized = _norm(g, w, prefix + ".encoder_attn_layer_norm", x)
        cq = _linear(g, w, prefix + ".encoder_attn.q_proj", normalized)
        ck = g.get_rows(cross_cache, cross_read_k)
        cv = g.get_rows(cross_cache, cross_read_v)
        attended = attend(g, cq, ck, cv, heads=config.decoder_attention_heads,
                              width=h, queries=q, keys=source, mask=cross_mask)
        x = g.add(x, _linear(g, w, prefix + ".encoder_attn.out_proj", attended))
        x = g.add(x, _ffn(g, w, prefix, _norm(g, w, prefix + ".final_layer_norm", x)))
    x = _norm(g, w, "decoder.layer_norm", x)
    # Last actual query, never a padded row. The model owns suppression and
    # effects; the implementation chooses how to realize its admitted policy.
    last = g.view(x, (h, 1), (4, 4 * h), offset=4 * h * (q - 1))
    logits = g.mul_mat(w("decoder.proj_out.weight"), last) if readout is None else readout(g, config, last)
    logits = g.add(logits, g.tensor(f"{SAMPLER}.logit_mask", (config.vocab_size, 1)))
    if diagnostics:
        g.output("logits", logits)
    tokens = g.cast(g.argmax(logits), "i64") if sample_tokens is None else sample_tokens(g, logits, SAMPLER)
    g.output("tokens", tokens)
    return g
