"""Optional ordinary Torch Whisper math over admitted contiguous state.

Weights and arithmetic may use FP64 while the generated logical outputs and
runtime-owned KV remain FP32. The optimizer supplies feed-forward computation,
including any partition and collective; this module owns no process group,
request state, sampler offset or serving lifecycle.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.engine.resources.sequence import DenseSequenceView
from mstar.model.whisper.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, POS


def column_major_weights(binding, weights):
    """Cache logical 2D weights in column-major storage, preserving aliases."""
    key = (column_major_weights, id(weights))
    if key not in binding.prepared:
        converted, identities = {}, {}
        for name, weight in weights.items():
            if weight.ndim == 2:
                identity = id(weight)
                if identity not in identities:
                    identities[identity] = weight.t().contiguous().t()
                converted[name] = identities[identity]
            else:
                converted[name] = weight
        binding.prepared[key] = converted
    return binding.prepared[key]


def linear(weights, prefix, x):
    return F.linear(x, weights[prefix + ".weight"], weights[prefix + ".bias"])


def normalize(weights, prefix, x):
    return F.layer_norm(x, (x.shape[-1],), weights[prefix + ".weight"],
                        weights[prefix + ".bias"], eps=1.0e-5)


def feed_forward(weights, prefix, x):
    """The unpartitioned FFN, useful for the replicated encoder realization."""
    return linear(weights, prefix + ".fc2",
                  F.gelu(linear(weights, prefix + ".fc1", x), approximate="none"))


def attention(q, k, v, heads, *, mask=None, scale_queries=False):
    queries, width = q.shape
    keys, head_dim = k.shape[0], width // heads
    if scale_queries:
        q = q * head_dim ** -0.5
    q = q.reshape(queries, heads, head_dim).transpose(0, 1)
    k = k.reshape(keys, heads, head_dim).transpose(0, 1)
    v = v.reshape(keys, heads, head_dim).transpose(0, 1)
    scores = q @ k.transpose(-1, -2)
    if not scale_queries:
        scores = scores * head_dim ** -0.5
    if mask is not None:
        scores = scores + mask.to(dtype=scores.dtype)
    return (scores.softmax(-1) @ v).transpose(0, 1).reshape(queries, width)


def sdpa_attention(q, k, v, heads, *, mask=None, scale_queries=False):
    """Tiled Torch attention, selected explicitly by an implementation."""
    queries, width = q.shape
    keys, head_dim = k.shape[0], width // heads
    if scale_queries:
        q = q * head_dim ** -0.5
    q = q.reshape(queries, heads, head_dim).transpose(0, 1)
    k = k.reshape(keys, heads, head_dim).transpose(0, 1)
    v = v.reshape(keys, heads, head_dim).transpose(0, 1)
    attended = F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        attn_mask=None if mask is None else mask.to(dtype=q.dtype),
        dropout_p=0.0, scale=1.0 if scale_queries else head_dim ** -0.5,
    ).squeeze(0)
    return attended.transpose(0, 1).reshape(queries, width)


def sharded_greedy_readout(config, weights, group):
    """Borrow vocabulary row intervals from the prepared weights and TP group."""
    start = config.vocab_size * group.rank // group.world_size
    end = config.vocab_size * (group.rank + 1) // group.world_size
    output_weight = weights["decoder.proj_out.weight"][start:end]
    suppressed = {
        walk: tuple(index - start for index in config.suppressed_tokens(walk)
                    if start <= index < end)
        for walk in ("prefill", "decode")
    }

    def select_tokens(x, call):
        walk = call.graph_walk
        logits = F.linear(x, output_weight)
        if suppressed[walk]:
            logits[:, suppressed[walk]] = float("-inf")
        values, indices = logits.max(dim=-1)
        # Widen F32 maxima exactly and carry vocabulary indices in FP64. Rank
        # order is contiguous vocabulary order, preserving first-index ties.
        candidate = torch.stack((values.to(torch.float64),
                                 (indices + start).to(torch.float64)), dim=-1)
        candidates = group.all_gather(candidate, dim=0)
        winner = candidates[:, 0].argmax(dim=0)
        return candidates[winner, 1].to(torch.int64).reshape(1)

    return select_tokens


class WhisperTorch:
    """Retained encoder/attention math with an implementation-selected FFN."""

    def __init__(self, config, weights, resources, *, ffn=None, encoder_attention=attention,
                 decoder_attention=attention, token_selector=None):
        self.config, self.weights = config, weights
        self.ffn = ffn or (lambda prefix, x: feed_forward(weights, prefix, x))
        self.encoder_attention = encoder_attention
        self.decoder_attention = decoder_attention
        self.token_selector = token_selector
        self.views = {key: resources[key].view(DenseSequenceView())
                      for key in (KV_CACHE, CROSS_KV_CACHE) if key in resources}
        self.dtype = next(iter(weights.values())).dtype

    @torch.no_grad()
    def encoder(self, call):
        features, = call.inputs.audio_features
        x = features.to(self.dtype).unsqueeze(0)
        for name, stride in (("conv1", 1), ("conv2", 2)):
            prefix = "encoder." + name
            x = F.gelu(F.conv1d(x, self.weights[prefix + ".weight"],
                               self.weights[prefix + ".bias"], stride=stride, padding=1),
                       approximate="none")
        x = x.squeeze(0).transpose(0, 1) + self.weights["encoder.embed_positions.weight"]
        for layer in range(self.config.encoder_layers):
            prefix = f"encoder.layers.{layer}"
            normalized = normalize(self.weights, prefix + ".self_attn_layer_norm", x)
            q, k, v = (linear(self.weights, prefix + ".self_attn." + projection, normalized)
                       for projection in ("q_proj", "k_proj", "v_proj"))
            attended = self.encoder_attention(q, k, v, self.config.encoder_attention_heads,
                                             scale_queries=True)
            x = x + linear(self.weights, prefix + ".self_attn.out_proj", attended)
            x = x + self.ffn(prefix, normalize(self.weights, prefix + ".final_layer_norm", x))
        return {"encoder_states": (normalize(self.weights, "encoder.layer_norm", x),)}

    @torch.no_grad()
    def decoder(self, call):
        ids, = call.inputs.input_ids
        positions = getattr(call.plan_results, POS)["main"]
        x = F.embedding(ids, self.weights["decoder.embed_tokens.weight"])
        if self.config.scale_embedding:
            x = x * self.config.d_model ** 0.5
        x = x + F.embedding(positions.long(), self.weights["decoder.embed_positions.weight"])
        prefill = call.graph_walk == "prefill"
        self_write, = self.views[KV_CACHE].for_step(getattr(call.plan_results, KV_CACHE)["main"])
        self_read, = self.views[KV_CACHE].for_attention(getattr(call.plan_results, ATTN)["main"])
        cross_read, = self.views[CROSS_KV_CACHE].for_attention(getattr(call.plan_results, CROSS_ATTN)["main"])
        if prefill:
            encoder, = call.inputs.encoder_states
            encoder = encoder.to(self.dtype)
            cross_write, = self.views[CROSS_KV_CACHE].for_step(
                getattr(call.plan_results, CROSS_KV_CACHE)["main"])

        def write(row, layer, keys, values):
            target_k, target_v = row.keys[layer], row.values[layer]
            target_k.view(target_k.shape[0], -1).index_copy_(0, row.positions, keys.to(target_k.dtype))
            target_v.view(target_v.shape[0], -1).index_copy_(0, row.positions, values.to(target_v.dtype))

        def read(row, layer):
            length = row.valid_length
            keys = row.keys[layer][:length].reshape(length, -1).to(self.dtype)
            values = row.values[layer][:length].reshape(length, -1).to(self.dtype)
            return keys, values, row.mask[:, :length]

        for layer in range(self.config.decoder_layers):
            prefix = f"decoder.layers.{layer}"
            if prefill:
                keys = linear(self.weights, prefix + ".encoder_attn.k_proj", encoder)
                values = linear(self.weights, prefix + ".encoder_attn.v_proj", encoder)
                write(cross_write, layer, keys, values)
            normalized = normalize(self.weights, prefix + ".self_attn_layer_norm", x)
            q, k, v = (linear(self.weights, prefix + ".self_attn." + projection, normalized)
                       for projection in ("q_proj", "k_proj", "v_proj"))
            write(self_write, layer, k, v)
            k, v, mask = read(self_read, layer)
            x = x + linear(self.weights, prefix + ".self_attn.out_proj",
                           self.decoder_attention(q, k, v, self.config.decoder_attention_heads, mask=mask))
            normalized = normalize(self.weights, prefix + ".encoder_attn_layer_norm", x)
            q = linear(self.weights, prefix + ".encoder_attn.q_proj", normalized)
            k, v, mask = read(cross_read, layer)
            x = x + linear(self.weights, prefix + ".encoder_attn.out_proj",
                           self.decoder_attention(q, k, v, self.config.decoder_attention_heads, mask=mask))
            x = x + self.ffn(prefix, normalize(self.weights, prefix + ".final_layer_norm", x))
        x = normalize(self.weights, "decoder.layer_norm", x)[-1:]
        if self.token_selector is not None:
            return {"tokens": self.token_selector(x, call)}
        logits = F.linear(x, self.weights["decoder.proj_out.weight"])
        suppressed = self.config.suppressed_tokens(call.graph_walk)
        if suppressed:
            logits[:, suppressed] = float("-inf")
        return {"tokens": logits.argmax(-1)}
