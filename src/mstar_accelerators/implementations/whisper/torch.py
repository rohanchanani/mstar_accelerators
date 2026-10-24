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


class WhisperTorch:
    """Retained encoder/attention math with an implementation-selected FFN."""

    def __init__(self, config, weights, resources, *, ffn=None):
        self.config, self.weights = config, weights
        self.ffn = ffn or (lambda prefix, x: feed_forward(weights, prefix, x))
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
            attended = attention(q, k, v, self.config.encoder_attention_heads, scale_queries=True)
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
                           attention(q, k, v, self.config.decoder_attention_heads, mask=mask))
            normalized = normalize(self.weights, prefix + ".encoder_attn_layer_norm", x)
            q = linear(self.weights, prefix + ".encoder_attn.q_proj", normalized)
            k, v, mask = read(cross_read, layer)
            x = x + linear(self.weights, prefix + ".encoder_attn.out_proj",
                           attention(q, k, v, self.config.decoder_attention_heads, mask=mask))
            x = x + self.ffn(prefix, normalize(self.weights, prefix + ".final_layer_norm", x))
        x = normalize(self.weights, "decoder.layer_norm", x)[-1:]
        logits = F.linear(x, self.weights["decoder.proj_out.weight"])
        suppressed = self.config.suppressed_tokens(call.graph_walk)
        if suppressed:
            logits[:, suppressed] = float("-inf")
        return {"tokens": logits.argmax(-1)}
