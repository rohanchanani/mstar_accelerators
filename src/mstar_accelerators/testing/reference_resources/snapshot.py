"""CPU consumers of an existing invocation's exact resource-plan objects.

Qualification owns a separate physical copy of state, but no second page
allocator, request registry, position counter or RNG cursor.  Only the runtime
that produced ``plans`` can commit the logical transition.
"""
from __future__ import annotations

import torch

from mstar.engine.resources.kv.cache import KVCache
from mstar.model.whisper.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, POS, SAMPLER

from . import attention_program


class _Cursor:
    _default_label = "main"
    _default_layer_idx = None

    @property
    def default_label(self): return self._default_label
    def set_default_label(self, label): self._default_label = label
    def set_default_layer_idx(self, index): self._default_layer_idx = index


class _KV(_Cursor):
    def __init__(self, config):
        self.kv_cache = KVCache(config, torch.device("cpu"), torch.float32)
        self.plans = None

    def layer_view(self):
        return self.kv_cache.layer_view(self._default_layer_idx)

    def write_kv(self, k, v, layer_idx=None, label=None, return_tensor=False):
        write = self.plans[label or self._default_label].write_locations
        return self.kv_cache.write_tokens(
            layer_idx=self._default_layer_idx if layer_idx is None else layer_idx,
            k=k, v=v, page_idx=write.token_to_page, cache_idx=write.token_to_cache,
            return_tensor=return_tensor,
        )


class _Attention(_Cursor):
    requires_kv_write = True

    def run(self, q, label=None, kv_cache_layer=None, **kwargs):
        return attention_program(q, self.plans[label or self._default_label], kv_cache_layer)

    def select_last_hidden(self, hidden, label=None):
        plan = self.plans[label or self._default_label]
        return hidden[plan.qo_indptr[1:].long() - 1]


class _Positions:
    def pos_ids(self, label): return self.plans[label]


class _Sampler:
    def sample(self, request_ids, logits):
        if self.plan.apply_penalty or any(cfg.temperature != 0 for cfg in self.plan.configs):
            raise ValueError("reference snapshot consumer realizes only the greedy test profile")
        return logits.argmax(dim=-1)


class ReferencePlanConsumer:
    def __init__(self, runtime_resources):
        self.resources = {
            KV_CACHE: _KV(runtime_resources[KV_CACHE].config),
            CROSS_KV_CACHE: _KV(runtime_resources[CROSS_KV_CACHE].config),
            ATTN: _Attention(), CROSS_ATTN: _Attention(),
            POS: _Positions(), SAMPLER: _Sampler(),
        }

    def use(self, plans):
        for key in (KV_CACHE, CROSS_KV_CACHE, ATTN, CROSS_ATTN, POS):
            self.resources[key].plans = plans[key]
        self.resources[SAMPLER].plan = plans[SAMPLER]
