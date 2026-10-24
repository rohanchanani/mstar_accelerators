"""CPU physical resources for the executable reference, never a serving fallback."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.engine.resources.attn.planned import PlannedAttentionResource, PlannedCrossAttentionResource
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.binding import build_bound_resource
from mstar.engine.resources.sampler.resource import SamplerResource


def attention_program(q, plan, cache):
    outputs = []
    for row, view in enumerate(plan.contexts):
        begin, end = int(plan.qo_indptr[row]), int(plan.qo_indptr[row + 1])
        positions = torch.arange(view.start, view.start + view.length)
        pages = torch.tensor(view.page_idxs)[positions // plan.page_size]
        slots = positions % plan.page_size
        k, v = cache[pages, 0, slots], cache[pages, 1, slots]
        queries = q[begin:end]
        mask = None
        if plan.causal:
            qpos = view.length - (end - begin) + torch.arange(end - begin)
            mask = torch.arange(view.length)[None, :] <= qpos[:, None]
        result = F.scaled_dot_product_attention(
            queries.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            attn_mask=mask, dropout_p=0.0,
        )
        outputs.append(result.transpose(0, 1))
    return torch.cat(outputs, dim=0)


def paged_kv(spec, info):
    return KVManager(spec.config, spec.resource_key, info.joint_comm_group, info.transfer_engine_info,
                     torch.device("cpu"), info.kv_dtype, needs_remote_transfer=False, host_plan=True)


def attention(spec, info):
    cfg = info.dependency(spec.config.kv_cache).config
    return PlannedAttentionResource(spec.config.kv_cache, cfg.page_size, attention_program)


def cross_attention(spec, info):
    cfg = info.dependency(spec.config.kv_cache).config
    return PlannedCrossAttentionResource(spec.config.kv_cache, cfg.page_size,
                                         spec.config.query_kv_cache, spec.config.context_label,
                                         attention_program)


def positions(spec, info):
    return build_bound_resource(spec, info)


def _greedy(logits, plan):
    if any(config.temperature != 0 for config in plan.configs):
        raise ValueError("this reference qualification profile is greedy")
    return logits.argmax(dim=-1)


def greedy_sampler(spec, info):
    return SamplerResource(spec.vocab_size, spec.enable_repetion_penalty, torch.device("cpu"),
                           info.joint_comm_group, spec.enable_min_p, planned=True, sample=_greedy)
