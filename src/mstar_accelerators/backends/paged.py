"""Typed physical addressing derived from immutable runtime plans.

The padded read uses M*'s existing zero sink page.  This provider does not run
captured dummy writes, and real write maps explicitly exclude the sink.  A
capture implementation with writable sink padding needs its own read strategy.
"""
from dataclasses import dataclass

import torch
from mstar.engine.resources.attn.planned import AttentionPlan
from mstar.engine.resources.kv.plan import SINK_PAGE, KVWritePlan


@dataclass(frozen=True)
class WriteRows:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class ReadRows:
    keys: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor


def write_rows(plan: KVWritePlan, *, page_size: int, num_pages: int) -> WriteRows:
    pages, slots = plan.token_to_page, plan.token_to_cache
    if torch.any(pages <= SINK_PAGE) or torch.any(pages >= num_pages):
        raise ValueError("real KV writes must use allocated non-sink pages")
    if torch.any(slots < 0) or torch.any(slots >= page_size):
        raise ValueError("KV slot lies outside the physical page")
    rows = pages * (2 * page_size) + slots
    if rows.unique().numel() != rows.numel():
        raise ValueError("ggml set_rows requires distinct destinations")
    return WriteRows(rows, rows + page_size)


def read_rows(plan: AttentionPlan, *, capacity: int, num_pages: int) -> ReadRows:
    if len(plan.contexts) != 1:
        raise ValueError("the B1 template needs one attention context")
    view = plan.contexts[0]
    q = int(plan.qo_indptr[-1])
    if not 0 < view.length <= capacity:
        raise ValueError("attention context does not fit the fixed template")
    positions = torch.arange(view.start, view.start + view.length, dtype=torch.long)
    pages = torch.tensor(view.page_idxs, dtype=torch.long)[positions // plan.page_size]
    if torch.any(pages <= SINK_PAGE) or torch.any(pages >= num_pages):
        raise ValueError("attention reads require allocated non-sink context pages")
    rows = pages * (2 * plan.page_size) + positions % plan.page_size
    key_rows = torch.full((capacity,), SINK_PAGE * 2 * plan.page_size, dtype=torch.int32)
    value_rows = torch.full((capacity,), (SINK_PAGE * 2 + 1) * plan.page_size, dtype=torch.int32)
    key_rows[:view.length] = rows.to(torch.int32)
    value_rows[:view.length] = (rows + plan.page_size).to(torch.int32)
    visible = torch.arange(capacity)[None, :] < view.length
    visible = visible.expand(q, capacity)
    if plan.causal:
        if q > view.length:
            raise ValueError("causal query exceeds its context")
        last_visible = view.length - q + torch.arange(q)[:, None]
        visible = visible & (torch.arange(capacity)[None, :] <= last_visible)
    mask = torch.zeros((q, capacity), dtype=torch.float32).masked_fill_(~visible, float("-inf"))
    return ReadRows(key_rows, value_rows, mask)
