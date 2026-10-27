"""Logical Whisper decoder records shared by CPU physical implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from mstar.engine.resources.sampler.plan import SamplingPlan
from mstar.engine.resources.sequence_view import AttentionUse, SequenceUse
from mstar.model.whisper.results import DecoderResult


@dataclass(frozen=True)
class Inputs:
    input_ids: tuple[torch.Tensor, ...]  # per-row shape ('tokens',)
    encoder_states: tuple[torch.Tensor | None, ...]  # per-row shape (1500, 384); present on prefill; absent rows are None, all absent is empty


class Resources:
    """Model-owned identities; physical views are chosen in implementation.resources."""
    kv_cache = 'kv_cache'
    attn = 'attn'
    cross_kv_cache = 'cross_kv_cache'
    cross_attn = 'cross_attn'
    positions = 'positions'
    sampler = 'sampler'


@dataclass(frozen=True)
class Plans:
    kv_cache: dict[str, SequenceUse]
    attn: dict[str, AttentionUse]
    cross_kv_cache: dict[str, SequenceUse]
    cross_attn: dict[str, AttentionUse]
    positions: dict[str, torch.Tensor]
    sampler: SamplingPlan


@dataclass(frozen=True)
class Call:
    graph_walk: Literal['decode', 'prefill']
    inputs: Inputs
    outputs: DecoderResult
    plan_results: Plans
    row_walks: tuple[str, ...]
