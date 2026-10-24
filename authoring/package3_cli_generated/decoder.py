"""Editable implementation of one logical node. Assessment belongs to the harness."""
from __future__ import annotations

from dataclasses import dataclass
from mstar.engine.execution import Execution
from mstar.engine.resources.sampler.plan import SamplingPlan
from mstar.engine.resources.sequence_view import AttentionUse
from mstar.engine.resources.sequence_view import SequenceUse
from mstar.model.whisper.results import DecoderResult
from mstar.model.whisper.submodules import WhisperDecoderSubmodule
from mstar_accelerators import Acceleration, Eq, Implementation, Or, PreparedImplementation, When
from mstar_accelerators.compiler import bind_default_call, default_resource_requests
from typing import Literal
import torch


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


def prepare(binding, resources):
    defaults = bind_default_call(binding, resources, input_type=Inputs, call_type=Call, plan_type=Plans)

    def primitive(call: Call, execution: Execution) -> DecoderResult:
        return defaults.zero(call, execution)

    return PreparedImplementation(
        materialize=defaults.materialize, primitive=primitive,
        acquire=defaults.acquire, close=defaults.close,
    )


acceleration = Acceleration(
    reference=WhisperDecoderSubmodule,
    when=When(
        Eq('node_name', 'decoder'),
        Eq('device_type', 'cpu'),
        Eq('dtype', torch.float32),
        Or(When(Eq('graph_walk', 'decode'), Eq('tp_size', 2), Eq('sp_size', 1)), When(Eq('graph_walk', 'prefill'), Eq('tp_size', 2), Eq('sp_size', 1))),
    ),
    # An editable initial physical choice; the generated call is representation-neutral.
    implementation=Implementation(prepare=prepare, resources=default_resource_requests),
)
