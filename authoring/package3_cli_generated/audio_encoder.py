"""Editable implementation of one logical node. Assessment belongs to the harness."""
from __future__ import annotations

from dataclasses import dataclass
from mstar.engine.execution import Execution
from mstar.model.whisper.results import EncoderResult
from mstar.model.whisper.submodules import WhisperEncoderSubmodule
from mstar_accelerators import Acceleration, Eq, Implementation, Or, PreparedImplementation, When
from mstar_accelerators.compiler import bind_default_call, default_resource_requests
from typing import Literal
import torch


@dataclass(frozen=True)
class Inputs:
    audio_features: tuple[torch.Tensor, ...]  # per-row shape (80, 3000)


class Resources:
    """Model-owned identities; physical views are chosen in implementation.resources."""


@dataclass(frozen=True)
class Plans:
    pass


@dataclass(frozen=True)
class Call:
    graph_walk: Literal['prefill']
    inputs: Inputs
    outputs: EncoderResult
    plan_results: Plans
    row_walks: tuple[str, ...]


def prepare(binding, resources):
    defaults = bind_default_call(binding, resources, input_type=Inputs, call_type=Call, plan_type=Plans)

    def primitive(call: Call, execution: Execution) -> EncoderResult:
        return defaults.zero(call, execution)

    return PreparedImplementation(
        materialize=defaults.materialize, primitive=primitive,
        acquire=defaults.acquire, close=defaults.close,
    )


acceleration = Acceleration(
    reference=WhisperEncoderSubmodule,
    when=When(
        Eq('node_name', 'audio_encoder'),
        Eq('device_type', 'cpu'),
        Eq('dtype', torch.float32),
        When(Eq('graph_walk', 'prefill'), Eq('tp_size', 1), Eq('sp_size', 1)),
    ),
    # An editable initial physical choice; the generated call is representation-neutral.
    implementation=Implementation(prepare=prepare, resources=default_resource_requests),
)
