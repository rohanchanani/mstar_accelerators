"""Editable implementation of one logical node. Assessment belongs to the harness."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from mstar_accelerators.backends.graph import GraphProgram, NativeGraphProvider, prepare_graph_programs
from mstar_accelerators.implementations.whisper.pi_v3d.graph import encoder_graph
from mstar.model.whisper.results import EncoderResult
from mstar.model.whisper.submodules import WhisperEncoderSubmodule
from mstar_accelerators import Acceleration, Eq, Implementation, When
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

    return prepare_graph_programs(
        binding, resources,
        {"prefill": GraphProgram(partial(encoder_graph, binding.submodule.config), slots=1)},
        provider=NativeGraphProvider(), call_binding=defaults,
    )


acceleration = Acceleration(
    reference=WhisperEncoderSubmodule,
    when=When(
        Eq('node_name', 'audio_encoder'),
        Eq('device_type', 'mstar_pi'),
        Eq('dtype', torch.float32),
        When(Eq('graph_walk', 'prefill'), Eq('tp_size', 1), Eq('sp_size', 1)),
        Eq('batch_size', 1),
        Eq('config.d_model', 384),
        Eq('config.encoder_layers', 4),
        Eq('config.encoder_attention_heads', 6),
        Eq('config.encoder_ffn_dim', 1536),
        Eq('config.num_mel_bins', 80),
        Eq('config.max_source_positions', 1500),
        Eq('config.activation_function', 'gelu'),
    ),
    implementation=Implementation(prepare=prepare, resources=default_resource_requests),
)
