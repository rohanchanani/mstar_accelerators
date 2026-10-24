"""FP64 Torch encoder with the generated FP32 output boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from mstar.model.whisper.results import EncoderResult
from mstar.model.whisper.submodules import WhisperEncoderSubmodule
from mstar_accelerators import Acceleration, Eq, Implementation, When
from mstar_accelerators.backends.torch import TorchProgram, prepare_torch_programs
from mstar_accelerators.backends.weights import prepare_weights
from mstar_accelerators.compiler import bind_default_call, default_resource_requests
from mstar_accelerators.implementations.whisper.torch import WhisperTorch


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
    defaults = bind_default_call(
        binding, resources, input_type=Inputs, call_type=Call, plan_type=Plans,
    )
    weights = prepare_weights(binding, defaults.contract.weight_prefix, dtype=torch.float64)
    math = WhisperTorch(binding.submodule.config, weights, resources)
    return prepare_torch_programs(
        binding, resources, {"prefill": TorchProgram(math.encoder, slots=1)},
        call_binding=defaults,
    )


acceleration = Acceleration(
    reference=WhisperEncoderSubmodule,
    when=When(
        Eq('node_name', 'audio_encoder'),
        Eq('device_type', 'cpu'),
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
