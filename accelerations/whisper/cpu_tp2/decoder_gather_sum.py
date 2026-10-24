"""FP64 FFN partitions over replicated runtime-owned FP32 attention state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from mstar.engine.resources.kv.config import KVSpec
from mstar.engine.resources.sampler.plan import SamplingPlan
from mstar.engine.resources.sequence import DenseSequenceView, Replicated
from mstar.engine.resources.sequence_view import AttentionUse, SequenceUse
from mstar.model.whisper.results import DecoderResult
from mstar.model.whisper.submodules import WhisperDecoderSubmodule
from mstar_accelerators import Acceleration, AllRows, Eq, Implementation, Or, When
from mstar_accelerators.backends.torch import TorchProgram, prepare_torch_programs
from mstar_accelerators.backends.weights import TensorShard, prepare_weights
from mstar_accelerators.compiler import bind_default_call
from mstar_accelerators.implementations.whisper.torch import WhisperTorch


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


def resource_requests(binding):
    return {spec.resource_key: DenseSequenceView(partition=Replicated())
            for spec in binding.resource_specs
            if isinstance(spec, KVSpec) and binding.node_name in spec.nodes}


def prepare(binding, resources):
    defaults = bind_default_call(
        binding, resources, input_type=Inputs, call_type=Call, plan_type=Plans,
    )
    config = binding.submodule.config
    shards = {
        f"{defaults.contract.weight_prefix}layers.{layer}.{name}": TensorShard(axis)
        for layer in range(config.decoder_layers)
        for name, axis in (("fc1.weight", 0), ("fc1.bias", 0), ("fc2.weight", 1))
    }
    weights = prepare_weights(binding, defaults.contract.weight_prefix,
                              dtype=torch.float64, shards=shards)

    def ffn(prefix, x):
        hidden = F.gelu(F.linear(x, weights[prefix + ".fc1.weight"],
                                weights[prefix + ".fc1.bias"]), approximate="none")
        partial = F.linear(hidden, weights[prefix + ".fc2.weight"])
        combined = binding.tp_group.all_gather(partial.unsqueeze(0), dim=0).sum(dim=0)
        return combined + weights[prefix + ".fc2.bias"]

    math = WhisperTorch(config, weights, resources, ffn=ffn)
    return prepare_torch_programs(
        binding, resources,
        {"prefill": TorchProgram(math.decoder, slots=2),
         "decode": TorchProgram(math.decoder, slots=2)},
        call_binding=defaults,
    )


acceleration = Acceleration(
    reference=WhisperDecoderSubmodule,
    when=When(
        Eq('node_name', 'decoder'),
        Eq('device_type', 'cpu'),
        Eq('dtype', torch.float32),
        Or(
            When(Eq('graph_walk', 'decode'), Eq('tp_size', 2), Eq('sp_size', 1),
                 Eq('token_lengths', (1,))),
            When(Eq('graph_walk', 'prefill'), Eq('tp_size', 2), Eq('sp_size', 1),
                 Eq('token_lengths', (4,))),
        ),
        Eq('batch_size', 1),
        AllRows(Resources.sampler, When(
            Eq('temperature', 0.0), Eq('top_k', 0), Eq('top_p', 1.0),
            Eq('min_p', 0.0), Eq('repetition_penalty', 1.0),
        )),
        Eq('config.d_model', 384),
        Eq('config.decoder_layers', 4),
        Eq('config.decoder_attention_heads', 6),
        Eq('config.decoder_ffn_dim', 1536),
        Eq('config.max_source_positions', 1500),
        Eq('config.max_target_positions', 448),
        Eq('config.vocab_size', 51865),
        Eq('config.activation_function', 'gelu'),
        Eq('config.scale_embedding', False),
    ),
    implementation=Implementation(prepare=prepare, resources=resource_requests, cost=lambda _facts: 2.0),
)
