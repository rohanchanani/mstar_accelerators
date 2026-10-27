"""Editable implementation of one logical node. Assessment belongs to the harness."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from mstar_accelerators.backends.graph import GraphProgram, prepare_graph_programs
from mstar_accelerators.backends.graph import PagedLayout
from mstar_accelerators.implementations.whisper.pi_v3d.graph import decoder_graph
from mstar_accelerators.implementations.whisper.pi_v3d.sdpa_queries import prefill_attention
from mstar.engine.resources.sampler.plan import SamplingPlan
from mstar.engine.resources.sequence_view import AttentionUse
from mstar.engine.resources.sequence_view import SequenceUse
from mstar.model.whisper.results import DecoderResult
from mstar.model.whisper.submodules import WhisperDecoderSubmodule
from mstar_accelerators import Acceleration, AllRows, Eq, Implementation, When
from mstar_accelerators.compiler import bind_default_call, default_resource_requests
from mstar_accelerators.implementations.whisper.pi_v3d.q8 import Q8ReadoutProvider, q8_readout
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

    layout = PagedLayout.from_resources(binding, resources)
    attention = prefill_attention(binding.submodule.config)
    return prepare_graph_programs(
        binding, resources,
        {"prefill": GraphProgram(partial(decoder_graph, binding.submodule.config, layout, prefill=True, readout=q8_readout, attention=attention), slots=2)},
        provider=Q8ReadoutProvider(binding), call_binding=defaults,
    )


acceleration = Acceleration(
    reference=WhisperDecoderSubmodule,
    when=When(
        Eq('node_name', 'decoder'),
        Eq('device_type', 'mstar_pi'),
        Eq('dtype', torch.float32),
        When(Eq('graph_walk', 'prefill'), Eq('tp_size', 1), Eq('sp_size', 1)),
        Eq('batch_size', 1),
        Eq('token_lengths', (4,)),
        AllRows(Resources.sampler, When(Eq('temperature', 0.0), Eq('repetition_penalty', 1.0), Eq('min_tokens', 0))),
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
    implementation=Implementation(prepare=prepare, resources=default_resource_requests),
)
