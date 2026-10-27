"""Editable implementation of one logical node. Assessment belongs to the harness."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from mstar_accelerators.backends.graph import GraphProgram, NativeGraphProvider, prepare_graph_programs
from mstar.engine.resources.kv.config import KVSpec
from mstar.engine.resources.sequence import DenseSequenceView
from mstar_accelerators.backends.graph import dense_graph_bindings, dense_views
from mstar_accelerators.implementations.whisper.pi_v3d.dense import dense_decoder_graph
from mstar.engine.resources.sampler.plan import SamplingPlan
from mstar.engine.resources.sequence_view import AttentionUse
from mstar.engine.resources.sequence_view import SequenceUse
from mstar.model.whisper.results import DecoderResult
from mstar.model.whisper.submodules import WhisperDecoderSubmodule
from mstar_accelerators import Acceleration, AllRows, Eq, Implementation, When
from mstar_accelerators.compiler import bind_default_call
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


def resource_requests(binding):
    return {spec.resource_key: DenseSequenceView() for spec in binding.resource_specs
            if isinstance(spec, KVSpec) and binding.node_name in spec.nodes}


def prepare(binding, resources):
    defaults = bind_default_call(binding, resources, input_type=Inputs, call_type=Call, plan_type=Plans)

    views = dense_views(binding, resources)
    return prepare_graph_programs(
        binding, resources,
        {"decode": GraphProgram(partial(dense_decoder_graph, binding.submodule.config), slots=2)},
        provider=NativeGraphProvider(), call_binding=defaults,
        input_bindings=partial(dense_graph_bindings, views=views),
    )


acceleration = Acceleration(
    reference=WhisperDecoderSubmodule,
    when=When(
        Eq('node_name', 'decoder'),
        Eq('device_type', 'mstar_pi'),
        Eq('dtype', torch.float32),
        When(Eq('graph_walk', 'decode'), Eq('tp_size', 1), Eq('sp_size', 1)),
        Eq('batch_size', 1),
        Eq('token_lengths', (1,)),
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
    implementation=Implementation(prepare=prepare, resources=resource_requests),
)
