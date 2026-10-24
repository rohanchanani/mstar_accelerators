"""A: acceleration-owned copy of the pinned AMD Talker physical program."""
from __future__ import annotations

from dataclasses import dataclass
from mstar.engine.resources.sampler.plan import SamplingPlan
from mstar.engine.resources.sequence_view import AttentionUse
from mstar.engine.resources.sequence_view import SequenceUse
from mstar.engine.tensor_forward import CapturePolicy, CompilePolicy
from mstar.model.qwen3_omni.logical import TalkerSubmodule
from mstar.model.qwen3_omni.results import TalkerResult
from mstar_accelerators import Acceleration, Eq, Implementation, Or, Range, TensorForwardExecution, When
from mstar_accelerators.backends.torch.tensor_forward import tensor_resource_requests
from mstar_accelerators.implementations.qwen3_omni.amd_a.preparation import prepare_program
from typing import Literal
import torch


@dataclass(frozen=True)
class Inputs:
    thinker_states: tuple[torch.Tensor, ...]  # per-row shape ('tokens', 2048)
    multimodal_mask: tuple[torch.Tensor | None, ...]  # per-row shape ('tokens',); present on talker_prefill; absent rows are None, all absent is empty
    codec_ids: tuple[torch.Tensor | None, ...]  # per-row shape ('tokens',); present on talker_last_prefill; absent rows are None, all absent is empty
    codec_embeds: tuple[torch.Tensor | None, ...]  # per-row shape ('tokens', 1024); present on talker_decode; absent rows are None, all absent is empty


class Resources:
    """Model-owned identities; physical views are chosen in implementation.resources."""
    talker_kv = 'talker_kv'
    talker_attn = 'talker_attn'
    talker_pos = 'talker_pos'
    talker_sampler = 'talker_sampler'
    code_sampler = 'code_sampler'


@dataclass(frozen=True)
class Plans:
    talker_kv: dict[str, SequenceUse]
    talker_attn: dict[str, AttentionUse]
    talker_pos: dict[str, torch.Tensor]
    talker_sampler: SamplingPlan
    code_sampler: SamplingPlan


@dataclass(frozen=True)
class Call:
    graph_walk: Literal['talker_decode', 'talker_last_prefill', 'talker_prefill']
    inputs: Inputs
    outputs: TalkerResult
    plan_results: Plans
    row_walks: tuple[str, ...]


acceleration = Acceleration(
    reference=TalkerSubmodule,
    when=When(
        Eq('node_name', 'Talker'),
        Eq('device_type', 'cuda'),
        Eq('platform', 'rocm'),
        Range('batch_size', 1, 32),
        Eq('dtype', torch.bfloat16),
        Or(When(Eq('graph_walk', 'talker_decode'), Eq('tp_size', 1), Eq('sp_size', 1)), When(Eq('graph_walk', 'talker_last_prefill'), Eq('tp_size', 1), Eq('sp_size', 1)), When(Eq('graph_walk', 'talker_prefill'), Eq('tp_size', 1), Eq('sp_size', 1))),
    ),
    implementation=Implementation(
        prepare=prepare_program,
        # Editable physical view and runtime numerical resource choices.
        resources=tensor_resource_requests,
        execution=TensorForwardExecution(
            compile=CompilePolicy(enabled=True, fullgraph=False, dynamic=None, platform_overrides=True),
            capture=CapturePolicy(
                enabled=True, required=False,
                compile=CompilePolicy(enabled=True, mode="max-autotune-no-cudagraphs",
                                      fullgraph=False, dynamic=False, platform_overrides=True),
            ),
            autocast_dtype=torch.bfloat16,
        ),
    ),
)
