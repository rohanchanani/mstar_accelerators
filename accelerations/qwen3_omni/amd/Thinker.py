from __future__ import annotations
from dataclasses import dataclass
from mstar.engine.resources.sampler.plan import SamplingPlan
from mstar.engine.resources.sequence_view import AttentionUse
from mstar.engine.resources.sequence_view import SequenceUse
from mstar.engine.tensor_forward import CapturePolicy, CompilePolicy
from mstar.model.qwen3_omni.results import ThinkerResult
from mstar.model.qwen3_omni.thinker_logical import ThinkerLogical
from mstar_accelerators.backends.torch.tensor_forward import tensor_resource_requests
from typing import Literal
from mstar_accelerators import Acceleration, Eq, Implementation, Range, TensorForwardExecution, When
import torch
from mstar_accelerators.implementations.qwen3_omni.amd.thinker_preparation import prepare_thinker

@dataclass(frozen=True)
class Inputs:
    input_ids: tuple[torch.Tensor | None, ...]  # per-row shape ('tokens',); present on prefill_text, thinker_decode; absent rows are None, all absent is empty
    media_embeds: tuple[torch.Tensor | None, ...]  # per-row shape ('media_tokens', 2048); present on prefill_audio, prefill_vision; absent rows are None, all absent is empty
    positions: tuple[torch.Tensor, ...]  # per-row shape (3, 'tokens')
    masks: tuple[torch.Tensor, ...]  # per-row shape (2, 'tokens')
    deepstack_0: tuple[torch.Tensor | None, ...]  # per-row shape ('tokens', 2048); present on prefill_vision; absent rows are None, all absent is empty
    deepstack_1: tuple[torch.Tensor | None, ...]  # per-row shape ('tokens', 2048); present on prefill_vision; absent rows are None, all absent is empty
    deepstack_2: tuple[torch.Tensor | None, ...]  # per-row shape ('tokens', 2048); present on prefill_vision; absent rows are None, all absent is empty


class Resources:
    """Model-owned identities; physical views are chosen in implementation.resources."""
    thinker_kv = 'thinker_kv'
    thinker_attn = 'thinker_attn'
    thinker_pos = 'thinker_pos'
    thinker_sampler = 'thinker_sampler'


@dataclass(frozen=True)
class Plans:
    thinker_kv: dict[str, SequenceUse]
    thinker_attn: dict[str, AttentionUse]
    thinker_pos: dict[str, torch.Tensor]
    thinker_sampler: SamplingPlan


@dataclass(frozen=True)
class Call:
    graph_walk: Literal['prefill_audio', 'prefill_text', 'prefill_vision', 'thinker_decode']
    inputs: Inputs
    outputs: ThinkerResult
    plan_results: Plans
    row_walks: tuple[str, ...]


acceleration = Acceleration(
    reference=ThinkerLogical,
    when=When(Eq("node_name", "Thinker"), Eq("device_type", "cuda"), Eq("platform", "rocm"),
              Eq("dtype", torch.bfloat16), Eq("tp_size", 1), Eq("sp_size", 1), Range("batch_size", 1, 32)),
    implementation=Implementation(prepare=prepare_thinker, resources=tensor_resource_requests,
        execution=TensorForwardExecution(
            compile=CompilePolicy(enabled=True, fullgraph=False, dynamic=None, platform_overrides=True),
            capture=CapturePolicy(enabled=True, required=False,
                compile=CompilePolicy(enabled=True, mode="max-autotune-no-cudagraphs", fullgraph=False,
                                      dynamic=False, platform_overrides=True)),
            autocast_dtype=torch.bfloat16)))
