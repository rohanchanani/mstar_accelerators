"""Editable implementation of one logical node. Assessment belongs to the harness."""
from __future__ import annotations

from dataclasses import dataclass
from mstar.engine.resources.sampler.plan import SamplingPlan
from mstar.engine.resources.sequence_view import AttentionUse
from mstar.engine.resources.sequence_view import SequenceUse
from mstar.engine.tensor_forward import CapturePolicy, CompilePolicy
from mstar.model.qwen3_omni.logical import TalkerSubmodule
from mstar.model.qwen3_omni.results import TalkerResult
from mstar_accelerators import Acceleration, Eq, Implementation, Or, TensorForwardExecution, When
from mstar_accelerators.backends.torch.tensor_forward import tensor_resource_requests
from mstar_accelerators.compiler import pack_tensor_result
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


class Program(torch.nn.Module):
    """Implementation-owned tensor packing, computation and capture recipes."""

    def __init__(self, binding, resources):
        super().__init__()
        self.device = binding.device
        self.dtype = binding.dtype
        self.contract = binding.submodule.logical_contract()
        self.resources = resources
        # Prepare physical weights and resource views here; keep the logical shell unchanged.

    def preprocess(self, graph_walk, engine_inputs, inputs):
        # Pack the already-interpreted input rows described by Inputs into forward kwargs.
        # Plans/Call describe logical uses; choose physical views through self.resources.
        return {'batch_size': len(inputs)}

    def forward_batched(self, graph_walk, engine_inputs, *, batch_size):
        # Replace this executable zero program with the intended physical computation.
        result = TalkerResult(
            codec_tokens=torch.zeros((batch_size, 16), dtype=torch.int64, device=self.device),
            codec_embeds=torch.zeros((batch_size, 1024), dtype=torch.float32, device=self.device),
            tokens=torch.zeros((batch_size,), dtype=torch.int64, device=self.device),
        )
        # Logical field names only; the model owns graph-edge and request semantics.
        return pack_tensor_result(result, self.contract)

    def forward(self, graph_walk, engine_inputs, *, batch_size):
        return self.forward_batched(graph_walk, engine_inputs, batch_size=batch_size)

    def can_batch(self, batch, inputs):
        return True

    def get_accelerator_graph_configs(self, device, tp_world_size=1):
        # Add explicit static shapes and a forward method before enabling capture below.
        return []

    def close(self):
        pass


def prepare(binding, resources):
    return Program(binding, resources)


acceleration = Acceleration(
    reference=TalkerSubmodule,
    when=When(
        Eq('node_name', 'Talker'),
        Eq('device_type', 'cuda'),
        Eq('dtype', torch.bfloat16),
        Or(When(Eq('graph_walk', 'talker_decode'), Eq('tp_size', 1), Eq('sp_size', 1)), When(Eq('graph_walk', 'talker_last_prefill'), Eq('tp_size', 1), Eq('sp_size', 1)), When(Eq('graph_walk', 'talker_prefill'), Eq('tp_size', 1), Eq('sp_size', 1))),
    ),
    implementation=Implementation(
        prepare=prepare,
        # Editable physical view and runtime numerical resource choices.
        resources=tensor_resource_requests,
        execution=TensorForwardExecution(
            compile=CompilePolicy(enabled=False),
            capture=CapturePolicy(
                enabled=False, required=False, compile=CompilePolicy(enabled=False),
            ),
            autocast_dtype=None,
        ),
    ),
)
