from __future__ import annotations
from dataclasses import dataclass
from mstar.engine.tensor_forward import CapturePolicy, CompilePolicy
from mstar.model.qwen3_omni.audio_logical import AudioEncoderLogical
from mstar.model.qwen3_omni.results import AudioResult
from mstar_accelerators.backends.torch.tensor_forward import tensor_resource_requests
from typing import Literal
from mstar_accelerators import Acceleration, Eq, Implementation, TensorForwardExecution, When
import torch
from mstar_accelerators.implementations.qwen3_omni.amd.audio_program import build_audio

@dataclass(frozen=True)
class Inputs:
    audio_features: tuple[torch.Tensor, ...]  # per-row shape ('mel_bins', 'frames')
    audio_seqlens: tuple[torch.Tensor, ...]  # per-row shape ('segments',)


class Resources:
    """Model-owned identities; physical views are chosen in implementation.resources."""


@dataclass(frozen=True)
class Plans:
    pass


@dataclass(frozen=True)
class Call:
    graph_walk: Literal['prefill_audio']
    inputs: Inputs
    outputs: AudioResult
    plan_results: Plans
    row_walks: tuple[str, ...]


# Preserve grouped scheduling: the existing runner decomposes this non-batched
# program into per-request forwards. Deployment admission bounds the group size.
acceleration = Acceleration(
    reference=AudioEncoderLogical,
    when=When(Eq("node_name", "audio_encoder"), Eq("device_type", "cuda"), Eq("platform", "rocm"),
              Eq("dtype", torch.bfloat16), Eq("tp_size", 1), Eq("sp_size", 1)),
    implementation=Implementation(resources=tensor_resource_requests,
        execution=TensorForwardExecution(build=build_audio, bind_result=None, compile=CompilePolicy(enabled=True, dynamic=None, platform_overrides=True, fallback_to_eager=True),
                                         autocast_dtype=torch.bfloat16)))
