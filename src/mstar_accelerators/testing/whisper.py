"""Shared Whisper case helpers for the centralized qualification harness.

The case adapter owns reference/input interpretation. The central consumer owns
coverage and verdicts; the managed worker owns native initialization, exceptional
drain, fatal containment and shutdown. This module has no qualification CLI.
"""
from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np
import torch
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.binding import build_bound_resource, resolve_bound_specs
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.model.whisper.config import (
    ATTN,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    KV_CACHE,
    POS,
    SAMPLER,
)

from mstar_accelerators.testing.numerics import assert_close


def _observe(tensors, stream):
    staged = {key: torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
              for key, tensor in tensors.items()}
    for key, tensor in tensors.items():
        staged[key].copy_(tensor, non_blocking=True)
    event = torch.Event(device=stream.device, enable_timing=False)
    event.record(stream)
    event.synchronize()
    return {key: tensor.clone() for key, tensor in staged.items()}

def _compare(name, actual, expected, *, dtype=None):
    """Host-test compatibility; numerical policy is owned centrally."""
    return assert_close(name, actual, expected, dtype=dtype)

def _resource_specs(model):
    specs = model.get_node_resources()
    budgets = {KV_CACHE: 16 * 1024**2, CROSS_KV_CACHE: 56 * 1024**2}
    apply_yaml_overrides(specs, {"resources": {
        spec.resource_key: {"max_bytes": budgets[spec.resource_key], "prefix_cache": False}
        for spec in specs if spec.resource_key in budgets
    }})
    return specs

def _runtime_resources(model, device, requests, *, specs=None, session=None,
                       group=None, agreement=None):
    specs = _resource_specs(model) if specs is None else specs
    specs = resolve_bound_specs(specs, requests, dtype=torch.float32,
                                group_sizes={spec.resource_key: group.world_size for spec in specs}
                                if group is not None else None)
    if agreement is not None:
        from mstar.distributed.execution import structural_resource_identity
        agreement.agree(structural_resource_identity(specs), "qualification resource layout")
    by_key = {spec.resource_key: spec for spec in specs}
    resources = {}
    # Constructors can allocate or initialize device storage. Keep every
    # completed owner reachable even if a later constructor fails; the managed
    # worker must settle that work before releasing these roots.
    if session is not None:
        session.resource_sets.append(resources)
    for spec in specs:
        resources[spec.resource_key] = build_bound_resource(
            spec, EngineResourceInfo(device, joint_comm_group=group, kv_dtype=torch.float32,
                                     needs_remote_transfer=False,
                                     dependencies={key: by_key[key] for key in spec.depends_on()}),
            requests.get(spec.resource_key, ()))
    return specs, resources

class QualificationSession:
    """Execution roots owned by the centralized qualification worker.

    A successful case settles its stream before closing these roots. Failed
    execution or settlement leaves them retained for the managed worker's
    health check and drain; uncertain work remains in M*'s fatal hold.
    """

    def __init__(self):
        self.engines = []
        self.assemblies = []
        self.resources = {}
        self.resource_sets = []
        self.bindings = []
        self.prepared = []
        self.retained = []

    def close(self):
        for engine in reversed(self.engines):
            engine.shutdown()
        self.engines.clear()
        for assembly in self.assemblies:
            if assembly.close is not None:
                assembly.close()
        self.assemblies.clear()
        for program in reversed(self.prepared):
            program.close()
        self.prepared.clear()
        for binding in self.bindings:
            binding.prepared.clear()
        self.bindings.clear()
        self.retained.clear()
        self.resources.clear()
        self.resource_sets.clear()

def audio_inputs(model, path, *, language="en", task="transcribe"):
    """Use the model's real prompt processor on the demo's canonical PCM WAV."""
    path = Path(path).resolve()
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError("the fixture must be mono 16 kHz signed 16-bit PCM WAV")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    waveform = torch.from_numpy(samples.astype(np.float32) / np.float32(32768.0))
    inputs = model.process_prompt(None, ["audio"], ["text"],
                                  tensors={"audio_inputs": [waveform]}, language=language, task=task)
    identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "samples": waveform.numel(), "sample_rate": 16000}
    return inputs, identity
