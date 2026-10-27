"""Optional synchronous ARM Q8 C ABI; code handles stay with bound programs."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import platform
from collections.abc import Mapping

from mstar.engine.execution import Support
from mstar_accelerators.conditions import Constant

logger = logging.getLogger(__name__)


def validate_artifact_config(config):
    if config is None:
        return None
    if not isinstance(config, Mapping) or set(config) != {"library", "sha256"}:
        raise ValueError("cpu_q8 must contain exactly library and sha256")
    path, digest = config["library"], config["sha256"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("cpu_q8.library must be an absolute artifact path")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("cpu_q8.sha256 must be a lowercase SHA256 digest")
    return Path(path), digest


def _arm_platform():
    return platform.system() == "Linux" and platform.machine() == "aarch64"


@dataclass(frozen=True)
class _Artifact:
    path: Path
    sha256: str


def library(binding):
    """Consume the same explicit physical artifact setting as the runtime factory."""
    setting = binding.model_config.get("runtime_extension", {}).get("cpu_q8")
    specification = validate_artifact_config(setting)
    if specification is None or binding.device.type != "cpu" or not _arm_platform():
        return None
    key = _Artifact(*specification)
    if key not in binding.prepared:
        if hashlib.sha256(key.path.read_bytes()).hexdigest() != key.sha256:
            raise ValueError("cpu_q8 library does not match its configured SHA256")
        handle = ctypes.CDLL(str(key.path), mode=os.RTLD_LOCAL | os.RTLD_NOW)
        handle.q8cpu_capability.argtypes = []
        handle.q8cpu_capability.restype = ctypes.c_int
        handle.q8cpu_readout.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        handle.q8cpu_readout.restype = ctypes.c_int
        if handle.q8cpu_capability() != 1:
            return None
        binding.prepared[key] = handle
        logger.info("CPU_Q8_LIBRARY_READY path=%s sha256=%s", key.path, key.sha256)
    return binding.prepared[key]


@dataclass(frozen=True)
class ArmQ8Available:
    """A local physical capability; unresolved authoring facts remain unknown."""

    def evaluate(self, facts):
        binding = facts.get("binding")
        if binding is None:
            return Support.UNKNOWN
        return Support.YES if library(binding) is not None else Support.NO

    def specialize(self, facts):
        result = self.evaluate(facts)
        return self if result is Support.UNKNOWN else Constant(result)
