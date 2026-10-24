"""Explicit host-provider rehearsal through ordinary M* serving.

The acceleration files and their program callbacks are the shipped sources.
This test adapter projects only their declared target-device predicate and
executes those programs synchronously on CPU. It is not Pi emulation, a Pi
qualification result, or a fallback registered in a Pi deployment.
"""
from __future__ import annotations

from dataclasses import replace
import logging
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from mstar_accelerators.backends.graph import NativeGraphProvider
from mstar_accelerators.registry import Registry
from mstar_accelerators.sources import load_accelerations

from .provider import HostGraphProvider


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = Path(os.environ.get("MSTAR_ACCELERATION_SOURCE", ROOT / "accelerations/whisper/pi_v3d"))
_admission_path = str(ROOT / "backends/pi/torch/mstar_pi")
if _admission_path not in sys.path:
    sys.path.insert(0, _admission_path)
from admission import AdmissionContext, ENVIRONMENT  # noqa: E402


logger = logging.getLogger(__name__)


def host_entries(source=DEFAULT_SOURCES, *, provider=None):
    """Bind the declared Pi target for this explicit program interpretation.

    Geometry, dtype, walks, shapes, resource conditions and all implementation
    callbacks remain unchanged. The actual runtime device remains CPU.
    """
    provider = HostGraphProvider() if provider is None else provider
    entries = {}
    for name, acceleration in load_accelerations(source).items():
        original = acceleration.implementation.prepare

        def prepare(binding, resources, *, original=original):
            # An explicit test double at the optional graph provider boundary.
            # The real native driver never performs this substitution, and the
            # registry/compiler has no privileged graph-provider service.
            with patch.object(NativeGraphProvider, "prepare",
                              lambda self, *args, **kwargs: provider.prepare(*args, **kwargs)):
                return original(binding, resources)

        entries[name] = replace(
            acceleration, when=acceleration.when.specialize({"device_type": "mstar_pi"}),
            implementation=replace(acceleration.implementation, prepare=prepare))
    return entries


class CpuRuntime:
    def __init__(self, config):
        self.config = config
        self.registry = None
        self.owner = None
        self.provider = None

    def admission_context(self):
        context = AdmissionContext.from_environment()
        if context is None:
            raise RuntimeError("CPU host rehearsal requires its temporary admission cohort")
        return context

    def reserve_worker(self, context, name):
        return context.reserve(name)

    def initialize(self, device, owner):
        self.owner = owner
        owner.activate()
        if device.type != "cpu":
            raise ValueError("the explicit graph interpreter requires CPU placement")
        torch.set_num_threads(self.config.get("threads", 4))
        logger.info("HOST_TEST_RUNTIME_READY owner=%s", owner.owner_id)

    def assemble(self, binding):
        if self.registry is None:
            self.provider = HostGraphProvider()
            self.registry = Registry(
                host_entries(self.config.get("accelerations", DEFAULT_SOURCES), provider=self.provider),
            )
        return self.registry.assemble(binding)

    def is_fatal(self):
        return False

    def check_health(self):
        pass

    def drain(self):
        # Worker.close joins submitters; the host provider settles each call.
        pass

    def close(self):
        if self.provider is not None:
            self.provider.last_submission = None
        self.owner.release()
        logger.info("HOST_TEST_RUNTIME_RELEASED owner=%s", self.owner.owner_id)

    def hold(self, reason):
        self.owner.hold(reason)
