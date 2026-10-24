"""Registry binding for an explicitly CPU, ordinary Torch deployment."""
from __future__ import annotations

import logging
import os

import torch

from mstar_accelerators.registry import Registry
from mstar_accelerators.sources import load_accelerations, normalize_source


logger = logging.getLogger(__name__)


class TorchRuntime:
    def __init__(self, config):
        self.config = {**config, "accelerations": normalize_source(config["accelerations"])}
        self.registry = None

    def admission_context(self):
        return None

    def reserve_worker(self, context, name):
        return None

    def initialize(self, device, owner=None):
        if device.type != "cpu":
            raise ValueError("TorchRuntime requires an explicit CPU deployment")
        torch.set_num_threads(self.config.get("threads", 1))
        logger.info("TORCH_RUNTIME_READY pid=%d device=%s threads=%s",
                    os.getpid(), device, torch.get_num_threads())

    def assemble(self, binding):
        if self.registry is None:
            self.registry = Registry(load_accelerations(self.config["accelerations"]))
        return self.registry.assemble(binding)

    def is_fatal(self):
        return False

    def check_health(self):
        pass

    def drain(self):
        # CPU callbacks return after computation. M* joins its submitter before
        # calling this boundary and owns communication-group teardown.
        pass

    def close(self):
        self.registry = None
        logger.info("TORCH_RUNTIME_RELEASED pid=%d", os.getpid())

    def hold(self, reason):
        raise RuntimeError(f"CPU Torch cannot establish release: {reason}")
