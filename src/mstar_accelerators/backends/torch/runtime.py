"""Registry binding for an explicitly CPU, ordinary Torch deployment."""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping

import torch

from mstar_accelerators.registry import Registry
from mstar_accelerators.sources import load_accelerations, normalize_source


logger = logging.getLogger(__name__)


class TorchRuntime:
    def __init__(self, config):
        self.config = {**config, "accelerations": normalize_source(config["accelerations"])}
        node_threads = config.get("node_threads", {})
        if not isinstance(node_threads, Mapping):
            raise ValueError("node_threads must map node names to positive integers")
        self.node_threads = dict(node_threads)
        if self.node_threads:
            values = (*self.node_threads.values(), config.get("threads", 1))
            if (any(not isinstance(name, str) or not name for name in self.node_threads)
                    or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                           for value in values)):
                raise ValueError("node_threads and threads must use positive integer thread counts")
        self._thread_policy = None
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
        if self.node_threads:
            threads = self.node_threads.get(binding.node_name, self.config.get("threads", 1))
            if self._thread_policy is not None and threads != self._thread_policy[1]:
                raise ValueError(
                    f"nodes {self._thread_policy[0]!r} and {binding.node_name!r} share a worker "
                    "but request different Torch thread counts; place them on separate workers"
                )
            # M* assembles every local node before preparation or executor startup.
            # Never switch this process-wide policy during an invocation.
            if self._thread_policy is None:
                if torch.get_num_threads() != threads:
                    torch.set_num_threads(threads)
                self._thread_policy = (binding.node_name, threads)
            logger.info("TORCH_RUNTIME_NODE_THREADS pid=%d node=%s threads=%d",
                        os.getpid(), binding.node_name, torch.get_num_threads())
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
