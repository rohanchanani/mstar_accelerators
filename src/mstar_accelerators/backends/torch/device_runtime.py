"""Torch GPU lifecycle and ordinary registry assembly, independent of runner choice."""
import logging
import os

import torch

from mstar.platforms import current_platform
from mstar_accelerators.backends.torch.runtime import TorchRuntime


logger = logging.getLogger(__name__)


class DeviceTorchRuntime(TorchRuntime):
    def initialize(self, device, owner=None):
        del owner
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("DeviceTorchRuntime requires an available CUDA or ROCm GPU")
        torch.cuda.set_device(device)
        self.device = device
        logger.info("DEVICE_TORCH_RUNTIME_READY pid=%d platform=%s device=%s arch=%s",
                    os.getpid(), current_platform.name, device, current_platform.arch(device))

    def drain(self):
        # M* joins submitters first. Graph slots, resources and output tensors
        # remain owned until the ordinary Torch stream work has completed.
        if getattr(self, "device", None) is not None:
            torch.cuda.synchronize(self.device)

    def close(self):
        self.registry = None
        logger.info("DEVICE_TORCH_RUNTIME_RELEASED pid=%d", os.getpid())

    def hold(self, reason):
        raise RuntimeError(f"Torch accelerator could not establish release: {reason}")
