"""Pi binding below M*'s ordinary Torch execution and completion path."""

import json
import logging

import mstar_pi
from mstar_pi.admission import AdmissionContext

from mstar.runtime_extensions import BackendFatalError
from mstar_accelerators.registry import Registry
from mstar_accelerators.sources import load_accelerations, normalize_source

logger = logging.getLogger(__name__)


class PiRuntime:
    def __init__(self, config):
        self.config = {**config, "accelerations": normalize_source(config["accelerations"])}
        # A spawned worker loads source locally. Dynamically imported callbacks
        # and prepared native objects never cross the process boundary.
        self.registry = None
        self.owner = None

    def admission_context(self):
        context = AdmissionContext.from_environment()
        if context is None:
            raise RuntimeError("launch the Pi deployment through run_hardware.py --managed")
        return context

    def reserve_worker(self, context, name):
        return context.reserve(name)

    def initialize(self, device, owner):
        self.owner = owner
        owner.activate()
        if device.type != "mstar_pi":
            raise ValueError("PiRuntime requires the mstar_pi Torch device")
        mstar_pi.initialize(**self.config.get("native", {}))
        logger.info("PI_RUNTIME_READY %s", json.dumps(mstar_pi.health(), sort_keys=True))

    def assemble(self, binding):
        if self.registry is None:
            self.registry = Registry(load_accelerations(self.config["accelerations"]))
        return self.registry.assemble(binding)

    def is_fatal(self):
        health = mstar_pi.health()
        return bool(health["poisoned"] or health.get("retirement_failures", 0))

    def check_health(self):
        if self.is_fatal():
            health = mstar_pi.health()
            raise BackendFatalError(health.get("reason") or health.get("retirement_reason"))

    def drain(self):
        self.check_health()
        if mstar_pi.is_initialized():
            mstar_pi.synchronize()
            mstar_pi.collect()
        self.check_health()

    def close(self):
        self.check_health()
        if mstar_pi.is_initialized():
            mstar_pi.collect()
            mstar_pi.shutdown()
        logger.info("PI_RUNTIME_RELEASED %s", json.dumps(mstar_pi.health(), sort_keys=True))
        self.owner.release()

    def hold(self, reason):
        self.owner.hold(reason)
