"""Fault controls for the ordinary CPU service harness, outside corpus discovery.

They wrap actual registered programs. The production selector, resource owners,
collectives, worker and conductor remain responsible for handling the outcome.
"""
from __future__ import annotations

from dataclasses import replace
import json
import logging

from mstar_accelerators.backends.torch.runtime import TorchRuntime


logger = logging.getLogger(__name__)


class ControlledTorchRuntime(TorchRuntime):
    def assemble(self, binding):
        assembly = super().assemble(binding)
        mode = self.config["distributed_control"]
        if mode not in {"refuse_once", "fail_after_execution"}:
            raise ValueError(f"unknown distributed service control {mode!r}")
        if binding.node_name != "decoder":
            return assembly
        if binding.tp_group.world_size != 2:
            raise ValueError("distributed service controls require decoder TP2")
        pending = binding.tp_group.rank == 1

        def record(event, implementation):
            logger.info("DISTRIBUTED_CONTROL %s", json.dumps({
                "event": event, "node": binding.node_name,
                "rank": binding.tp_group.rank, "implementation": implementation,
            }, sort_keys=True))

        def bind(resources):
            capabilities = assembly.bind(resources)
            options = []
            for option in capabilities.options:
                def acquire(facts, option=option):
                    nonlocal pending
                    if pending and mode == "refuse_once":
                        pending = False
                        record("workspace_refusal", option.name)
                        return None
                    return option.acquire(facts)

                def primitive(call, execution, option=option):
                    nonlocal pending
                    result = option.primitive(call, execution)
                    if pending and mode == "fail_after_execution":
                        pending = False
                        record("after_execution_failure", option.name)
                        raise RuntimeError("controlled failure after the real TP2 primitive returned")
                    return result

                options.append(replace(option, acquire=acquire, primitive=primitive))
            return replace(capabilities, options=tuple(options))

        return replace(assembly, bind=bind)
