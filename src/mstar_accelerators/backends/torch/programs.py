"""Synchronous Torch programs using generated calls and owned result storage."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import threading

import torch

from mstar_accelerators.compiler import bind_default_call
from mstar_accelerators.conditions import In
from mstar_accelerators.definition import PreparedImplementation


@dataclass(frozen=True)
class TorchProgram:
    """Return logical output fields; the helper copies them into owned outputs."""

    run: Callable
    slots: int = 2


class _Pool:
    def __init__(self, program):
        if program.slots < 1:
            raise ValueError("a Torch program needs at least one workspace slot")
        self.program = program
        self.active = 0
        self.closed = False
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("Torch program is closed")
            if self.active == self.program.slots:
                return None
            self.active += 1
            return _Lease(self)

    def close(self):
        with self.lock:
            if self.active:
                raise RuntimeError("Torch program still has acquired workspaces")
            self.closed = True
            self.program = None


class _Lease:
    def __init__(self, pool):
        self.pool = pool
        self.closed = False

    def close(self):
        with self.pool.lock:
            if not self.closed:
                self.closed = True
                self.pool.active -= 1


def prepare_torch_programs(binding, resources, programs: Mapping[str, TorchProgram], *,
                           call_binding=None):
    """Bind explicitly CPU Torch work without owning resources or RNG commits.

    Callbacks may execute collectives. M* must finish distributed selection and
    admission before invoking either materialization or computation. Slot close
    follows the engine's completion boundary; this helper has no retry policy.
    """
    if binding.device.type != "cpu":
        raise ValueError("this synchronous Torch helper requires an explicit CPU binding")
    if binding.services is not None:
        binding.services.require(In("graph_walk", tuple(programs)), origin="Torch program mapping")
    if binding.graph_walks is not None:
        programs = {walk: program for walk, program in programs.items()
                    if walk in binding.graph_walks}
    defaults = call_binding or bind_default_call(binding, resources)
    pools = {walk: _Pool(program) for walk, program in programs.items()}

    def acquire(facts):
        return pools[facts.graph_walk].acquire()

    @torch.no_grad()
    def primitive(call, execution):
        lease = execution.workspace
        if lease.closed:
            raise RuntimeError("Torch workspace has already been released")
        values = lease.pool.program.run(call)
        expected = {spec.name for spec in defaults.contract.outputs}
        if set(values) != expected:
            raise ValueError(f"Torch output fields differ: expected {sorted(expected)}, got {sorted(values)}")
        for spec in defaults.contract.outputs:
            destination = getattr(call.outputs, spec.name)
            source = values[spec.name]
            pairs = zip(destination, source, strict=True) if spec.per_request else ((destination, source),)
            for target, value in pairs:
                if target.shape != value.shape:
                    raise ValueError(f"{spec.name}: expected {tuple(target.shape)}, got {tuple(value.shape)}")
                target.copy_(value)
        return defaults.result(call)

    def close():
        for pool in reversed(tuple(pools.values())):
            pool.close()

    return PreparedImplementation(defaults.materialize, primitive, acquire, close)
