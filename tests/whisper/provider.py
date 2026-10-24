"""Explicit host execution of the shipped graph builders for interface tests.

This provider performs synchronous Torch computation on CPU. It exercises the
same preparation/materialization/program path without emulating native fences,
device storage, or physical Pi execution.
"""
from dataclasses import dataclass

import torch

from .evaluator import EagerGraph


class ObservedGraph(EagerGraph):
    def __init__(self, bindings):
        super().__init__(bindings)
        self.argmax_inputs = []

    def argmax(self, value):
        # Observe the same arithmetic that supplies the returned token. This
        # does not replace the computation with a diagnostic program variant.
        self.argmax_inputs.append(value.detach().clone())
        return super().argmax(value)


@dataclass
class Submission:
    outputs: dict[str, torch.Tensor]
    argmax_inputs: list[torch.Tensor]


class HostGraphProvider:
    def __init__(self):
        self.prepared = []
        self.last_submission = None
        self.submissions = 0
        self.observe = None

    def prepare(self, builder, constants, *, slots=2):
        program = HostProgram(self, builder, constants, slots)
        self.prepared.append(program)
        return program


class HostProgram:
    def __init__(self, provider, builder, constants, slots):
        self.provider = provider
        self.builder = builder
        self.constants = dict(constants)
        self.slots = slots
        self.active = 0
        self.closed = False

    def try_acquire(self):
        if self.closed:
            raise RuntimeError("host graph is closed")
        if self.active == self.slots:
            return None
        self.active += 1
        return HostLease(self)

    def close(self):
        if self.active:
            raise RuntimeError("host graph still has an acquired workspace")
        self.closed = True
        self.constants.clear()
        self.builder = None


class HostLease:
    def __init__(self, program):
        self.program = program
        self.closed = False

    @torch.no_grad()
    def submit(self, inputs, outputs):
        if self.closed:
            raise RuntimeError("host workspace has already settled")
        try:
            graph = ObservedGraph({**self.program.constants, **inputs})
            self.program.builder(graph=graph)
            if set(graph.outputs) != set(outputs):
                raise ValueError("graph output names differ from the bound result")
            for name, output in outputs.items():
                output.copy_(graph.outputs[name].reshape_as(output))
            provider = self.program.provider
            provider.submissions += 1
            provider.last_submission = Submission(dict(outputs), graph.argmax_inputs)
            if provider.observe is not None:
                provider.observe(provider.last_submission)
        finally:
            # CPU execution and all copies have settled on both success and
            # failure; no physical use survives this call.
            self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self.program.active -= 1
