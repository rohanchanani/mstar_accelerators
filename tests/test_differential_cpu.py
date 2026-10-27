"""Exercise the real default consumer with independently executed CPU programs.

These local declarations are test fixtures, not repository accelerations or a
fallback for the Pi. They show that another reference family uses the same
solver, implementation preparation, comparison policy and registry loop.
"""
from types import SimpleNamespace

import torch
from torch import nn

from mstar.engine.execution import Support
from mstar_accelerators import Acceleration, Eq, Implementation, Range, When
from mstar_accelerators.definition import PreparedImplementation
from mstar_accelerators.testing.harness import CasePlan, Comparison, DifferentialConsumer, Status, run_registry
from mstar_accelerators.testing.numerics import DEFAULT, differential_comparator


class ReferenceProgram(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.first = nn.Linear(5, 7, dtype=torch.float64)
        self.last = nn.Linear(7, 3, dtype=torch.float64)
        self.load_state_dict({key: value.double() for key, value in values.items()})

    def forward(self, x):
        return self.last(torch.tanh(self.first(x)))


class PhysicalProgram(nn.Module):
    def __init__(self, values, *, corrupt=False):
        super().__init__()
        for name, source in values.items():
            self.register_buffer(name.replace(".", "_"), source.clone())
        if corrupt:
            self.last_weight[0, 0] += 1.

    def forward(self, x):
        hidden = torch.tanh(x @ self.first_weight.T + self.first_bias)
        return hidden @ self.last_weight.T + self.last_bias


def setup_case(*, corrupt=False):
    generator = torch.Generator().manual_seed(3721)
    values = {
        "first.weight": torch.randn(7, 5, generator=generator) / 5,
        "first.bias": torch.randn(7, generator=generator) / 5,
        "last.weight": torch.randn(3, 7, generator=generator) / 7,
        "last.bias": torch.randn(3, generator=generator) / 7,
    }
    events = []

    class Lease:
        def close(self):
            events.append("lease_closed")

    def prepare(binding, resources):
        assert binding.device == torch.device("cpu") and resources == {}
        events.append("actual_implementation_prepare")
        physical = PhysicalProgram(binding.weights, corrupt=corrupt).eval()
        return PreparedImplementation(
            materialize=lambda logical, execution: logical,
            primitive=lambda call, execution: physical(call),
            acquire=lambda facts: Lease(),
            close=lambda: events.append("program_closed"),
        )

    acceleration = Acceleration(
        ReferenceProgram,
        When(Eq("device_type", "cpu"), Eq("dtype", torch.float32),
             Range("rows", 1, 4), Eq("features", 5)),
        Implementation(prepare),
    )

    def adapter(name, registered):
        reference = ReferenceProgram(values).eval()
        binding = SimpleNamespace(device=torch.device("cpu"), weights=values)

        def execute(witness):
            with torch.inference_mode():
                x = torch.arange(witness["rows"] * witness["features"], dtype=witness["dtype"])
                x = x.reshape(witness["rows"], witness["features"]) / 13 - .5
                actual_facts = {"device_type": x.device.type, "dtype": x.dtype,
                                "rows": x.shape[0], "features": x.shape[1]}
                assert registered.when.evaluate(actual_facts) is Support.YES
                program = registered.implementation.prepare(binding, {})
                lease = program.acquire(SimpleNamespace(rows=x.shape[0]))
                try:
                    execution = SimpleNamespace(stream=None, workspace=lease)
                    actual = program.primitive(program.materialize(x, execution), execution)
                    expected = reference(x.double())
                    events.append(("executed_rows", x.shape[0]))
                    yield Comparison("output", actual, reference.float()(x), dtype=torch.float32, promoted=expected)
                finally:
                    lease.close()
                    program.close()

        return CasePlan({"device_type": "cpu", "dtype": torch.float32},
                        {"rows": (2, 4)}, execute,
                        "independent CPU nn.Linear/tanh in FP64", "actual CPU execution")

    return acceleration, adapter, events


def test_default_consumer_solves_prepares_and_executes_distinct_torch_programs():
    acceleration, adapter, events = setup_case()
    report = run_registry({"cpu.example": acceleration},
                          DifferentialConsumer({ReferenceProgram: adapter}, differential_comparator))
    assert report.status is Status.PASS and report.complete
    check = report.results[0].checks[0]
    assert check["policy"] == DEFAULT.name
    assert check["actual_dtype"] == "torch.float32" and check["reference_dtype"] == "torch.float32"
    assert report.results[0].details["cases"][0]["solver"]["witness"]["rows"] == 2
    assert events == ["actual_implementation_prepare", ("executed_rows", 2),
                      "lease_closed", "program_closed"]


def test_numerical_bug_fails_while_missing_reference_is_a_distinct_skip():
    acceleration, adapter, events = setup_case(corrupt=True)

    class NoReference:
        pass

    missing = Acceleration(NoReference, acceleration.when, acceleration.implementation)
    report = run_registry({"cpu.wrong": acceleration, "optional.reference": missing},
                          DifferentialConsumer({ReferenceProgram: adapter}, differential_comparator))
    assert report.status is Status.FAIL and not report.complete
    assert report.to_dict()["counts"] == {"PASS": 0, "FAIL": 1, "SKIP": 1}
    assert not report.results[0].checks[0]["passed"]
    assert report.results[1].details["code"] == "missing_reference"
    assert events.count("actual_implementation_prepare") == 1
    assert events[-2:] == ["lease_closed", "program_closed"]
