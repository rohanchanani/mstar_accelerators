"""Central consumption is extensible without inventing successful evidence."""
import json

import pytest
from mstar.engine.execution import Support

from mstar_accelerators import Acceleration, Eq, Implementation, Registry, When
from mstar_accelerators.conditions import In, Or, Predicate
from mstar_accelerators.testing.harness import (
    CasePlan,
    Comparison,
    DifferentialConsumer,
    Result,
    Status,
    Unavailable,
    run_registry,
)


class Reference:
    pass


def acceleration():
    return Acceleration(Reference, When(Eq("device_type", "cpu"), Eq("batch_size", 1)),
                        Implementation(lambda binding, resources: None))


def exact_comparator(acceleration, comparison):
    return {"name": comparison.name, "passed": comparison.actual == comparison.expected,
            "policy": "test-exact", "exact": comparison.exact}


def plan(*comparisons, facts=None):
    return CasePlan(facts or {"device_type": "cpu"}, {}, lambda witness: iter(comparisons),
                    "independent test reference", "host test")


def test_custom_consumer_is_the_whole_outer_harness_contract():
    registry = Registry({"one": acceleration(), "two": acceleration()})
    visited = []

    def consume(name, acceleration):
        visited.append(name)
        return Result(name, Status.PASS, "a deliberately trivial consumer")

    report = run_registry(registry, consume)
    assert visited == ["one", "two"]
    assert report.status is Status.PASS and report.complete
    assert report.to_dict()["counts"] == {"PASS": 2, "FAIL": 0, "SKIP": 0}


def test_missing_reference_is_diagnosed_skip_not_failure_or_pass():
    report = run_registry({"one": acceleration()}, DifferentialConsumer({}, exact_comparator))
    assert report.status is Status.SKIP and not report.complete
    result = report.results[0]
    assert result.details["code"] == "missing_reference"
    assert result.checks == ()


def test_explicit_reference_unavailability_is_skip():
    def unavailable(name, acceleration):
        raise Unavailable("reference_unavailable", "optional reference package is absent")

    report = run_registry({"one": acceleration()}, DifferentialConsumer({Reference: unavailable}, exact_comparator))
    assert report.status is Status.SKIP
    assert report.results[0].details["code"] == "reference_unavailable"


def test_unexpected_reference_bug_is_not_reclassified_as_skip():
    def broken(name, acceleration):
        raise AssertionError("reference has the wrong parameter shape")

    with pytest.raises(AssertionError, match="parameter shape"):
        run_registry({"one": acceleration()}, DifferentialConsumer({Reference: broken}, exact_comparator))


def test_solver_checks_target_before_execution():
    calls = []
    consumer = DifferentialConsumer({Reference: lambda name, acceleration:
        CasePlan({"device_type": "other"}, {}, lambda witness: calls.append(witness),
                 "CPU reference", "native target")}, exact_comparator)
    report = run_registry({"one": acceleration()}, consumer)
    assert report.status is Status.SKIP
    assert report.results[0].details["cases"][0]["solver"]["status"] == "UNSAT"
    assert report.results[0].details["executed"] is False
    assert calls == []


def test_comparisons_accumulate_failures_and_keep_the_policy_identity():
    consumer = DifferentialConsumer({Reference: lambda name, acceleration:
        plan(Comparison("output", 1, 2), Comparison("effect", 3, 3, exact=True))}, exact_comparator)
    report = run_registry({"one": acceleration()}, consumer)
    assert report.status is Status.FAIL and report.complete
    assert len(report.results[0].checks) == 2
    assert report.results[0].checks[0]["policy"] == "test-exact"
    assert report.results[0].details["cases"][0]["solver"]["witness"]["batch_size"] == 1
    assert report.results[0].details["executed"] is True
    json.dumps(report.to_dict())


def test_empty_differential_evidence_does_not_manufacture_a_pass():
    report = run_registry({"one": acceleration()}, DifferentialConsumer(
        {Reference: lambda name, acceleration: plan()}, exact_comparator))
    assert report.status is Status.SKIP
    assert "no comparisons" in report.results[0].reason


def test_empty_all_skip_and_mixed_suites_report_coverage():
    assert run_registry({}, lambda *args: None).status is Status.SKIP
    assert not run_registry({}, lambda *args: None).complete
    entries = {"pass": acceleration(), "skip": acceleration()}
    report = run_registry(entries, lambda name, acceleration:
                          Result(name, Status.PASS if name == "pass" else Status.SKIP))
    assert report.status is Status.PASS and not report.complete
    assert report.to_dict()["counts"] == {"PASS": 1, "FAIL": 0, "SKIP": 1}


def test_execution_exception_propagates_without_submitting_another_case():
    entered = []

    def execute(witness):
        entered.append("first")
        raise RuntimeError("accepted native work needs managed drain")

    def adapter(name, acceleration):
        if name == "second":
            entered.append("second")
        return CasePlan({"device_type": "cpu"}, {}, execute, "reference", "native")

    with pytest.raises(RuntimeError, match="managed drain"):
        run_registry({"first": acceleration(), "second": acceleration()},
                     DifferentialConsumer({Reference: adapter}, exact_comparator))
    assert entered == ["first"]


def test_unavailable_after_execution_started_is_not_a_skip():
    def execute(witness):
        raise Unavailable("not-a-preflight", "work already started")

    consumer = DifferentialConsumer({Reference: lambda name, acceleration:
        CasePlan({"device_type": "cpu"}, {}, execute, "reference", "native")}, exact_comparator)
    with pytest.raises(Unavailable, match="already started"):
        run_registry({"one": acceleration()}, consumer)


def test_result_cannot_claim_a_different_registration():
    with pytest.raises(ValueError, match="different"):
        run_registry({"one": acceleration()}, lambda name, acceleration: Result("different", Status.PASS))


def test_case_constraints_relate_walk_and_shape_before_materialization():
    class LogicalShape:
        def evaluate(self, facts):
            if facts.get("walk") is None or facts.get("length") is None:
                return Support.UNKNOWN
            expected = 4 if facts["walk"] == "prefill" else 1
            return Support.YES if facts["length"] == expected else Support.NO

        def specialize(self, facts):
            return self

    declaration = Acceleration(Reference, When(Eq("walk", "decode"), In("length", (4, 1))),
                               Implementation(lambda *args: None))
    seen = []

    def execute(witness):
        seen.append(dict(witness))
        return [Comparison("length", witness["length"], 1, exact=True)]

    consumer = DifferentialConsumer({Reference: lambda name, acceleration:
        CasePlan({}, {"walk": ("prefill", "decode"), "length": (4, 1)}, execute,
                 "reference", "test", constraints=LogicalShape())}, exact_comparator)
    report = run_registry({"decode": declaration}, consumer)
    assert report.status is Status.PASS
    assert seen == [{"walk": "decode", "length": 1}]
    assert declaration.when.evaluate(seen[0]) is Support.YES


def test_consumer_derives_alternative_cases_and_returns_one_aggregate_result():
    declaration = Acceleration(Reference, When(
        Eq("device_type", "cpu"),
        Or(When(Eq("mode", "many"), Eq("length", 4)),
           When(Eq("mode", "one"), Eq("length", 1)))), Implementation(lambda *args: None))
    seen = []

    def execute(witness):
        seen.append(dict(witness))
        return [Comparison("result", 1, 1, exact=True)]

    consumer = DifferentialConsumer({Reference: lambda *_: CasePlan(
        {"device_type": "cpu"}, {"mode": ("many", "one"), "length": (4, 1)},
        execute, "reference", "test")}, exact_comparator)
    report = run_registry({"one-registration": declaration}, consumer)
    assert len(report.results) == 1 and report.complete
    result = report.results[0]
    assert result.status is Status.PASS and len(result.checks) == 2
    assert [(item["mode"], item["length"]) for item in seen] == [("many", 4), ("one", 1)]
    assert [item["case_index"] for item in result.checks] == [0, 1]
    assert result.details["coverage"]["complete_in_adapter_domain"]


@pytest.mark.parametrize("wrong", [False, True])
def test_partial_sweep_preserves_evidence_without_claiming_complete_pass(wrong):
    declaration = Acceleration(Reference, Or(Eq("length", 1), Eq("length", 2)),
                               Implementation(lambda *args: None))
    consumer = DifferentialConsumer({Reference: lambda *_: CasePlan(
        {}, {}, lambda witness: [Comparison("result", 0 if wrong else 1, 1)],
        "reference", "test")}, exact_comparator, max_branches=1)
    result = consumer("bounded", declaration)
    assert result.status is (Status.FAIL if wrong else Status.SKIP)
    assert len(result.checks) == 1
    assert not result.details["coverage"]["enumeration_complete"]


def test_unresolved_alternative_does_not_become_a_complete_pass():
    class Opaque(Predicate):
        fact = "unavailable"

        def matches(self, value):
            return bool(value)

    declaration = Acceleration(Reference, Or(Eq("length", 1), Opaque()),
                               Implementation(lambda *args: None))
    consumer = DifferentialConsumer({Reference: lambda *_: CasePlan(
        {}, {}, lambda witness: [Comparison("result", 1, 1)], "reference", "test")}, exact_comparator)
    result = consumer("partial", declaration)
    assert result.status is Status.SKIP and len(result.checks) == 1
    assert [case["solver"]["status"] for case in result.details["cases"]] == ["SAT", "UNRESOLVED"]


def test_sweep_execution_failure_prevents_later_alternatives():
    entered = []
    declaration = Acceleration(Reference, Or(Eq("length", 1), Eq("length", 2)),
                               Implementation(lambda *args: None))

    def execute(witness):
        entered.append(witness["length"])
        raise RuntimeError("native drain still required")

    consumer = DifferentialConsumer({Reference: lambda *_: CasePlan(
        {}, {}, execute, "reference", "test")}, exact_comparator)
    with pytest.raises(RuntimeError, match="drain"):
        consumer("one-registration", declaration)
    assert entered == [1]
