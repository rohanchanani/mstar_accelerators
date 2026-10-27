"""One registry-consumption loop; differential testing is a replaceable consumer.

The serving declaration has no mandatory test implementation or proof field.
This optimization-platform module owns adapter resolution, case construction,
the interpretation of missing evidence, and per-acceleration verdicts.
"""
from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mstar_accelerators.conditions import Condition
from mstar_accelerators.definition import Acceleration

from .solve import SolveStatus, solve_alternatives


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class Result:
    name: str
    status: Status
    reason: str = ""
    checks: tuple[Mapping[str, Any], ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {"name": self.name, "status": self.status.value, "reason": self.reason,
                "checks": list(self.checks), "details": dict(self.details)}


@dataclass(frozen=True)
class SuiteReport:
    results: tuple[Result, ...]

    @property
    def status(self):
        if any(result.status is Status.FAIL for result in self.results):
            return Status.FAIL
        if any(result.status is Status.PASS for result in self.results):
            return Status.PASS
        return Status.SKIP

    @property
    def complete(self):
        """Every registered item produced a non-skipped verdict."""
        return bool(self.results) and all(result.status is not Status.SKIP for result in self.results)

    def to_dict(self):
        return {"status": self.status.value, "complete": self.complete,
                "counts": {status.value: sum(result.status is status for result in self.results)
                           for status in Status},
                "accelerations": [result.to_dict() for result in self.results]}


class Unavailable(Exception):
    """A diagnosed absence detected during reference/case preflight.

    Do not raise this to reclassify a submitted computation failure as a skip.
    DifferentialConsumer catches it only while constructing a CasePlan.
    """

    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code, self.reason = code, reason


@dataclass(frozen=True)
class Comparison:
    name: str
    actual: Any
    expected: Any
    exact: bool = False
    dtype: Any = None
    promoted: Any = None
    suppression_mask: Any = None
    internal: bool = False


@dataclass(frozen=True)
class CasePlan:
    facts: Mapping[str, Any]
    domains: Mapping[str, Sequence[Any]]
    execute: Callable[[Mapping[str, Any]], Iterable[Comparison]]
    reference: str
    scope: str
    details: Mapping[str, Any] = field(default_factory=dict)
    constraints: Condition | None = None


def _fact_description(value):
    """Record scalar witness facts without serializing model/binding objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_fact_description(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _fact_description(item) for key, item in value.items()}
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
        return str(value)
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


class DifferentialConsumer:
    """The seeded consumer: resolve a reference, sample when, run, compare.

    Adapters are keyed by logical reference type and supply executable reference
    construction and realizable input/state cases. A numerical policy is an
    ordinary injected comparator; it may inspect the acceleration and comparison.

    The current sampling policy solves one witness per declared disjunct, within
    the adapter's realizable domain and a bounded search. It is owned here, not
    by a driver, recipe, runtime, or acceleration author. Larger sweeps can
    replace this consumer without changing the registry-consumption contract.

    Numerical mismatches are comparison data and accumulate into FAIL. Execution
    exceptions propagate unchanged: the existing worker owns draining or fatal
    containment before any cleanup or further native submission.
    """

    def __init__(self, adapters: Mapping[type, Callable[[str, Acceleration], CasePlan]],
                 comparator: Callable[[Acceleration, Comparison], Mapping[str, Any]], *,
                 max_trials: int = 1024, max_branches: int = 32):
        self.adapters = dict(adapters)
        self.comparator = comparator
        self.max_trials = max_trials
        self.max_branches = max_branches

    def __call__(self, name: str, acceleration: Acceleration) -> Result:
        adapter = self.adapters.get(acceleration.reference)
        if adapter is None:
            return Result(name, Status.SKIP, "no executable-reference adapter is registered",
                          details={"code": "missing_reference", "scope": "not executed", "executed": False})
        try:
            plan = adapter(name, acceleration)
        except Unavailable as exc:
            return Result(name, Status.SKIP, exc.reason,
                          details={"code": exc.code, "scope": "not executed", "executed": False})
        # Case constraints describe realizable logical input/state combinations.
        # They restrict the generated test domain; they do not change when or
        # become another serving requirement on the acceleration.
        sweep = solve_alternatives(acceleration.when, plan.facts, plan.domains,
                                   constraints=plan.constraints, max_trials=self.max_trials,
                                   max_branches=self.max_branches)
        details = {**plan.details, "reference": plan.reference, "scope": plan.scope,
                   "coverage": {"policy": "declared-alternatives-v1",
                                "branches_enumerated": len(sweep.branches),
                                "enumeration_complete": sweep.complete,
                                "max_branches": self.max_branches,
                                "max_trials_per_branch": self.max_trials},
                   "cases": [], "executed": False}
        checks = []
        incomplete = not sweep.complete
        empty_case = False
        for index, solved in enumerate(sweep.branches):
            case = {"index": index, "solver": {
                "status": solved.status.value, "reason": solved.reason,
                "attempts": solved.attempts,
                "witness": {key: _fact_description(value) for key, value in solved.facts.items()},
            }, "executed": False}
            details["cases"].append(case)
            if solved.status is not SolveStatus.SAT:
                incomplete |= solved.status is SolveStatus.UNRESOLVED
                continue
            details["executed"] = case["executed"] = True
            observed = [dict(self.comparator(acceleration, comparison), case_index=index)
                        for comparison in plan.execute(solved.facts)]
            case["comparisons"] = len(observed)
            empty_case |= not observed
            checks.extend(observed)
            # Comparisons have been reduced to report values. Reclaim cycles
            # between reference programs, engines and lazy-reference closures
            # before allocating the next case, not at an arbitrary GC threshold.
            gc.collect()
        details["coverage"]["complete_in_adapter_domain"] = not (incomplete or empty_case)
        checks = tuple(checks)
        if not checks:
            return Result(name, Status.SKIP, "the differential consumer produced no comparisons",
                          details=details)
        failed = [check for check in checks if not check["passed"]]
        if failed:
            status, reason = Status.FAIL, f"{len(failed)} of {len(checks)} comparisons failed"
        elif incomplete or empty_case:
            status, reason = Status.SKIP, "declared-alternative coverage is incomplete in the adapter domain"
        else:
            status, reason = Status.PASS, "all comparisons passed for the consumer's solved alternatives"
        return Result(name, status, reason, checks, details)


def run_registry(registry, consumer: Callable[[str, Acceleration], Result]) -> SuiteReport:
    """Consume each registration with the selected optimization-time policy.

    A caller may supply any consumer, including a discovery-only skip consumer.
    The outer loop does not prescribe testing techniques or require proof.
    It deliberately does not catch execution exceptions and keep submitting work.
    """
    entries = registry.accelerations if hasattr(registry, "accelerations") else registry
    results = []
    for name, acceleration in entries.items():
        result = consumer(name, acceleration)
        if result.name != name:
            raise ValueError(f"consumer returned {result.name!r} for registration {name!r}")
        results.append(result)
    return SuiteReport(tuple(results))
