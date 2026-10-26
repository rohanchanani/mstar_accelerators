"""Optimization-platform numerical policy, independent of serving admission.

These are empirical qualification budgets, not a theorem about every model.
The default combines a per-element bound with a tighter aggregate error bound.
Reference RMS supplies a scale near zero; a large tensor maximum does not.
Callers may select another policy, including strict pointwise comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float
    rms_atol: float
    rms_rtol: float


@dataclass(frozen=True)
class NumericalPolicy:
    name: str
    tolerances: dict[torch.dtype, Tolerance]
    rms_floor: bool = True
    check_rms: bool = True


# These profiles belong to this consumer. A future quantized, task-level or
# adversarial-review consumer can use another policy without changing M* or
# requiring a new field in every acceleration declaration.
DEFAULT = NumericalPolicy("component-rms-v1", {
    torch.float64: Tolerance(1e-10, 1e-10, 1e-11, 1e-11),
    torch.float32: Tolerance(1e-3, 1e-3, 1e-5, 1e-5),
    torch.float16: Tolerance(3e-3, 3e-3, 3e-4, 3e-4),
    torch.bfloat16: Tolerance(3e-2, 3e-2, 3e-3, 3e-3),
})

# Retains the old Whisper rule as named diagnostic evidence. It is not the
# default acceptance rule and does not acquire a new verdict retroactively.
LEGACY_POINTWISE = NumericalPolicy(
    "legacy-pointwise-3e-4", {torch.float32: Tolerance(3e-4, 3e-4, 0., 0.)},
    rms_floor=False, check_rms=False,
)


def _rms(values: torch.Tensor) -> torch.Tensor:
    """Scale before squaring so finite FP64 magnitudes do not overflow."""
    peak = values.abs().max()
    return peak * ((values / peak).square().mean().sqrt()) if peak else peak


def compare(name: str, actual: Any, expected: Any, *, exact: bool = False,
            dtype: torch.dtype | None = None,
            policy: NumericalPolicy = DEFAULT) -> dict:
    """Return a serializable check; a numerical mismatch never interrupts drain.

    ``dtype`` identifies the required output storage dtype. It defaults to the
    reference dtype; supply it explicitly for an FP64 diagnostic reference of
    a logically FP32 output. Integer/bool outputs and exact effects use equality.
    Tensors may be staged to CPU here, at this explicit testing read boundary.
    Backend errors from staging are allowed to propagate to the managed owner.
    """
    result = {"name": name, "passed": False, "policy": policy.name}
    tensor_pair = isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    if not tensor_pair:
        if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
            return result | {"reason": "tensor/value kind mismatch"}
        if not exact:
            raise TypeError("non-tensor observations must explicitly request exact comparison")
        same = type(actual) is type(expected) and actual == expected
        return result | {"passed": bool(same), "policy": "exact-v1",
                         "reason": "equal" if same else "exact value mismatch"}

    logical_dtype = expected.dtype if dtype is None else dtype
    result.update(actual_dtype=str(actual.dtype), reference_dtype=str(expected.dtype),
                  required_dtype=str(logical_dtype), shape=list(actual.shape))
    if actual.shape != expected.shape:
        return result | {"reason": "shape mismatch", "reference_shape": list(expected.shape)}
    if actual.dtype != logical_dtype:
        return result | {"reason": "output dtype mismatch"}
    if exact or not actual.is_floating_point():
        if expected.dtype != actual.dtype:
            return result | {"reason": "exact comparison dtype mismatch"}
        same = torch.equal(actual.detach().cpu(), expected.detach().cpu())
        return result | {"passed": same, "policy": "exact-v1",
                         "reason": "equal" if same else "exact tensor mismatch"}
    if not expected.is_floating_point():
        return result | {"reason": "floating/reference kind mismatch"}

    tolerance = policy.tolerances.get(logical_dtype)
    if tolerance is None:
        raise ValueError(f"{policy.name} has no profile for {logical_dtype}")
    result.update(atol=tolerance.atol, rtol=tolerance.rtol,
                  rms_atol=tolerance.rms_atol, rms_rtol=tolerance.rms_rtol,
                  rms_floor=policy.rms_floor, check_rms=policy.check_rms)
    a = actual.detach().to(device="cpu", dtype=torch.float64)
    b = expected.detach().to(device="cpu", dtype=torch.float64)
    if a.isnan().any() or b.isnan().any():
        return result | {"reason": "NaN is not accepted by this policy"}
    finite = b.isfinite()
    if not torch.equal(a.isfinite(), finite) or not torch.equal(a[~finite], b[~finite]):
        return result | {"reason": "nonfinite position/sign mismatch"}
    count = int(finite.sum())
    result.update(elements=a.numel(), finite_elements=count)
    if count == 0:
        return result | {"passed": True, "reason": "matching empty or infinite-only output",
                         "max_abs_error": 0., "rms_error": 0., "mismatched_elements": 0}
    a, b = a[finite], b[finite]
    delta = (a - b).abs()
    if not delta.isfinite().all():
        return result | {"reason": "finite outputs have an overflowing discrepancy"}
    reference_rms = _rms(b)
    rms_error = _rms(delta)
    scale = torch.maximum(b.abs(), reference_rms) if policy.rms_floor else b.abs()
    budget = tolerance.atol + tolerance.rtol * scale
    bad = delta > budget
    rms_budget = tolerance.rms_atol + tolerance.rms_rtol * reference_rms
    pointwise_ok = not bool(bad.any())
    rms_ok = not policy.check_rms or bool(rms_error <= rms_budget)
    passed = pointwise_ok and rms_ok
    # Profiles have positive absolute floors. Exact comparison is a separate
    # path, rather than a zero-denominator floating tolerance convention.
    max_budget_ratio = float((delta / budget).max())
    ratio_overflow = not math.isfinite(max_budget_ratio)
    result.update(passed=passed,
                  reason="within numerical policy" if passed else "numerical policy exceeded",
                  reference_rms=float(reference_rms), rms_error=float(rms_error),
                  rms_budget=float(rms_budget), max_abs_error=float(delta.max()),
                  max_budget_ratio=None if ratio_overflow else max_budget_ratio,
                  mismatched_elements=int(bad.sum()), pointwise_passed=pointwise_ok,
                  rms_passed=rms_ok)
    if ratio_overflow:
        # An enormous finite error still needs a valid JSON failure receipt.
        # This diagnostic ratio does not participate in either acceptance test.
        result["ratio_overflow"] = True
    if not pointwise_ok:
        flat_positions = finite.flatten().nonzero().flatten()
        failures = bad.nonzero().flatten()[:8]
        result["first_failures"] = [
            {"flat_index": int(flat_positions[i]), "actual": float(a[i]),
             "reference": float(b[i]), "error": float(delta[i]), "budget": float(budget[i])}
            for i in failures
        ]
    return result


def assert_close(name: str, actual: Any, expected: Any, **kwargs) -> dict:
    """Compatibility for ordinary host tests; all policy still lives above."""
    result = compare(name, actual, expected, **kwargs)
    if not result["passed"]:
        raise AssertionError(result)
    return result


def differential_comparator(acceleration, observation) -> dict:
    """Default harness callback; replacement consumers may inspect acceleration."""
    del acceleration
    from .sampling import CategoricalDistribution, CategoricalSupport, compare_categorical

    from .generation import GreedyOutput, HiddenStates, compare_greedy, compare_hidden
    if isinstance(observation.expected, GreedyOutput):
        return compare_greedy(observation.name, observation.actual, observation.expected)
    if isinstance(observation.expected, HiddenStates):
        return compare_hidden(observation.name, observation.actual, observation.expected)
    if isinstance(observation.expected, (CategoricalSupport, CategoricalDistribution)):
        return compare_categorical(observation.name, observation.actual, observation.expected)
    return compare(observation.name, observation.actual, observation.expected,
                   exact=observation.exact, dtype=observation.dtype)
