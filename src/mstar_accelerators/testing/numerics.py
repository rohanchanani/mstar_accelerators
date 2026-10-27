"""One pinned PyTorch numerical policy, behind logical schema/state validation."""
from types import SimpleNamespace
import torch
from ._vendor.pytorch_same import same

DEFAULT = SimpleNamespace(name="pytorch-dynamo-repro-2.9-v1")
UPSTREAM_OPTIONS = dict(tol=1e-3, equal_nan=True, exact_dtype=True,
    cos_similarity=False, ignore_non_fp=False,
    use_larger_multiplier_for_smaller_tensor=False, force_max_multiplier=False)


def compare(name, actual, expected, *, exact=False, dtype=None, promoted=None,
            suppression_mask=None, structural_only=False):
    """Compare one declared tensor; never regroup rows or cast away its dtype.

    Suppression is a semantic mask supplied by the case constructor, not inferred
    from candidate errors. Ordinary outputs must be finite. Discrete values and
    explicitly exact effects bypass floating tolerance entirely.
    """
    result = {"name": name, "passed": False, "policy": DEFAULT.name}
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
            return result | {"reason": "tensor/value kind mismatch"}
        if not exact:
            raise TypeError("non-tensor observations require exact comparison")
        passed = type(actual) is type(expected) and actual == expected
        return result | {"policy": "exact-v1", "passed": bool(passed), "reason": "exact value comparison"}
    required = expected.dtype if dtype is None else dtype
    result.update(shape=list(actual.shape), actual_dtype=str(actual.dtype),
                  reference_dtype=str(expected.dtype), required_dtype=str(required))
    if actual.shape != expected.shape:
        return result | {"reason": "shape mismatch"}
    if actual.dtype != required or expected.dtype != required:
        return result | {"reason": "output/reference dtype mismatch"}
    a, b = actual.detach().cpu(), expected.detach().cpu()
    if exact or not (a.is_floating_point() or a.is_complex()):
        return result | {"policy": "exact-v1", "passed": torch.equal(a, b),
                         "reason": "exact tensor comparison"}
    mask = None
    if suppression_mask is not None:
        mask = suppression_mask.detach().cpu()
        if mask.dtype != torch.bool or mask.shape != b.shape:
            return result | {"reason": "invalid logical suppression mask"}
        if any(not torch.equal(torch.isneginf(x), mask) for x in (a, b)):
            return result | {"reason": "suppression mask mismatch"}
        a, b = a[~mask], b[~mask]
        if not a.numel():
            return result | {"reason": "no unsuppressed logits"}
    if any(not bool(x.isfinite().all()) for x in (a, b)):
        return result | {"reason": "nonfinite logical numerical payload"}
    if structural_only:
        return result | {"passed": True, "policy": "quality-structural-v1",
                         "reason": "schema, suppression and finite payload valid; fidelity checked by task continuation"}
    # The promoted execution is lazy: allclose is the upstream first branch.
    direct = torch.allclose(b, a, atol=1e-3, rtol=1e-3, equal_nan=True)
    promoted_value = None if direct else (promoted() if callable(promoted) else promoted)
    h = None if promoted_value is None else promoted_value.detach().cpu()
    if h is not None:
        if h.shape != expected.shape:
            return result | {"reason": "promoted reference shape mismatch"}
        if mask is not None:
            if not torch.equal(torch.isneginf(h), mask):
                return result | {"reason": "suppression mask mismatch"}
            h = h[~mask]
        if not bool(h.isfinite().all()):
            return result | {"reason": "nonfinite logical numerical payload"}
    errors = []
    passed = same(b, a, h, log_error=lambda fmt, *args: errors.append(fmt % args), **UPSTREAM_OPTIONS)
    direct = torch.allclose(b, a, atol=1e-3, rtol=1e-3, equal_nan=True)
    result.update(passed=bool(passed), reason="within upstream policy" if passed else "upstream numerical comparison failed",
                  branch="allclose" if direct else "fp64-rmse" if (h if h is not None else b).dtype == torch.float64 else "allclose-failed",
                  promoted_reference=h is not None, upstream_errors=errors)
    return result


def assert_close(name, actual, expected, **kwargs):
    result = compare(name, actual, expected, **kwargs)
    if not result["passed"]:
        raise AssertionError(result)
    return result


def differential_comparator(acceleration, observation):
    del acceleration
    from .speech import SpeechOutput, score_speech
    if isinstance(observation.expected, SpeechOutput):
        return score_speech(observation.name, observation.actual, observation.expected)
    from .sampling import CategoricalDistribution, CategoricalSupport, compare_categorical
    from .generation import GreedyOutput, GreedySequence, compare_greedy, compare_sequence
    if isinstance(observation.expected, GreedySequence):
        return compare_sequence(observation.name, observation.actual, observation.expected,
                                suppression_mask=observation.suppression_mask)
    if isinstance(observation.expected, GreedyOutput):
        return compare_greedy(observation.name, observation.actual, observation.expected,
                              promoted=observation.promoted, suppression_mask=observation.suppression_mask)
    if isinstance(observation.expected, (CategoricalSupport, CategoricalDistribution)):
        return compare_categorical(observation.name, observation.actual, observation.expected)
    return compare(observation.name, observation.actual, observation.expected,
                   exact=observation.exact, dtype=observation.dtype,
                   promoted=observation.promoted, suppression_mask=observation.suppression_mask)
