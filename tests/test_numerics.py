"""Numerical qualification must reject real defects as well as accept rounding."""
import json

import pytest
import torch

from mstar_accelerators.testing.numerics import DEFAULT, LEGACY_POINTWISE, compare


def test_fp64_reference_does_not_relax_required_output_dtype():
    reference = torch.tensor([0.1, -2.0], dtype=torch.float64)
    actual = reference.float()
    assert compare("output", actual, reference, dtype=torch.float32)["passed"]
    assert not compare("output", actual.half(), reference, dtype=torch.float32)["passed"]
    assert not compare("output", actual, reference)["passed"]


def test_scale_floor_accepts_small_near_zero_error_but_keeps_pointwise_gate():
    reference = torch.ones(10000, dtype=torch.float64)
    reference[0] = 0
    actual = reference.float()
    actual[0] = 4e-4
    assert compare("output", actual, reference, dtype=torch.float32)["passed"]
    assert not compare("old", actual, reference.float(), policy=LEGACY_POINTWISE)["passed"]
    actual[0] = 1e-2
    check = compare("output", actual, reference, dtype=torch.float32)
    assert not check["passed"] and not check["pointwise_passed"]


def test_aggregate_gate_rejects_small_but_systematic_error():
    reference = torch.ones(1000)
    actual = reference + 2e-4
    check = compare("output", actual, reference)
    assert check["pointwise_passed"]
    assert not check["rms_passed"] and not check["passed"]


def test_coordinate_and_aggregate_requirements_reject_different_defects():
    reference = torch.ones(1_000_000)
    isolated = reference.clone()
    isolated[0] += 1e-2
    check = compare("isolated_corruption", isolated, reference)
    assert check["rms_passed"] and not check["pointwise_passed"]
    assert not check["passed"]

    # A dense bias may be smaller than every coordinate's allowance while
    # still exceeding the stricter complete-component accuracy requirement.
    reference = torch.ones(4096)
    check = compare("dense_bias", reference + 1e-4, reference)
    assert check["pointwise_passed"] and not check["rms_passed"]
    assert not check["passed"]


def test_nans_reject_even_with_matching_masks_and_infinities_require_signs():
    for actual, expected in (([float("nan")], [float("nan")]),
                             ([float("inf")], [-float("inf")]),
                             ([float("inf")], [1.0])):
        assert not compare("output", torch.tensor(actual), torch.tensor(expected))["passed"]
    reference = torch.tensor([-float("inf"), 1., float("inf")])
    assert compare("logits", reference.clone(), reference)["passed"]
    assert compare("mask", reference[[0, 2]], reference[[0, 2]])["passed"]


def test_infinite_masks_do_not_inflate_finite_scale():
    reference = torch.tensor([-float("inf"), 1., 0.])
    actual = reference.clone()
    actual[2] = .01
    result = compare("logits", actual, reference)
    assert not result["passed"]
    assert result["reference_rms"] == pytest.approx(2**-.5)


def test_integer_state_effects_and_padding_can_be_checked_exactly():
    assert compare("offset", 4, 4, exact=True)["passed"]
    assert not compare("offset", 5, 4, exact=True)["passed"]
    assert not compare("offset", True, 1, exact=True)["passed"]
    a = torch.tensor([2**60, 2**60+1], dtype=torch.int64)
    b = a.clone()
    b[1] += 1
    assert not compare("ids", a, b)["passed"]  # never round through float
    assert not compare("sink", torch.tensor([1e-8]), torch.zeros(1), exact=True)["passed"]


def test_shape_mismatch_cannot_broadcast_and_empty_is_well_defined():
    assert not compare("shape", torch.ones(1, 4), torch.ones(4))["passed"]
    assert compare("empty", torch.empty(0, 4), torch.empty(0, 4))["passed"]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_profiles_are_selected_by_required_logical_dtype(dtype):
    reference = torch.tensor([1., -.75, .125], dtype=torch.float64)
    result = compare("output", reference.to(dtype), reference, dtype=dtype)
    assert result["passed"]
    assert result["atol"] == DEFAULT.tolerances[dtype].atol


def test_transposition_and_nonzero_output_corruption_fail():
    reference = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    assert not compare("layout", reference.T, reference)["passed"]
    assert not compare("missing_compute", torch.zeros_like(reference), reference)["passed"]


def test_large_finite_fp64_values_cannot_pass_through_infinite_budgets():
    reference = torch.full((2,), 1e200, dtype=torch.float64)
    assert compare("large_equal", reference.clone(), reference)["passed"]
    assert not compare("large_error", reference * 2, reference)["passed"]
    reference.fill_(1e308)
    assert not compare("overflowing_difference", -reference, reference)["passed"]


def test_overflowing_diagnostic_ratio_keeps_serializable_failure_receipt():
    actual = torch.tensor([1e308], dtype=torch.float64)
    check = compare("large_error", actual, torch.zeros_like(actual))
    assert not check["passed"]
    assert not check["pointwise_passed"] and not check["rms_passed"]
    assert check["max_budget_ratio"] is None and check["ratio_overflow"]
    assert check["max_abs_error"] == 1e308
    assert json.loads(json.dumps(check, allow_nan=False)) == check
