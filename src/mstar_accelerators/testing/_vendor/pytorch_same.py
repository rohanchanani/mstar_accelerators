"""Tensor branch of PyTorch v2.9.1 torch/_dynamo/utils.py::same.

Upstream source SHA256: daba4f1b89ec099f58f3ceafd5fe0f52f7ee0e1bacf45447307501e70c63021a
Numerical branch preserved verbatim. Only the surrounding tensor-only signature
and default-fp64 assignment are extracted. See PYTORCH_LICENSE.
"""
import math
import logging
from typing import Any, Callable
import torch
import torch._inductor.config
log = logging.getLogger(__name__)

def rmse(ref, res):
    return torch.sqrt(torch.mean(torch.square(ref - res)))

def same(ref, res, fp64_ref=None, cos_similarity=False, tol=1e-4,
         equal_nan=False, exact_dtype=True, ignore_non_fp=False,
         log_error=log.error, use_larger_multiplier_for_smaller_tensor=False,
         force_max_multiplier=False):
    if fp64_ref is None:
        fp64_ref = ref
    assert not isinstance(ref, torch._subclasses.FakeTensor)
    assert not isinstance(res, torch._subclasses.FakeTensor)

    def to_tensor(t: Any) -> torch.Tensor:
        return t if isinstance(t, torch.Tensor) else torch.tensor(t)

    ref, res, fp64_ref = (to_tensor(val) for val in (ref, res, fp64_ref))

    if ref.is_sparse:
        assert res.is_sparse
        ref = ref.to_dense()
        res = res.to_dense()
    assert isinstance(res, torch.Tensor), f"type mismatch {type(ref)} {type(res)}"
    if exact_dtype:
        if ref.dtype != res.dtype:
            log_error("dtype mismatch %s, %s", ref.dtype, res.dtype)
            return False
        if ref.dtype == torch.bool:
            if ignore_non_fp:
                return True
            # triton stores bool as int8, so add this for more accurate checking
            r = torch.allclose(
                ref.to(dtype=torch.uint8),
                res.to(dtype=torch.uint8),
                atol=tol,
                rtol=tol,
                equal_nan=equal_nan,
            )
            if not r:
                log_error("Accuracy failed: uint8 tensor did not match")
            return r

    if cos_similarity:
        ref = ref.flatten().to(torch.float32)
        res = res.flatten().to(torch.float32)
        if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=True):
            # early exit that handles zero/nan better
            # cosine_similarity(zeros(10), zeros(10), dim=0) is 0
            return True
        score = torch.nn.functional.cosine_similarity(ref, res, dim=0, eps=1e-6)
        if score < 0.99:
            log.warning("Similarity score=%s", score.detach().cpu().item())
        return bool(score >= 0.99)
    else:
        if not exact_dtype:
            ref = ref.to(res.dtype)

        # First try usual allclose
        if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=equal_nan):
            return True

        # Check error from fp64 version
        if fp64_ref.dtype == torch.float64:
            # Fix a corner case that res and fp64_ref does not contains NaN and match (with loose tolerance)
            # while the ref contains NaN. In this case, RMSE should not match any ways.
            # But res is 'BETTER' than ref so we count it pass.
            #
            # This happens for Super_SloMo when loop ordering after fusion is enabled:
            # https://gist.github.com/shunting314/11f235c70f7db0d52718d26f4a701cab
            loose_tol = 1e-2 * 4
            if (
                not fp64_ref.isnan().any()
                and not res.isnan().any()
                and ref.isnan().any()
                and torch.allclose(
                    fp64_ref.to(dtype=res.dtype),
                    res,
                    atol=loose_tol,
                    rtol=loose_tol,
                    equal_nan=equal_nan,
                )
            ):
                return True
            ref_error = rmse(fp64_ref, ref).item()
            # ref unable to produce this with stable numerics in this precision, ignore
            if math.isnan(ref_error):
                log.warning(
                    "Found nan in reference. Consider running in higher precision."
                )

            res_error = rmse(fp64_ref, res).item()

            def get_multiplier() -> float:
                # In some particular cases, we expect high difference in results.
                # At the moment one of this cases is inductor freezing bfloat16 convolution const folding.
                # In case of it the res_error is at least one order of magnitude higher.
                if force_max_multiplier:
                    return 10.0
                # In the case of using AMP (Automatic Mixed Precision), certain models have
                # failed the benchmark's correctness check. However, the end-to-end model's
                # accuracy when comparing AMP with FP32 is within a difference of less than 0.1%.
                # Thus, it's possible that the correctness check failures for these models are
                # false alarms. We use multiplier of 3 instead of 2 to avoid these false alarms.
                multiplier = (
                    3.0 if res.dtype in (torch.float16, torch.bfloat16) else 2.0
                )

                if use_larger_multiplier_for_smaller_tensor and (
                    fp64_ref.numel() <= 10
                ):
                    multiplier = 10.0
                elif use_larger_multiplier_for_smaller_tensor and (
                    fp64_ref.numel() <= 500
                ):
                    multiplier = 8.0
                elif (
                    fp64_ref.numel() < 1000
                    or (ref.ndim == 4 and ref.shape[-1] == ref.shape[-2] == 1)
                    # large tol means a benchmark has been specified as REQUIRE_HIGHER_TOLERANCE
                    or tol >= 2 * 1e-2
                ):
                    # In the presence of noise, noise might dominate our error
                    # metric for smaller tensors.
                    # Similarly, for 1x1 kernels, there seems to be high noise with amp.
                    multiplier = 3.0
                return multiplier

            multiplier = get_multiplier()

            passes_test = res_error <= (multiplier * ref_error + tol / 10.0)
            if (
                not passes_test
                and equal_nan
                and math.isnan(ref_error)
                and math.isnan(res_error)
                # Some unit test for the accuracy minifier relies on
                # returning false in this case.
                and not torch._inductor.config.cpp.inject_relu_bug_TESTING_ONLY
            ):
                passes_test = True
            if not passes_test:
                log_error(
                    "RMSE (res-fp64): %.5f, (ref-fp64): %.5f and shape=%s. res.dtype: %s, multiplier: %f, tol: %f"
                    ", use_larger_multiplier_for_smaller_tensor: %d",
                    res_error,
                    ref_error,
                    res.size(),
                    res.dtype,
                    multiplier,
                    tol,
                    use_larger_multiplier_for_smaller_tensor,
                )
            return passes_test

        if ignore_non_fp:
            return True

        log_error("Accuracy failed: allclose not within tol=%s", tol)
        return False
