"""The algebra consumer widens values, not the physical storage contract."""

import pytest
import torch

from mstar.model.whisper.reference import build_encoder
from mstar_accelerators.implementations.whisper.pi_v3d.graph import encoder_graph
from mstar_accelerators.testing.numerics import compare

from .evaluator import EagerGraph, ReadOnlyAlgebraGraph
from .test_complete_programs import config, random_weights  # noqa: F401


def test_algebra_computes_in_fp64_from_exact_fp32_values():
    # The product's low term is representable in FP64 but rounds away in FP32.
    value = torch.tensor([[1.0 + 2.0 ** -12]], dtype=torch.float32)
    original = value.clone()
    graph = ReadOnlyAlgebraGraph({"value": value})
    widened = graph.tensor("value", (1, 1))
    product = graph.mul_mat(widened, widened)
    assert product.dtype == torch.float64
    assert product.item() == (1.0 + 2.0 ** -12) ** 2
    assert product.item() != (value @ value).item()
    widened.zero_()
    assert torch.equal(value, original)

    # Logical indices are not floating-point evaluation values.
    indices = torch.tensor([2 ** 53 + 1], dtype=torch.int64)
    index_graph = ReadOnlyAlgebraGraph({"indices": indices})
    assert torch.equal(index_graph.tensor("indices", (1,), dtype="i64").reshape(-1), indices)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_algebra_rejects_wrong_logical_dtype_before_widening(dtype):
    graph = ReadOnlyAlgebraGraph({"features": torch.ones(2, dtype=dtype)})
    with pytest.raises(AssertionError):
        graph.tensor("features", (2,), dtype="f32")


@pytest.mark.parametrize("operation", ["view", "set_rows", "effect", "cast"])
def test_algebra_does_not_claim_physical_storage_semantics(operation):
    graph = ReadOnlyAlgebraGraph({"value": torch.ones(2, dtype=torch.float32)})
    value = graph.tensor("value", (2,))
    arguments = {
        "view": (value, (1,), (4,), 4),
        "set_rows": (value, torch.tensor([0]), value),
        "effect": (value,),
        "cast": (value, "f32"),
    }
    with pytest.raises(NotImplementedError):
        getattr(graph, operation)(*arguments[operation])


class _WrongGeluGraph(ReadOnlyAlgebraGraph):
    def gelu_erf(self, value):
        return value.relu()


@torch.no_grad()
def test_encoder_algebra_has_independent_oracle_and_preserves_fp32_evaluator(config):
    weights = random_weights(config)
    features = torch.randn(config.num_mel_bins, 2 * config.max_source_positions)
    bindings = {**weights, "audio_features": features}
    before = {name: value.clone() for name, value in bindings.items()}

    fp32_before = EagerGraph(bindings)
    encoder_graph(config, graph=fp32_before)
    expected = build_encoder(config, weights, dtype=torch.float64).run(features)
    graph = ReadOnlyAlgebraGraph(bindings)
    encoder_graph(config, graph=graph)
    actual = graph.outputs["encoder_states"].reshape_as(expected)
    check = compare("encoder.algebra", actual, expected)
    assert check["passed"], check
    assert check["required_dtype"] == "torch.float64"

    # Shape-correct physical programs with genuinely different computation
    # must still fail the unchanged central FP64 contract.
    wrong_weights = dict(bindings)
    wrong_weights["encoder.conv1.weight"] = torch.zeros_like(weights["encoder.conv1.weight"])
    for name, mutant in (("wrong_weight", ReadOnlyAlgebraGraph(wrong_weights)),
                         ("wrong_gelu", _WrongGeluGraph(bindings))):
        encoder_graph(config, graph=mutant)
        result = compare(name, mutant.outputs["encoder_states"].reshape_as(expected), expected)
        assert not result["passed"], result
        assert result["reason"] == "numerical policy exceeded"

    fp32_after = EagerGraph(bindings)
    encoder_graph(config, graph=fp32_after)
    assert torch.equal(fp32_before.outputs["encoder_states"], fp32_after.outputs["encoder_states"])
    for name, value in bindings.items():
        assert value.dtype == before[name].dtype
        assert torch.equal(value, before[name]), name
