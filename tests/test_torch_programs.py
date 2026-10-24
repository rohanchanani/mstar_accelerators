"""Focused ordinary-Torch binding and logical-weight representation checks."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.execution import Execution, InvocationFacts, LogicalInvocation
from mstar.model.contract import NodeContract, TensorInput, TensorOutput
from mstar.model.weights import LogicalWeightSource, WeightSpec
from mstar_accelerators.backends.torch import TorchProgram, prepare_torch_programs
from mstar_accelerators.backends.weights import TensorShard, prepare_weights


@dataclass(frozen=True)
class Result:
    y: torch.Tensor


def test_torch_program_slot_and_fresh_result_ownership():
    contract = NodeContract((TensorInput("x", "tensor_inputs.x", (2,)),),
                            (TensorOutput("y", (2,)),), Result, "node.")
    binding = SimpleNamespace(submodule=SimpleNamespace(logical_contract=lambda: contract),
                              device=torch.device("cpu"), dtype=torch.float32,
                              services=None, graph_walks=("decode",))
    scratch = torch.empty(2, dtype=torch.float64)

    def run(call):
        scratch.copy_(call.inputs.x[0] * 2)
        return {"y": scratch}

    prepared = prepare_torch_programs(binding, {}, {"decode": TorchProgram(run, slots=1)})
    facts = InvocationFacts("decode", 1)

    def execute(values):
        lease = prepared.acquire(facts)
        assert prepared.acquire(facts) is None
        execution = Execution(None, lease)
        invocation = LogicalInvocation("decode", [7],
            [SimpleNamespace(tensor_inputs={"x": torch.tensor(values)})], None, {})
        call = prepared.materialize(invocation, execution)
        result = prepared.primitive(call, execution)
        lease.close()
        return result

    first, second = execute([2., 3.]), execute([5., 7.])
    assert first.y.dtype == torch.float32
    assert torch.equal(first.y, torch.tensor([4., 6.]))
    assert torch.equal(second.y, torch.tensor([10., 14.]))
    assert first.y.data_ptr() != second.y.data_ptr() != scratch.data_ptr()
    prepared.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.acquire(facts)


def test_role_partitioned_weights_preserve_ffn_algebra_and_aliases():
    generator = torch.Generator().manual_seed(109)
    values = {"ff.fc1.weight": torch.randn(8, 3, generator=generator),
              "ff.fc1.bias": torch.randn(8, generator=generator),
              "ff.fc2.weight": torch.randn(3, 8, generator=generator),
              "ff.fc2.bias": torch.randn(3, generator=generator)}
    specs = [WeightSpec(name, tuple(value.shape)) for name, value in values.items()]
    specs.append(WeightSpec("ff.bias_alias", (3,), alias_of="ff.fc2.bias"))
    loads = []

    def tensors():
        loads.append(True)
        return values.items()

    source = LogicalWeightSource(specs, tensors)
    shards = {"ff.fc1.weight": TensorShard(0), "ff.fc1.bias": TensorShard(0),
              "ff.fc2.weight": TensorShard(1)}
    x = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    partials = []
    for rank in range(2):
        binding = SimpleNamespace(model=SimpleNamespace(logical_weights=lambda: source),
            device=torch.device("cpu"), dtype=torch.float32, prepared={},
            tp_group=SimpleNamespace(rank=rank, world_size=2))
        weights = prepare_weights(binding, "ff.", dtype=torch.float64, shards=shards)
        assert weights is prepare_weights(binding, "ff.", dtype=torch.float64, shards=shards)
        assert weights["ff.bias_alias"] is weights["ff.fc2.bias"]
        assert weights["ff.fc1.weight"].shape == (4, 3)
        hidden = F.gelu(F.linear(x, weights["ff.fc1.weight"], weights["ff.fc1.bias"]),
                        approximate="none")
        partials.append(F.linear(hidden, weights["ff.fc2.weight"]))
    assert len(loads) == 2
    assert all(partial.count_nonzero() for partial in partials)
    expected = F.linear(F.gelu(F.linear(x, values["ff.fc1.weight"].double(),
                values["ff.fc1.bias"].double()), approximate="none"),
                values["ff.fc2.weight"].double(), values["ff.fc2.bias"].double())
    actual = sum(partials) + values["ff.fc2.bias"].double()
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    assert not torch.allclose(actual + values["ff.fc2.bias"].double(), expected)
