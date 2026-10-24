"""Focused ownership checks for shared compilation, independent of Whisper."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.execution import Execution, InvocationFacts, LogicalInvocation
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig, ResolvedSequenceConfig
from mstar.engine.resources.kv.plan import KVWritePlan
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.resources.sampler.plan import SamplingPlan
from mstar.engine.resources.sequence_view import (
    AttentionUse, PagedSequenceAccess, SequenceUse, sequence_access,
)
from mstar.model.contract import NodeContract, TensorInput, TensorOutput
from mstar.model.weights import LogicalWeightSource, WeightSpec
from mstar_accelerators.backends.graph import (
    GraphProgram, categorical_tokens, dense_graph_bindings, dense_views, graph_bindings, prepare_graph_programs,
)
from mstar_accelerators.compiler import PreparationScope, bind_default_call
from mstar_accelerators.conditions import Eq
from tests.whisper.provider import HostGraphProvider
from tests.whisper.evaluator import EagerGraph


@dataclass(frozen=True)
class Result:
    y: torch.Tensor


def _binding(*, specs=(), resources=None):
    loads = []

    def tensors():
        loads.append(True)
        return [("scale", torch.tensor([2., 3.]))]

    source = LogicalWeightSource((WeightSpec("node.scale", (2,), source="scale"),
                                 WeightSpec("node.alias", (2,), alias_of="node.scale")), tensors)
    contract = NodeContract((TensorInput("x", "tensor_inputs.x", (2,)),),
                            (TensorOutput("y", (2,)),), Result, "node.")
    provider = HostGraphProvider()
    binding = SimpleNamespace(
        node_name="node", model=SimpleNamespace(logical_weights=lambda: source),
        submodule=SimpleNamespace(logical_contract=lambda: contract), resource_specs=specs,
        device=torch.device("cpu"), dtype=torch.float32, prepared={},
        services=None, graph_walks=None,
    )
    return binding, resources or {}, provider, loads


def _multiply(*, graph):
    graph.output("y", graph.mul(graph.tensor("x", (2,)), graph.tensor("node.scale", (2,))))
    return graph


def _invocation(walk, values, plans=None):
    return LogicalInvocation(walk, [17], [SimpleNamespace(tensor_inputs={"x": torch.tensor(values)})],
                             None, plans or {})


def test_mode_pools_share_prepared_weights_and_preserve_result_ownership():
    binding, resources, provider, loads = _binding()
    prepared = prepare_graph_programs(binding, resources, {
        "prefill": GraphProgram(_multiply, slots=1),
        "decode": GraphProgram(_multiply, slots=1),
    }, provider=provider)
    first, second = provider.prepared
    assert len(loads) == 1
    assert first.constants["node.scale"] is second.constants["node.scale"]
    assert first.constants["node.alias"] is first.constants["node.scale"]

    prefill = prepared.acquire(InvocationFacts("prefill", 1))
    assert prepared.acquire(InvocationFacts("prefill", 1)) is None
    decode = prepared.acquire(InvocationFacts("decode", 1))
    assert decode is not None
    decode.close()

    execution = Execution(None, prefill)
    call = prepared.materialize(_invocation("prefill", [5., 7.]), execution)
    retained = prepared.primitive(call, execution)
    prefill.close()
    torch.testing.assert_close(retained.y, torch.tensor([10., 21.]))

    lease = prepared.acquire(InvocationFacts("decode", 1))
    execution = Execution(None, lease)
    call = prepared.materialize(_invocation("decode", [11., 13.]), execution)
    result = prepared.primitive(call, execution)
    lease.close()
    torch.testing.assert_close(result.y, torch.tensor([22., 39.]))
    torch.testing.assert_close(retained.y, torch.tensor([10., 21.]))
    assert result.y.untyped_storage().data_ptr() != retained.y.untyped_storage().data_ptr()
    assert len(loads) == 1
    prepared.close()
    assert all(program.closed for program in provider.prepared)


def test_generic_state_views_preserve_the_runtime_storage_across_walks():
    config = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=2,
                          max_seq_len=4, max_num_pages=3, page_size=2, prefix_cache=False)
    storage = torch.zeros(1, 3, 2, 2, 1, 2)
    resource = SimpleNamespace(config=config, kv_cache=SimpleNamespace(tensor=storage))
    resource.view = lambda request: PagedSequenceAccess(resource, request)
    resource.logical_use = lambda plans: {
        label: SequenceUse((17,), (label,), (1,), (1,), (0,), plan)
        for label, plan in plans.items()
    }
    spec = KVSpec(resource_key="history", nodes={"node"}, config=config)
    binding, resources, _, _ = _binding(specs=(spec,), resources={"history": resource})
    defaults = bind_default_call(binding, resources)

    def ports(walk, slot):
        plan = SimpleNamespace(write_locations=KVWritePlan(torch.tensor([1]), torch.tensor([slot]), 1))
        call = defaults.materialize(_invocation(walk, [1., 2.], {"history": {"main": plan}}), None)
        assert isinstance(call.plan_results.history["main"], SequenceUse)
        return graph_bindings(defaults, call)

    prefill = ports("prefill", 0)
    prefill["history.layer.0"][prefill["history.main.write.keys"][0]] = torch.tensor([9., 10.])
    decode = ports("decode", 1)
    assert prefill["history.layer.0"].data_ptr() == decode["history.layer.0"].data_ptr()
    assert decode["history.layer.0"].data_ptr() == storage.data_ptr()
    torch.testing.assert_close(decode["history.layer.0"][4], torch.tensor([9., 10.]))
    assert prefill["history.main.write.keys"].tolist() == [4]
    assert decode["history.main.write.keys"].tolist() == [5]


def test_known_missing_mode_fails_before_weights_and_local_placement_prunes_preparation():
    binding, resources, provider, loads = _binding()
    binding.services = PreparationScope("book", {"decode": Eq("batch_size", 1)}, {})
    with pytest.raises(ValueError, match="contradicts when.*graph program mapping.*decode"):
        prepare_graph_programs(binding, resources, {"prefill": GraphProgram(_multiply)}, provider=provider)
    assert loads == [] and provider.prepared == []
    binding.graph_walks = ("prefill",)
    binding.services = PreparationScope("book", {"prefill": Eq("batch_size", 1)}, {})
    prepared = prepare_graph_programs(binding, resources, {
        "prefill": GraphProgram(_multiply), "decode": GraphProgram(_multiply),
    }, provider=provider)
    assert len(provider.prepared) == 1
    prepared.close()


def test_partial_preparation_releases_already_prepared_mode():
    binding, resources, provider, _ = _binding()
    original = provider.prepare

    def reject_second(builder, constants, *, slots):
        if provider.prepared:
            raise ValueError("unsupported second physical program")
        return original(builder, constants, slots=slots)

    provider.prepare = reject_second
    with pytest.raises(ValueError, match="second physical program"):
        prepare_graph_programs(binding, resources, {
            "prefill": GraphProgram(_multiply), "decode": GraphProgram(_multiply),
        }, provider=provider)
    assert len(provider.prepared) == 1
    assert provider.prepared[0].closed


def test_default_call_preserves_mixed_walk_input_row_correspondence():
    binding, _, _, _ = _binding()
    contract = NodeContract((TensorInput("x", "tensor_inputs.x", (2,)),
                             TensorInput("context", "tensor_inputs.context", (2,), walks=("prefill",))),
                            (TensorOutput("y", (2,)),), Result, "node.")
    binding.submodule.logical_contract = lambda: contract
    defaults = bind_default_call(binding, {})
    first = SimpleNamespace(tensor_inputs={"x": torch.ones(2), "context": torch.arange(2.)})
    second = SimpleNamespace(tensor_inputs={"x": torch.zeros(2)})
    invocation = LogicalInvocation("prefill_decode", [17, 18], [first, second], None,
                                   row_walks=("prefill", "decode"))
    call = defaults.materialize(invocation, None)
    assert call.row_walks == ("prefill", "decode")
    assert len(call.inputs.x) == 2
    assert torch.equal(call.inputs.context[0], first.tensor_inputs["context"])
    assert call.inputs.context[1] is None


def test_dense_graph_ports_borrow_cache_aliases_and_owned_append_positions():
    config = ResolvedSequenceConfig(num_layers=1, num_kv_heads=1, head_dim=2,
                                    max_seq_len=4, max_num_pages=3, page_size=4,
                                    prefix_cache=False, dense=True)
    storage = torch.arange(48.).reshape(1, 3, 2, 4, 1, 2)
    resource = SimpleNamespace(config=config, kv_cache=SimpleNamespace(tensor=storage))
    resource.view = lambda request: sequence_access(resource, request)
    resource.logical_use = lambda plans: {
        label: SequenceUse.from_plan(plan) for label, plan in plans.items()}
    attention = SimpleNamespace(logical_use=lambda plans: {
        label: AttentionUse.from_plan(plan) for label, plan in plans.items()})
    specs = (KVSpec(resource_key="history", nodes={"node"}, config=config),
             AttentionSpec(resource_key="attention", nodes={"node"}, config=AttentionConfig("history")))
    binding, resources, _, _ = _binding(specs=specs, resources={"history": resource, "attention": attention})
    defaults = bind_default_call(binding, resources)
    sequence = SimpleNamespace(request_id=17, label="main", length=3, to_compute=1, start=0, page_idxs=[2])
    plan = SimpleNamespace(views=(sequence,))
    attn = SimpleNamespace(request_ids=(17,), qo_indptr=torch.tensor([0, 1]),
                           contexts=(sequence,), causal=True)
    call = defaults.materialize(_invocation("decode", [1., 2.], {
        "history": {"main": plan}, "attention": {"main": attn}}), None)
    ports = dense_graph_bindings(defaults, call, views=dense_views(binding, resources))
    keys = ports["history.main.keys.0"]
    assert keys.shape == (4, 2)
    assert keys.data_ptr() == storage[0, 2, 0].data_ptr()
    assert ports["attention.main.read.keys.0"].data_ptr() == keys.data_ptr()
    assert ports["history.main.write.positions"].tolist() == [2]
    assert ports["attention.main.read.mask"].tolist() == [[0., 0., 0., float("-inf")]]
    keys[2].fill_(123)
    assert torch.equal(storage[0, 2, 0, 2].flatten(), torch.tensor([123., 123.]))


def test_categorical_ports_use_owner_plan_without_advancing_or_truncating_it():
    spec = SamplerSpec(resource_key="draw", nodes={"node"}, vocab_size=4)
    binding, resources, _, _ = _binding(specs=(spec,), resources={"draw": object()})
    defaults = bind_default_call(binding, resources)
    seed, offset = 2**63 - 7, 2**32 + 13
    plan = SamplingPlan((17,), (SimpleNamespace(seed=seed, temperature=.75),),
                        (offset,), (True,), False, 1, (None,))
    call = defaults.materialize(_invocation("decode", [1., 2.], {"draw": plan}), None)
    assert not any(name.startswith("draw.") for name in graph_bindings(defaults, call))
    ports = graph_bindings(defaults, call, sampling=True)
    assert ports["draw.seeds"].tolist() == [seed]
    assert ports["draw.offsets"].tolist() == [offset]
    assert ports["draw.temperatures"].tolist() == [.75]
    assert ports["draw.seeds"].dtype == ports["draw.offsets"].dtype == torch.int64
    assert ports["draw.temperatures"].dtype == torch.float32

    graph = EagerGraph(ports)
    # Already-applied model suppression leaves exactly one lawful token.
    logits = torch.tensor([float("-inf"), float("-inf"), 3., float("-inf")]).reshape(1, 1, 1, 4)
    token = categorical_tokens(graph, logits, "draw")
    assert token.dtype == torch.int64 and token.item() == 2
    assert plan.offsets == (offset,) and plan.configs[0].seed == seed
    # Staged tensors are snapshots: a physical consumer cannot mutate the owner.
    ports["draw.offsets"].zero_()
    assert graph_bindings(defaults, call, sampling=True)["draw.offsets"].item() == offset
