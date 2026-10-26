"""The declaration compiler must preserve applicability and ownership."""
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from mstar.engine.execution import AmbiguousExecution, InvocationFacts, Support
from mstar.model.binding import NodeBinding
from mstar_accelerators import Acceleration, Implementation, Registry, When, Eq
from mstar_accelerators.conditions import In, Range, Or
from mstar_accelerators.definition import PreparedImplementation


class Shell:
    execution_result_is_output_only = True


def binding(**changes):
    graph = SimpleNamespace(get_nodes=lambda: {"n": None})
    value = NodeBinding(
        submodule=Shell(), node_name="n", device=torch.device("cpu"), dtype=torch.float32,
        model=SimpleNamespace(get_graph_walk_graphs=lambda: {"prefill": graph, "decode": graph}),
        tp_group=SimpleNamespace(world_size=1), sp_group=SimpleNamespace(world_size=1),
        model_config={}, resource_specs=(),
    )
    return replace(value, **changes)


def declaration(walk, *, sizes=(1,), cost=None, events=None, fail=False):
    events = [] if events is None else events

    class Lease:
        def close(self):
            events.append("release")

    def prepare(binding, resources):
        events.append(("prepare", binding.node_name, tuple(resources)))
        if fail:
            raise RuntimeError("preparation failed")
        return PreparedImplementation(lambda invocation, execution: invocation,
                                      lambda call, execution: None, lambda facts: Lease(),
                                      lambda: events.append("close"))
    return Acceleration(Shell, When(Eq("device_type", "cpu"), Eq("dtype", torch.float32),
                                   Eq("graph_walk", walk), In("batch_size", sizes),
                                   Range("total_tokens", 1, 16)),
                        Implementation(prepare, cost=None if cost is None else lambda facts: cost))


def test_static_partial_evaluation_preserves_unknown_aggregate_facts():
    events = []
    registry = Registry({"p": declaration("prefill", events=events),
                         "d": declaration("decode", sizes=(1, 4), events=events)})
    assembly = registry.assemble(binding())
    assert events == []  # construction discovers support; it does not prepare programs yet
    capabilities = assembly.bind({})
    assert capabilities.batch_limits == {"prefill": 1, "decode": 4}
    assert len(capabilities.possibilities(InvocationFacts("decode", 4))) == 1
    assert not capabilities.possibilities(InvocationFacts("decode", 2))
    option = capabilities.possibilities(InvocationFacts("decode", 4))[0]
    assert option.supports(InvocationFacts("decode", 4)) is Support.UNKNOWN
    assert option.supports(InvocationFacts("decode", 4, (4, 4, 4, 4))) is Support.YES
    assert option.supports(InvocationFacts("decode", 4, (8, 8, 8, 8))) is Support.NO
    selected = capabilities.reserve(InvocationFacts("decode", 1, (1,)))
    selected.close()
    capabilities.close()
    assert events.count("release") == 1 and events.count("close") == 2


def test_exact_two_does_not_establish_singleton_progress():
    events = []
    registry = Registry({"p": declaration("prefill", sizes=(2,), events=events),
                         "d": declaration("decode", events=events)})
    with pytest.raises(ValueError, match="singleton"):
        registry.assemble(binding()).bind({})
    assert events.count("close") == 2


def test_prepare_failure_closes_prepared_programs():
    events = []
    registry = Registry({"p": declaration("prefill", events=events),
                         "d": declaration("decode", fail=True, events=events)})
    with pytest.raises(RuntimeError, match="preparation failed"):
        registry.assemble(binding()).bind({})
    assert events[-1] == "close"


def test_overlap_needs_a_selection_policy_and_does_not_leak_leases():
    events = []
    registry = Registry({"p": declaration("prefill"),
                         "d1": declaration("decode", cost=1, events=events),
                         "d2": declaration("decode", cost=1, events=events)})
    capabilities = registry.assemble(binding()).bind({})
    with pytest.raises(AmbiguousExecution):
        capabilities.reserve(InvocationFacts("decode", 1, (1,)))
    assert events.count("release") == 2
    capabilities.close()


def test_one_acceleration_covers_different_hosted_walks_and_parallel_groups():
    from mstar_accelerators.conditions import Or

    common = declaration("prefill")
    acceleration = Acceleration(
        Shell,
        When(
            Eq("device_type", "cpu"), In("batch_size", (1,)),
            Or(When(Eq("graph_walk", "prefill"), Eq("tp_size", 1)),
               When(Eq("graph_walk", "decode"), Eq("tp_size", 2))),
        ),
        common.implementation,
    )
    registry = Registry({"decoder": acceleration})
    for walk, tp_size in (("prefill", 1), ("decode", 2)):
        local = binding(graph_walks=(walk,), tp_group=SimpleNamespace(world_size=tp_size))
        capabilities = registry.assemble(local).bind({})
        try:
            assert capabilities.batch_limits == {walk: 1}
            selected = capabilities.reserve(InvocationFacts(walk, 1, (1,)))
            assert selected.option.name == "decoder"
            selected.close()
        finally:
            capabilities.close()


def test_unhosted_candidates_do_not_request_resources_or_prepare():
    events = []

    def irrelevant_resources(binding):
        raise AssertionError("unhosted candidate reached resource resolution")

    decode = declaration("decode", events=events)
    decode = replace(decode, implementation=replace(decode.implementation, resources=irrelevant_resources))
    capabilities = Registry({"p": declaration("prefill"), "d": decode}).assemble(
        binding(graph_walks=("prefill",))).bind({})
    assert [option.name for option in capabilities.options] == ["p"]
    assert events == []
    capabilities.close()


def test_local_residual_bounds_do_not_inherit_an_unhosted_unbounded_branch():
    entry = replace(declaration("prefill"), when=Or(
        When(Eq("graph_walk", "prefill"), Eq("batch_size", 1)),
        Eq("graph_walk", "decode"),
    ))
    capabilities = Registry({"combined": entry}).assemble(binding(graph_walks=("prefill",))).bind({})
    assert capabilities.batch_limits == {"prefill": 1}
    capabilities.close()


def test_resource_requests_are_node_owned_and_unified_by_value():
    from mstar.engine.resources.sequence import PagedSequenceView, DenseSequenceView

    entry = declaration("prefill")
    foreign = replace(entry, implementation=replace(entry.implementation, resources={"foreign": PagedSequenceView()}))
    with pytest.raises(ValueError, match="undeclared resource requests.*foreign"):
        Registry({"bad": foreign}).assemble(binding(graph_walks=("prefill",)))

    first = replace(entry, implementation=replace(entry.implementation, resources={"state": PagedSequenceView()}))
    second = replace(entry, implementation=replace(entry.implementation, resources={"state": (PagedSequenceView(), DenseSequenceView())}))
    spec = SimpleNamespace(resource_key="state", nodes={"n"})
    assembly = Registry({"first": first, "second": second}).assemble(
        binding(graph_walks=("prefill",), resource_specs=(spec,)))
    assert assembly.resource_requests == {"state": (PagedSequenceView(), DenseSequenceView())}


def test_helper_requirement_reports_a_contradiction_without_narrowing_when():
    original = declaration("prefill", sizes=(1, 2))

    def prepare(binding, resources):
        binding.services.require(Eq("batch_size", 1), origin="singleton physical ABI")
        return original.implementation.prepare(binding, resources)

    entry = replace(original, implementation=replace(original.implementation, prepare=prepare))
    local = binding(graph_walks=("prefill",))
    with pytest.raises(ValueError, match="contradicts when.*singleton physical ABI.*batch_size.*2"):
        Registry({"book": entry}).assemble(local).bind({})
    assert entry.when.evaluate({"device_type": "cpu", "dtype": torch.float32,
                               "graph_walk": "prefill", "batch_size": 2, "total_tokens": 2}) is Support.YES
    # Deployment admission is an independent restriction, not a helper's
    # restriction smuggled into applicability to conceal the contradiction.
    capabilities = Registry({"book": entry}).assemble(
        replace(local, model_config={"max_concurrent_requests": 1})).bind({})
    capabilities.close()


def test_single_candidate_qualification_reuses_binding_without_union_coverage():
    entry = declaration("prefill", sizes=(2,))
    capabilities = Registry({"candidate": entry}).assemble(binding(), require_coverage=False).bind({})
    assert capabilities.batch_limits == {"prefill": 2}
    capabilities.close()


def test_policy_conditions_consume_resource_owned_rows_without_a_sampler_factory():
    from mstar_accelerators.conditions import AllRows

    entry = declaration("prefill", sizes=(1, 2))
    entry = replace(entry, when=When(entry.when, AllRows("sampler", Eq("temperature", 0.0))))
    capabilities = Registry({"greedy": entry}).assemble(binding(graph_walks=("prefill",))).bind({})
    option, = capabilities.options
    assert option.supports(InvocationFacts("prefill", 1, (1,))) is Support.UNKNOWN
    assert option.supports(InvocationFacts("prefill", 1, (1,),
                           {"sampler": ({"temperature": 0.0},)})) is Support.YES
    assert option.supports(InvocationFacts("prefill", 2, (1, 1),
                           {"sampler": ({"temperature": 0.0}, {"temperature": 0.8})})) is Support.NO
    # Admission has one request policy but no proposed walk, shape or batch.
    assert option.supports(InvocationFacts(None, None, resource_facts={
        "sampler": ({"temperature": 0.0},)})) is Support.UNKNOWN
    assert option.supports(InvocationFacts(None, None, resource_facts={
        "sampler": ({"temperature": 0.8},)})) is Support.NO
    capabilities.close()


def test_mixed_row_walks_remain_an_authored_invocation_choice():
    entry = declaration("prefill", sizes=(2,))
    mixed = replace(entry, when=When(Eq("batch_size", 2),
                                     Eq("row_walks", ("prefill", "decode"))))
    capabilities = Registry({"prefill": declaration("prefill"),
                             "decode": declaration("decode"),
                             "mixed": mixed}).assemble(binding()).bind({})
    try:
        facts = InvocationFacts("prefill_decode", 2, (6, 1),
                                row_walks=("prefill", "decode"))
        assert [option.name for option in capabilities.possibilities(facts)] == ["mixed"]
        selected = capabilities.reserve(facts)
        assert selected.option.name == "mixed"
        selected.close()
        # A queue alias does not replace the actual walk of a uniform proposal.
        decode = InvocationFacts("decode", 1, (1,), row_walks=("decode",))
        assert [option.name for option in capabilities.possibilities(decode)] == ["decode"]
        assert not capabilities.possibilities(replace(facts, row_walks=("prefill", "unhosted")))
    finally:
        capabilities.close()

@pytest.mark.parametrize('condition,fixed', [
    (Range('batch_size', 1, 8), True),
    (When(Range('batch_size', 1, 8), Range('total_tokens', 1, 16)), False),
    (Eq('graph_walk', 'prefill'), False),
])
@pytest.mark.parametrize("output_only", [False, True])
def test_fixed_tensor_binding_requires_positive_coverage(monkeypatch, condition, fixed, output_only):
    monkeypatch.setattr(Shell, "execution_result_is_output_only", output_only)
    from mstar_accelerators import TensorForwardExecution
    events = []

    class Program(torch.nn.Module):
        def close(self):
            events.append('close')

    def build(local):
        events.append('build')
        return Program()

    entry = Acceleration(Shell, condition,
        Implementation(execution=TensorForwardExecution(build=build, bind_result=None)))
    assembly = Registry({'tensor': entry}).assemble(
        binding(graph_walks=('prefill',), model_config={'max_concurrent_requests': 4}))
    assert events == ['build']
    caps = assembly.bind({})
    assert (caps.fixed_option is not None) is (fixed and output_only)
    assembly.close()  # ownership has transferred to capabilities
    assert events == ['build']
    caps.close()
    assert events == ['build', 'close']


def test_early_program_is_closed_when_assembly_is_abandoned():
    from mstar_accelerators import TensorForwardExecution
    events = []

    class Program(torch.nn.Module):
        def close(self):
            events.append('close')

    entry = Acceleration(Shell, Range('batch_size', 1, 8),
        Implementation(execution=TensorForwardExecution(build=lambda _: Program(), bind_result=None)))
    assembly = Registry({'tensor': entry}).assemble(
        binding(model_config={'max_concurrent_requests': 4}))
    assembly.close()
    assembly.close()
    assert events == ['close']


def test_late_row_adapter_keeps_general_execution_consumer():
    from mstar_accelerators import TensorForwardExecution

    class Program(torch.nn.Module):
        def prepare_rows(self, rows):
            return rows

    entry = Acceleration(Shell, Range("batch_size", 1, 8),
        Implementation(execution=TensorForwardExecution(build=lambda _: Program(), bind_result=None)))
    caps = Registry({"tensor": entry}).assemble(
        binding(model_config={"max_concurrent_requests": 4})).bind({})
    assert caps.fixed_option is None
    caps.close()
