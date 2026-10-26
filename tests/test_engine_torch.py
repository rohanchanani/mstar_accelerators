"""External tensor-forward declarations use the ordinary registry/run contract.

Actual engine capture, staging, fallback and stream lifetime tests live in
MStar. These cases cover the producer/compiler boundary and logical outputs.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.execution import InvocationFacts, LogicalInvocation, Support
from mstar.engine.resources.sequence import PagedSequenceView
from mstar.engine.tensor_forward import CapturePolicy, CompilePolicy
from mstar.model.binding import NodeBinding
from mstar.model.contract import NodeContract, TensorOutput
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar_accelerators import Acceleration, Eq, Implementation, Range, Registry, TensorForwardExecution, When
from mstar_accelerators.compiler import pack_tensor_result


@dataclass(frozen=True)
class Result:
    value: torch.Tensor
    summary: torch.Tensor


class Shell(NodeSubmodule):
    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs(input_seq_len=1, tensor_inputs={"x": inputs["x"][0]})

    def logical_contract(self):
        return NodeContract(inputs=(), weight_prefix='', outputs=(TensorOutput('value', ('batch', 4)),
                                     TensorOutput('summary', (4,))), result_type=Result)

    def forward(self, *args, **kwargs):
        pytest.fail('logical shell executed numerical work')


class Program(torch.nn.Module):
    def __init__(self, binding, gain, events):
        super().__init__()
        self.register_buffer('gain', torch.tensor(gain))
        self.contract = binding.submodule.logical_contract()
        self.events = events

    def preprocess(self, graph_walk, engine_inputs, inputs):
        self.events.append(('pack', tuple(engine_inputs.request_ids)))
        return {'x': torch.stack([row.tensor_inputs['x'] for row in inputs])}

    def forward_batched(self, graph_walk, engine_inputs, x):
        value = x * self.gain + 1
        self.events.append(('run', graph_walk))
        return pack_tensor_result(Result(value, value.sum(0)), self.contract)

    forward = forward_batched

    def close(self):
        self.events.append('close')


def binding():
    graph = SimpleNamespace(get_nodes=lambda: {'bound': None})
    model = SimpleNamespace(get_graph_walk_graphs=lambda: {'decode': graph})
    group = SimpleNamespace(world_size=1, rank=0)
    return NodeBinding(model, Shell(), 'bound', torch.device('cpu'), torch.float32,
                       group, group, {}, (), graph_walks=('decode',))


def declaration(events, *, gain=3., walk='decode', requests=None, cost=None):
    def prepare(binding, resources):
        events.append(('prepare', tuple(resources)))
        return Program(binding, gain, events)
    return Acceleration(Shell, When(Eq('device_type', 'cpu'), Eq('graph_walk', walk),
                                  Range('batch_size', 1, 4)),
                        Implementation(prepare, resources=requests or {}, cost=cost,
                                       execution=TensorForwardExecution()))


def invocation(values):
    ids = tuple(values)
    return LogicalInvocation('decode', ids,
        tuple(NodeInputs(input_seq_len=1, tensor_inputs={'x': torch.tensor(row)})
              for row in values.values()), ModelInputsFromEngine(list(ids), {}))


def test_registered_runner_executes_independent_program_and_whole_logical_outputs():
    events = []
    local = binding()
    assembly = Registry({'owned': declaration(events)}).assemble(local)
    assert events == [] and assembly.device_plan_resources == frozenset()
    capabilities = assembly.bind({})
    option, = capabilities.options
    assert option.runner.program is not local.submodule
    assert list(local.submodule.parameters()) == [] and not hasattr(local.submodule, 'gain')
    try:
        selected = capabilities.reserve(InvocationFacts('decode', 2, (1, 1)))
        try:
            result = selected.run(invocation({20: [1., 2., 3., 4.], 10: [-1., 0., 1., 2.]}), None)
        finally:
            selected.close()
        assert isinstance(result, Result)
        torch.testing.assert_close(result.value, torch.tensor([[4., 7., 10., 13.], [-2., 1., 4., 7.]]))
        torch.testing.assert_close(result.summary, result.value.sum(0))
        assert events == [('prepare', ()), ('pack', (20, 10)), ('run', 'decode')]
    finally:
        capabilities.close()
    assert events[-1] == 'close'


def test_multiple_tensor_programs_follow_authored_when_and_cost():
    events = []
    entries = {'costly': declaration(events, gain=9., cost=lambda facts: 2.),
               'chosen': declaration(events, gain=4., cost=lambda facts: 1.)}
    capabilities = Registry(entries).assemble(binding()).bind({})
    try:
        selected = capabilities.reserve(InvocationFacts('decode', 1, (1,)))
        try:
            assert selected.option.name == 'chosen'
            result = selected.run(invocation({7: [1., 2., 3., 4.]}), None)
            assert result.value.tolist() == [[5., 9., 13., 17.]]
        finally:
            selected.close()
        assert all(option.supports(InvocationFacts('decode', 5, (1,) * 5)) is Support.NO
                   for option in capabilities.options)
    finally:
        capabilities.close()
    assert events.count('close') == 2


def test_device_plan_requirement_is_derived_from_explicit_resource_choice():
    events = []
    spec = SimpleNamespace(resource_key='kv', nodes={'bound'})
    local = replace(binding(), resource_specs=(spec,))
    for requested in (False, True):
        entry = declaration(events, requests={'kv': PagedSequenceView(device_plan=requested)})
        assembly = Registry({'owned': entry}).assemble(local)
        assert assembly.device_plan_resources == (frozenset({'kv'}) if requested else frozenset())
    assert events == []


def test_bad_runner_description_releases_prepared_program():
    events = []
    entry = declaration(events)
    entry = replace(entry, implementation=replace(entry.implementation, execution=object()))
    with pytest.raises(TypeError, match='unsupported execution description'):
        Registry({'bad': entry}).assemble(binding()).bind({})
    assert events[-1] == 'close'


def test_a_book_requests_donor_policy_without_a_model_program_install():
    from pathlib import Path
    from mstar_accelerators.sources import load_accelerations
    from mstar_accelerators.implementations.qwen3_omni.amd.preparation import prepare_program

    entries = load_accelerations(str(Path(__file__).parents[1] / 'accelerations/qwen3_omni/amd'))
    book = entries["Talker"]
    assert book.implementation.prepare is prepare_program
    execution = book.implementation.execution
    assert execution == TensorForwardExecution(
        compile=CompilePolicy(enabled=True, platform_overrides=True),
        capture=CapturePolicy(enabled=True, compile=CompilePolicy(
            enabled=True, mode='max-autotune-no-cudagraphs', dynamic=False,
            platform_overrides=True)), autocast_dtype=torch.bfloat16)
    common = dict(node_name='Talker', device_type='cuda', platform='rocm', dtype=torch.bfloat16,
                  tp_size=1, sp_size=1)
    for walk in ('talker_prefill', 'talker_last_prefill', 'talker_decode'):
        assert book.when.evaluate(common | {'graph_walk': walk, 'batch_size': 8}) is Support.YES
    assert book.when.evaluate(common | {'graph_walk': 'talker_decode', 'batch_size': 33}) is Support.NO
    assert book.when.evaluate(common | {'graph_walk': 'talker_decode', 'batch_size': 1,
                                       'platform': 'cuda'}) is Support.NO
