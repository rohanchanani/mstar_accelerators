"""Shared binding preserves logical contracts across prepared and general forms."""
from types import SimpleNamespace

import pytest
import torch
from mstar.engine.execution import LogicalInvocation
from mstar.model.contract import NodeContract, TensorInput, TensorOutput

from mstar_accelerators.compiler import allocate_results, bind_default_call


def bind(contract, resources=None):
    return bind_default_call(SimpleNamespace(
        submodule=SimpleNamespace(logical_contract=lambda: contract),
        dtype=torch.float32, device=torch.device('cpu')), resources or {})


def invocation(rows, plans=None):
    ids = list(range(len(rows)))
    return LogicalInvocation('decode', ids, rows,
        SimpleNamespace(request_ids=ids, per_request_info={i: None for i in ids}), plans or {})


@pytest.mark.parametrize('kind', ['fixed', 'dimensions', 'optional', 'dtypes', 'tokens'])
def test_prepared_and_general_results_match_model_allocation(kind):
    contract = NodeContract(
        (TensorInput('x', 'tensor_inputs.x', ('tokens',)),),
        (TensorOutput('y', ('batch',), per_request=True),), SimpleNamespace, '')
    if kind == 'dimensions':
        contract = NodeContract(contract.inputs, (TensorOutput('y', ('width',), per_request=True),),
            SimpleNamespace, '', dimensions=lambda row, walk, info: {'width': row.input_seq_len + 2})
    elif kind == 'optional':
        contract = NodeContract(contract.inputs, (TensorOutput('y', (2,), per_request=True,
            present_when=lambda row, walk, info: row.input_seq_len > 1),), SimpleNamespace, '')
    elif kind == 'dtypes':
        contract = NodeContract(contract.inputs, (TensorOutput('y', (2,), per_request=True,
            dtype=(torch.int32, torch.int64)),), SimpleNamespace, '')
    elif kind == 'tokens':
        contract = NodeContract(contract.inputs, (TensorOutput('y', ('tokens',), per_request=True),),
                               SimpleNamespace, '')
    rows = [SimpleNamespace(input_seq_len=n, tensor_inputs={'x': torch.arange(n, dtype=torch.float32)})
            for n in (1, 3)]
    defaults = bind(contract)
    inv = invocation(rows)
    first = defaults.materialize(inv, None)
    second = defaults.materialize(inv, None)
    expected = allocate_results(contract, rows, inv.engine_inputs, torch.float32, torch.device('cpu'))
    for actual, next_result, reference in zip(first.outputs.y, second.outputs.y, expected['y']):
        if reference is None:
            assert actual is next_result is None
        else:
            assert actual.shape == reference.shape and actual.dtype == reference.dtype
            assert actual.data_ptr() != next_result.data_ptr()
    defaults.zero(first, None)
    assert all(value is None or torch.count_nonzero(value) == 0 for value in first.outputs.y)
    assert all(x is row.tensor_inputs['x'] for x, row in zip(first.inputs.x, rows))


@pytest.mark.parametrize('value', [torch.arange(4.), torch.arange(4., dtype=torch.float64),
                                   torch.arange(8.)[::2]])
def test_cpu_identity_or_required_conversion_preserves_values(value):
    contract = NodeContract((TensorInput('x', 'tensor_inputs.x', (4,)),), (), SimpleNamespace, '')
    call = bind(contract).materialize(invocation([SimpleNamespace(tensor_inputs={'x': value})]), None)
    actual = call.inputs.x[0]
    torch.testing.assert_close(actual, value.float())
    assert actual.is_contiguous()
    assert (actual is value) == (value.dtype == torch.float32 and value.is_contiguous())


def test_prepared_resource_converter_consumes_current_plan():
    contract = NodeContract((), (), SimpleNamespace, '')
    resource = SimpleNamespace(logical_use=lambda plan: ('logical', plan))
    defaults = bind(contract, {'r': resource, 'plain': object()})
    for plan in (object(), object()):
        result = defaults.materialize(invocation([SimpleNamespace()], {'r': plan, 'plain': plan}), None)
        assert result.plan_results.r == ('logical', plan)
        assert result.plan_results.plain is plan


@pytest.mark.parametrize('length', [-1, 1.5, True])
def test_prepared_shapes_retain_invalid_dimension_refusal(length):
    contract = NodeContract((TensorInput('x', 'tensor_inputs.x', (2,)),), (), SimpleNamespace, '')
    with pytest.raises(ValueError, match='nonnegative integers'):
        bind(contract).materialize(invocation([SimpleNamespace(
            input_seq_len=length, tensor_inputs={'x': torch.zeros(2)})]), None)
