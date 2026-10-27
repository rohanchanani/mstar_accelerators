"""Pinned compiler policy and M* semantic-boundary safeguards."""
import pytest
import torch
from mstar_accelerators.testing.numerics import compare, UPSTREAM_OPTIONS
from mstar_accelerators.testing._vendor.pytorch_same import same

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('shape', [(10,), (1024,), (2, 8, 1, 1)])
@pytest.mark.parametrize('error', [0., .002, .02, .2])
def test_pinned_decision_parity(dtype, shape, error):
    h = torch.linspace(-1, 1, torch.Size(shape).numel(), dtype=torch.float64).reshape(shape)
    eager = (h + .01).to(dtype)
    actual = (h + error).to(dtype)
    assert compare('x', actual, eager, promoted=h)['passed'] == bool(same(eager, actual, h, **UPSTREAM_OPTIONS))


def test_rmse_fallback_and_missing_reference():
    h = torch.ones(1024, dtype=torch.float64)
    e = (h + .01).float()
    a = (h + .018).float()
    assert not compare('x', a, e)['passed']
    assert compare('x', a, e, promoted=h)['passed']
    assert not compare('x', (h + .1).float(), e, promoted=h)['passed']


def test_schema_and_discrete_values_cannot_be_relaxed():
    assert not compare('x', torch.ones(1, 4), torch.ones(4))['passed']
    assert not compare('x', torch.ones(4).half(), torch.ones(4))['passed']
    assert not compare('x', torch.tensor([10001]), torch.tensor([10000]))['passed']
    assert not compare('x', True, 1, exact=True)['passed']
    assert compare('x', torch.empty(0), torch.empty(0))['passed']


def test_suppression_cannot_hide_finite_corruption():
    mask = torch.tensor([True, False, False])
    e = torch.tensor([-float('inf'), 1., 2.])
    h = e.double()
    a = e.clone(); a[1] = 100.
    assert not compare('x', a, e, promoted=h, suppression_mask=mask)['passed']
    assert compare('x', e.clone(), e, promoted=h, suppression_mask=mask)['passed']
    a = e.clone(); a[1] = -float('inf')
    assert not compare('x', a, e, promoted=h, suppression_mask=mask)['passed']
    for value in (float('nan'), float('inf'), -float('inf')):
        assert not compare('x', torch.tensor([value]), torch.tensor([value]))['passed']


def test_promoted_execution_is_lazy_and_cannot_rescue_schema_failures():
    calls = []
    def higher():
        calls.append(True)
        return torch.ones(1024, dtype=torch.float64)
    e = torch.full((1024,), 1.01)
    assert compare('x', e.clone(), e, promoted=higher)['passed']
    assert not compare('x', e.half(), e, promoted=higher)['passed']
    assert calls == []
    assert compare('x', torch.full_like(e, 1.018), e, promoted=higher)['passed']
    assert calls == [True]
