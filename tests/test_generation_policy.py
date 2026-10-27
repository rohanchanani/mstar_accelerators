import torch
from mstar_accelerators.testing.generation import (
    GreedyOutput, GreedySequence, compare_greedy, compare_sequence,
)


def test_generation_is_distinct_from_raw_tensor_equality():
    x = torch.arange(12, dtype=torch.float32)
    y = x + 10  # A constant logit shift changes no probabilities.
    result = compare_greedy('shift', GreedyOutput(x, 11), GreedyOutput(y, 11))
    assert result['passed']
    assert not result['numerical_diagnostic']['passed']


def test_own_argmax_is_strict_and_exact_ties_are_valid():
    x = torch.tensor([1., 1., 0.])
    assert compare_greedy('ties', GreedyOutput(x, 0), GreedyOutput(x, 1))['passed']
    assert not compare_greedy('sampler', GreedyOutput(x, 2), GreedyOutput(x, 0))['passed']


def test_reciprocal_membership_is_required():
    x = torch.tensor([2., 1.5] + [0.] * 10)
    y = torch.tensor([1.5, 2.] + [0.] * 10)
    assert compare_greedy('near', GreedyOutput(x, 0), GreedyOutput(y, 1))['passed']
    y[0] = -2.
    assert not compare_greedy('one-sided', GreedyOutput(x, 0), GreedyOutput(y, 1))['passed']
    y[0] = 0.9
    assert compare_greedy('no-added-gap-rule', GreedyOutput(x, 0), GreedyOutput(y, 1))['passed']


def test_wrong_row_corruption_and_nonfinite_are_rejected():
    x = torch.arange(12, dtype=torch.float32)
    y = x.flip(0)
    assert not compare_greedy('wrong-row', GreedyOutput(x, 11), GreedyOutput(y, 0))['passed']
    y[0] = float('nan')
    assert not compare_greedy('nan', GreedyOutput(x, 11), GreedyOutput(y, 0))['passed']


def test_suppression_and_dtype_cannot_be_bypassed_by_same_token():
    x = torch.tensor([2., 1., -torch.inf])
    mask = torch.tensor([False, False, True])
    assert compare_greedy('valid', GreedyOutput(x, 0), GreedyOutput(x, 0), suppression_mask=mask)['passed']
    assert not compare_greedy('dtype', GreedyOutput(x.double(), 0), GreedyOutput(x, 0), suppression_mask=mask)['passed']
    assert not compare_greedy('mask', GreedyOutput(x, 0), GreedyOutput(x, 0))['passed']


def test_first_divergence_bounds_sequence_agreement_not_sampler_obligations():
    x = torch.tensor([2., 1.5] + [0.] * 10)
    y = torch.tensor([1.5, 2.] + [0.] * 10)
    z = torch.arange(12, dtype=torch.float32)
    candidate = GreedySequence((GreedyOutput(x, 0), GreedyOutput(z, 11)))
    reference = GreedySequence((GreedyOutput(y, 1), GreedyOutput(z.flip(0), 0)))
    result = compare_sequence('sequence', candidate, reference)
    assert result['passed'] and result['first_divergence'] == 0
    assert not result['token_checks'][1]['passed']  # retained diagnostic
    corrupt = GreedySequence((candidate.outputs[0], GreedyOutput(z, 0)))
    assert not compare_sequence('invalid-sampler', corrupt, reference)['passed']
    malformed = GreedySequence((candidate.outputs[0], GreedyOutput(z.double(), 11)))
    assert not compare_sequence('invalid-dtype', malformed, reference)['passed']
    z[5] = float('nan')
    assert not compare_sequence('invalid-payload', candidate, reference)['passed']


def test_first_divergence_rejection_and_complete_sequence_shape():
    x = torch.arange(12, dtype=torch.float32)
    a = GreedyOutput(x, 11)
    b = GreedyOutput(x.flip(0), 0)
    result = compare_sequence('divergence', GreedySequence((a, a)), GreedySequence((a, b)))
    assert not result['passed'] and result['first_divergence'] == 1
    assert compare_sequence('equal', GreedySequence((a, a)), GreedySequence((a, a)))['passed']
    assert not compare_sequence('length', GreedySequence((a,)), GreedySequence((a, a)))['passed']
    assert not compare_sequence('empty', GreedySequence(()), GreedySequence(()))['passed']


def test_central_consumer_dispatches_sequences():
    from mstar_accelerators.testing.harness import Comparison
    from mstar_accelerators.testing.numerics import differential_comparator
    x = torch.arange(12, dtype=torch.float32)
    a = GreedySequence((GreedyOutput(x, 11),))
    assert differential_comparator(None, Comparison('sequence', a, a))['passed']
