"""Policy controls: numerical ambiguity is accepted, semantic corruption is not."""
from types import SimpleNamespace
import pytest
import torch
from mstar_accelerators.testing.generation import (
    GreedyOutput, HiddenStates, compare_greedy, compare_hidden,
)
from mstar_accelerators.testing.observations import SamplerObservation
from mstar.engine.resources.step import BucketKey, SlotLease


def test_greedy_near_tie_and_changed_order():
    a = torch.tensor([3., 3.01, 1., 0., -1., -2.])
    b = torch.tensor([3.01, 3., 1., 0., -1., -2.])
    check = compare_greedy('tie', GreedyOutput(a, 1), GreedyOutput(b, 0))
    assert check['passed'] and not check['tokens_equal']
    assert check['reference_regret'] > 0


def test_identical_greedy_choice_passes_even_if_topk_omits_it_at_a_tie():
    logits = torch.zeros(16)
    assert compare_greedy('same', GreedyOutput(logits, 0), GreedyOutput(logits, 0))['passed']


def test_rank_boundary_ties_do_not_depend_on_token_numbering():
    # The reference choice ties for fourth; only two of five tied candidates
    # fit into topk(5). Every permutation must have the same rank verdict.
    a = torch.tensor([9., 8., 7., 6., 6., 6., 6., 6.])
    b = torch.tensor([8., 7., 6., 5., 5., 5., 5., 9.])
    for shift in range(len(a)):
        check = compare_greedy('boundary', GreedyOutput(a.roll(shift), shift),
                               GreedyOutput(b.roll(shift), (7 + shift) % len(b)))
        assert check['passed'] and check['actual_rank_of_reference'] == 4
    a[7] = 5.  # Now seven strictly greater scores: still a failure.
    assert not compare_greedy('outside', GreedyOutput(a, 0), GreedyOutput(b, 7))['passed']


def test_both_samplers_may_choose_any_exact_argmax_tie():
    logits = torch.zeros(16)
    assert compare_greedy('ties', GreedyOutput(logits, 4), GreedyOutput(logits, 9))['passed']


@pytest.mark.parametrize('fault', ['wrong_draw', 'wrong_logits', 'suppressed', 'nan', 'shape', 'reference'])
def test_generation_rejects_broken_outputs(fault):
    b = torch.tensor([8., 3., 2., 1., 0., -1., float('-inf')])
    a = b.clone(); token = 0; ref_token = 0
    if fault == 'wrong_draw': token = 1
    if fault == 'wrong_logits': a[5] = 9.; token = 5
    if fault == 'suppressed': a[6] = 9.; token = 6
    if fault == 'nan': a[2] = float('nan')
    if fault == 'shape': a = a[:-1]
    if fault == 'reference': ref_token = 1
    assert not compare_greedy(fault, GreedyOutput(a, token), GreedyOutput(b, ref_token))['passed']


def test_conditioning_checks_direction_and_scale_per_row():
    reference = torch.tensor([[1., 2., -3.], [0., 0., 0.]], dtype=torch.bfloat16)
    assert compare_hidden('same', reference.clone(), HiddenStates(reference))['passed']
    assert not compare_hidden('scale', reference * 2, HiddenStates(reference))['passed']
    assert not compare_hidden('sign', -reference, HiddenStates(reference))['passed']
    a = reference.clone(); a[1, 0] = 1
    assert not compare_hidden('zero corruption', a, HiddenStates(reference))['passed']


def test_sampler_observation_reads_replayed_storage_and_restores_methods():
    class Resource:
        def plan(self, step, ctx): return 'planned'
        def sample(self, rids, logits): return logits.argmax(-1)
    owner = Resource(); old_plan, old_sample = owner.plan, owner.sample
    tap = SamplerObservation(owner)
    lease = SlotLease(0, BucketKey('decode', 1, 1))
    ctx = SimpleNamespace(is_preplan=False, slot_lease=lease, capture=True)
    assert owner.plan(None, ctx) == 'planned'
    logits = torch.tensor([[1., 2.]])
    assert owner.sample([10], logits).tolist() == [1]
    ctx.capture = False
    owner.plan(None, ctx)  # replay must not discard the captured observations
    captured_logits, captured_tokens = tap.snapshots[lease][0]
    captured_logits.copy_(torch.tensor([[3., 1.]])); captured_tokens.fill_(0)
    saved, tokens = tap.read(lease)[0]
    assert saved.tolist() == [[3., 1.]] and tokens.tolist() == [0]
    assert logits.tolist() == [[1., 2.]]
    tap.close()
    assert owner.plan == old_plan and owner.sample == old_sample


@pytest.mark.parametrize('corrupt_alias', [False, True])
def test_thinker_graph_edge_tracks_own_sample_after_a_reference_near_tie(monkeypatch, corrupt_alias):
    from mstar_accelerators.testing.models import qwen3_omni_nodes as module
    from mstar_accelerators.testing.numerics import differential_comparator
    # This isolates comparison construction, not execution qualification.
    model = SimpleNamespace(create_submodule=lambda name: None, get_node_resources=lambda: [])
    monkeypatch.setattr(module, 'NodeBinding', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'static_facts', lambda binding: {})
    adapter = module.NodeCaseAdapter(model, session=None, device='cpu', options={}, node='Thinker')
    def observations(candidate, batches, details):
        token = torch.tensor([0 if candidate is None else 1])
        logits = torch.tensor([3., 2.99]) if candidate is None else torch.tensor([2.99, 3.])
        alias = torch.tensor([0]) if corrupt_alias and candidate is not None else token
        return {('thinker_decode', 'test'): {'new_token': token, 'text_inputs': alias,
                'sampled_token': token, 'sampled_logits': logits}}
    monkeypatch.setattr(adapter, '_run', observations)
    plan = adapter('arbitrary-optimizer-name', None)
    checks = [differential_comparator(None, row) for row in plan.execute({})]
    failures = [row for row in checks if not row['passed']]
    if corrupt_alias:
        assert len(failures) == 1 and failures[0]['name'].endswith('/text_inputs_alias')
    else:
        assert not failures
