"""Central greedy-generation agreement, distinct from tensor equivalence.

vLLM's ``check_logprobs_close`` checks reciprocal membership at the first
divergence, then stops comparing the generated sequence. Its common tests
request five logprobs. This is not distribution equivalence or a sufficient
speech-quality test. See docs/QUALIFICATION.md for the pinned source.
"""
from dataclasses import dataclass
import torch

POLICY = "vllm-first-divergence-top5-v2"
TOP_K = 5

@dataclass(frozen=True)
class GreedyOutput:
    logits: torch.Tensor
    token: int


@dataclass(frozen=True)
class GreedySequence:
    outputs: tuple[GreedyOutput, ...]


def compare_greedy(name, actual, expected, *, promoted=None, suppression_mask=None):
    from .numerics import compare
    result = {"name": name, "passed": False, "obligations_passed": False, "policy": POLICY}
    if not isinstance(actual, GreedyOutput) or not isinstance(expected, GreedyOutput):
        return result | {"reason": "missing sampler observation"}
    schema = compare(name, actual.logits, expected.logits,
                     suppression_mask=suppression_mask, structural_only=True)
    if not schema['passed']:
        return result | {"reason": schema['reason'], "schema": schema}
    for label, item in (("actual", actual), ("reference", expected)):
        x = item.logits
        if (x.ndim != 1 or not 0 <= item.token < x.numel()
                or not torch.isfinite(x[item.token]) or x[item.token] != x.max()):
            return result | {"reason": label + " token is not an own-logit argmax"}
    # Keep numerical evidence; generation does not trigger a costly promoted
    # execution of internal logits. Declared floating outputs still do.
    numerical = compare(name, actual.logits, expected.logits,
                        suppression_mask=suppression_mask)
    result.update(numerical_diagnostic=numerical,
                  candidate_token=actual.token, reference_token=expected.token,
                  obligations_passed=True)
    if actual.token == expected.token:
        return result | {"passed": True, "reason": "same greedy token"}
    directions = []
    for own, other in ((actual, expected), (expected, actual)):
        x = own.logits.detach().double().cpu()
        rank = int((x > x[other.token]).sum()) + 1
        gap = float(x.max() - x[other.token])
        choices = x.topk(min(TOP_K, x.numel())).indices.tolist()
        directions.append({"other_token_rank": rank, "logprob_gap": gap,
                           "passed": other.token in choices})
    return result | {"passed": all(d['passed'] for d in directions),
                     "reason": "reciprocal top-five greedy alternatives", "directions": directions}


def sequence_result(name, token_checks):
    """Reduce observations of one autoregressive sequence, never independent rows.

    Later conditional comparisons remain visible diagnostics. Their schema and
    own-sampler obligations still gate acceptance, regardless of divergence.
    """
    first = next((i for i, check in enumerate(token_checks)
                  if check.get('candidate_token') != check.get('reference_token')), None)
    boundary = len(token_checks) if first is None else first + 1
    passed = (bool(token_checks)
              and all(check['obligations_passed'] for check in token_checks)
              and all(check['passed'] for check in token_checks[:boundary]))
    return dict(name=name, passed=passed, policy=POLICY,
                reason='common-prefix generation comparison through first divergence',
                first_divergence=first, compared_tokens=boundary,
                token_checks=token_checks)


def compare_sequence(name, actual, expected, *, suppression_mask=None):
    invalid = dict(name=name, passed=False, policy=POLICY, reason='invalid sequence observations')
    if not isinstance(actual, GreedySequence) or not isinstance(expected, GreedySequence):
        return invalid
    if len(actual.outputs) != len(expected.outputs):
        return invalid
    masks = (None,) * len(expected.outputs) if suppression_mask is None else suppression_mask
    if len(masks) != len(expected.outputs):
        return invalid
    checks = [compare_greedy(f'{name}/token{i}', a, b, suppression_mask=mask)
              for i, (a, b, mask) in enumerate(zip(actual.outputs, expected.outputs, masks))]
    return sequence_result(name, checks)
