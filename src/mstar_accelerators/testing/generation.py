"""Central inference-output policy; no acceleration-owned test metadata.

The reciprocal top-5 rule follows vLLM's check_logprobs_close. Histories must
match: the reference scores the actual prefix rather than independently
sampling a second trajectory. Exact sampler semantics are checked separately.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

POLICY = "generation-reciprocal-top5-v1"

@dataclass(frozen=True)
class GreedyOutput:
    logits: torch.Tensor
    token: int

@dataclass(frozen=True)
class HiddenStates:
    values: torch.Tensor


def compare_greedy(name, actual, expected):
    result = {"name": name, "policy": POLICY, "passed": False}
    if not isinstance(actual, GreedyOutput):
        return result | {"reason": "missing observed sampler input/output"}
    a, b = (x.logits.detach().to(device="cpu", dtype=torch.float32).flatten()
            for x in (actual, expected))
    if actual.logits.ndim != 1 or actual.logits.shape != expected.logits.shape or a.numel() == 0:
        return result | {"reason": "logits must be matching nonempty vocabulary vectors"}
    if any(x.isnan().any() or x.isposinf().any() or not x.isfinite().any() for x in (a, b)):
        return result | {"reason": "invalid logits"}
    if not torch.equal(a.isfinite(), b.isfinite()):
        return result | {"reason": "suppression mask mismatch"}
    token = actual.token
    if not 0 <= token < a.numel() or not a[token].isfinite():
        return result | {"reason": "invalid or suppressed sampled token", "token": token}
    # A near tie between implementations does not excuse a non-greedy sampler.
    if a[token] != a.max():
        return result | {"reason": "sample is not an argmax of its own penalized logits", "token": token}
    ref_token = int(b.argmax())
    if expected.token != ref_token:
        return result | {"reason": "reference token is not greedy"}
    k = min(5, int(a.isfinite().sum()))
    top_a, top_b = a.topk(k).indices, b.topk(k).indices
    passed = token == ref_token or bool((top_b == token).any() and (top_a == ref_token).any())
    log_a, log_b = a.log_softmax(-1), b.log_softmax(-1)
    return result | {"passed": passed, "reason": "reciprocal top choices" if passed else "top choices disagree",
        "token": token, "reference_token": ref_token, "tokens_equal": token == ref_token,
        "top_k": k, "actual_top_ids": top_a.tolist(), "reference_top_ids": top_b.tolist(),
        "actual_logprob": float(log_a[token]), "reference_logprob": float(log_b[token]),
        "reference_regret": float(log_b[ref_token] - log_b[token])}


def compare_hidden(name, actual, expected):
    """Embedding direction plus magnitude, at every graph-visible token.

    vLLM uses cosine >= .999 for embeddings. Its scale invariance is unsafe for
    conditioning, so also require norms close with PyTorch's BF16 defaults
    (rtol=.016, atol=1e-5). This rule is not used for kernel unit tests.
    """
    result = {"name": name, "policy": "hidden-cosine-norm-v1", "passed": False}
    b = expected.values
    if not isinstance(actual, torch.Tensor) or actual.shape != b.shape or actual.dtype != b.dtype:
        return result | {"reason": "hidden-state shape/dtype mismatch"}
    a, b = (x.detach().to(device="cpu", dtype=torch.float64) for x in (actual, b))
    if not a.isfinite().all() or not b.isfinite().all():
        return result | {"reason": "nonfinite hidden states"}
    if a.numel() == 0:
        return result | {"passed": True, "reason": "matching empty states"}
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    cosine = F.cosine_similarity(a, b, dim=-1)
    both_zero = (na == 0) & (nb == 0)
    cosine = torch.where(both_zero, 1., cosine)
    magnitude = (na - nb).abs() <= 1e-5 + .016 * nb
    passed = bool((cosine >= .999).all() and magnitude.all())
    return result | {"passed": passed, "reason": "within embedding policy" if passed else "embedding policy exceeded",
        "minimum_cosine": float(cosine.min()), "cosine_minimum": .999,
        "norm_rtol": .016, "norm_atol": 1e-5,
        "max_norm_relative_error": float(((na - nb).abs() / nb.clamp_min(1e-12)).max())}
