"""Central categorical qualification policy, independent of an RNG algorithm.

These bounded observations compare actual registered draws with probabilities
from the model-owned reference. They do not prescribe Philox, FlashInfer, or
matching individual draws across different sampling implementations. The fixed
seed set makes receipts reproducible; the DKW threshold is a nominal calibration
for independent categorical draws, not a proof about a pseudorandom generator
or untested model inputs. Peaked distributions have correspondingly low power.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice, product
import math

import torch
from mstar.engine.resources.sampler.config import (
    SamplingReqConfig, derive_seed, sampling_policy_facts,
)

from mstar_accelerators.conditions import All, AllRows, Eq, In, Or, Range


DRAW_SEEDS = tuple(derive_seed(index, "qualification:categorical-cdf-dkw-v1")
                   for index in range(64))
ALPHA = 1e-6


def policy_domains(condition, resource):
    """Realizable canonical row facts for this bounded model case adapter.

    Read ordinary row predicates when possible and always evaluate the original
    ``when`` afterwards. Opaque predicates still get the small default domain;
    there is no proof obligation or inference of serving support. This consumer's
    first categorical profile realizes disabled top-k/top-p/min-p and no penalty.
    Other declared policies remain outside its reported adapter domain.
    """
    choices = {"temperature": [0.0, 1.0], "ignore_eos": [False],
               "penalize_prompt": [True]}

    def row_values(node):
        if isinstance(node, (All, Or)):
            for child in node.conditions:
                row_values(child)
        elif isinstance(node, (Eq, In, Range)) and node.fact in choices:
            if isinstance(node, Eq):
                values = (node.value,)
            elif isinstance(node, In):
                values = node.values
            else:
                values = (node.minimum, node.maximum, (node.minimum + node.maximum) / 2)
            for value in values:
                if value not in choices[node.fact]:
                    choices[node.fact].append(value)

    def visit(node):
        if isinstance(node, (All, Or)):
            for child in node.conditions:
                visit(child)
        elif isinstance(node, AllRows) and node.resource == resource:
            row_values(node.condition)

    visit(condition)
    rows = []
    for values in islice(product(*choices.values()), 64):
        config = SamplingReqConfig(**dict(zip(choices, values)))
        try:
            config.validate()
        except ValueError:
            continue
        facts = {resource: (sampling_policy_facts(config),)}
        if facts not in rows:
            rows.append(facts)
    return tuple(rows)


def request_config(witness, resource, *, seed=DRAW_SEEDS[0]):
    """Materialize the solver's canonical policy through the real owner API."""
    policy, = witness["resources"][resource]
    return SamplingReqConfig(**policy, _seed=seed)


@dataclass(frozen=True)
class CategoricalSupport:
    logits: torch.Tensor


@dataclass(frozen=True)
class CategoricalDistribution:
    logits: torch.Tensor
    temperature: float
    seeds: tuple[int, ...] = DRAW_SEEDS
    alpha: float = ALPHA


def compare_categorical(name, actual, expected):
    """Support is exact; distribution uses a documented finite-sample budget."""
    distribution = isinstance(expected, CategoricalDistribution)
    result = {"name": name, "passed": False,
              "policy": "categorical-cdf-dkw-v1" if distribution else "categorical-support-v1"}
    logits = expected.logits.detach().to(device="cpu", dtype=torch.float64)
    if logits.ndim != 2 or logits.shape[0] != 1:
        return result | {"reason": "the categorical case policy expects one reference logit row"}
    lawful = logits[0].isfinite()
    if logits.isnan().any() or logits.isposinf().any() or not lawful.any():
        return result | {"reason": "reference logits must contain a finite choice and optional -inf suppression"}
    rows = len(expected.seeds) if distribution else 1
    if not isinstance(actual, torch.Tensor) or actual.dtype != torch.int64 or tuple(actual.shape) != (rows,):
        return result | {"reason": "sampled tokens must be an int64 vector with one token per draw",
                         "required_shape": [rows]}
    tokens = actual.detach().cpu()
    if bool(((tokens < 0) | (tokens >= logits.shape[1])).any()):
        return result | {"reason": "sampled token is outside the vocabulary"}
    if not bool(lawful[tokens].all()):
        return result | {"reason": "sampled a suppressed token", "tokens": tokens.tolist()}
    if not distribution:
        return result | {"passed": True, "reason": "sampled token is in the nonsuppressed reference support"}
    if not math.isfinite(expected.temperature) or expected.temperature <= 0:
        raise ValueError("categorical probability comparison requires positive temperature")
    probabilities = torch.softmax((logits[0] - logits[0].max()) / expected.temperature, dim=0)
    empirical = torch.bincount(tokens, minlength=logits.shape[1]).double() / rows
    distance = float((empirical.cumsum(0) - probabilities.cumsum(0)).abs().max())
    threshold = math.sqrt(math.log(2 / expected.alpha) / (2 * rows))
    passed = distance <= threshold
    return result | {
        "passed": passed, "reason": "within empirical CDF budget" if passed else "empirical CDF budget exceeded",
        "draws": rows, "seeds": list(expected.seeds), "alpha": expected.alpha,
        "temperature": expected.temperature, "cdf_distance": distance, "cdf_budget": threshold,
        "reference_peak_probability": float(probabilities.max()),
        "distinct_tokens": int(tokens.unique().numel()), "tokens": tokens.tolist(),
        "scope": "bounded deterministic seed set; nominal DKW calibration; low power for peaked distributions",
    }
