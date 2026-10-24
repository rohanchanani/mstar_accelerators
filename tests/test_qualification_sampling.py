"""Focused policy controls; these do not execute or qualify a native model."""
import torch

from mstar.engine.execution import Support
from mstar.engine.resources.sampler.config import sampling_policy_facts
from mstar_accelerators.conditions import AllRows, Eq, Range, When
from mstar_accelerators.testing.sampling import (
    CategoricalSupport, compare_categorical, policy_domains, request_config,
)


def test_policy_witnesses_materialize_canonical_owner_facts_without_author_metadata():
    condition = AllRows("decoder_sampler", When(Range("temperature", 0.6, 0.8),
                                                Eq("ignore_eos", True)))
    domains = policy_domains(condition, "decoder_sampler")
    selected = next(facts for facts in domains if condition.evaluate({"resources": facts}) is Support.YES)
    config = request_config({"resources": selected}, "decoder_sampler", seed=123)
    assert config.seed == 123 and config.ignore_eos and 0.6 <= config.temperature <= 0.8
    assert selected == {"decoder_sampler": (sampling_policy_facts(config),)}
    assert not any(condition.evaluate({"resources": {"unrelated": rows}}) is Support.YES
                   for facts in domains for rows in facts.values())


def test_categorical_support_rejects_suppressed_tokens_and_output_type_errors():
    reference = CategoricalSupport(torch.tensor([[0.0, float("-inf"), -1000.0]]))
    assert compare_categorical("sample", torch.tensor([0]), reference)["passed"]
    # A finite very small logit has legal support even if floating softmax
    # underflows; explicit model suppression is the exact exclusion boundary.
    assert compare_categorical("sample", torch.tensor([2]), reference)["passed"]
    assert not compare_categorical("sample", torch.tensor([1]), reference)["passed"]
    assert not compare_categorical("sample", torch.tensor([3]), reference)["passed"]
    assert not compare_categorical("sample", torch.tensor([0.0]), reference)["passed"]
