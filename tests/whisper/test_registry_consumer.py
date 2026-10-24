"""Central consumer of the same generated and filled filesystem declarations.

Only the target device predicate is explicitly projected for CPU interpretation.
No alternate model-specific preparation implementation is registered.
"""
from dataclasses import replace
import os
from pathlib import Path

import pytest
import torch

from mstar.engine.execution import Support
from mstar.engine.resources.sampler.config import SamplingReqConfig, sampling_policy_facts
from mstar.model.whisper.config import SAMPLER
from mstar.model.whisper.results import DecoderResult
from mstar.model.whisper.submodules import WhisperEncoderSubmodule, WhisperDecoderSubmodule
from mstar.model.whisper.whisper_model import WhisperModel
from mstar_accelerators.conditions import Eq, When
from mstar_accelerators.testing.harness import DifferentialConsumer, Status, run_registry
from mstar_accelerators.testing.models.whisper import adapters
from mstar_accelerators.testing.numerics import differential_comparator
from mstar_accelerators.testing.whisper import QualificationSession, audio_inputs

from .cpu_runtime import host_entries


@pytest.fixture
def case():
    checkpoint = os.environ.get("MSTAR_WHISPER_CHECKPOINT")
    if not checkpoint:
        pytest.skip("full checkpoint is an explicit asset")
    torch.set_num_threads(4)
    model = WhisperModel(checkpoint)
    audio = Path(__file__).parents[1] / "fixtures/whisper/jfk16k.wav"
    features = audio_inputs(model, audio)[0]["audio_features"][0]
    session = QualificationSession()
    consumer = DifferentialConsumer(
        adapters(model, features, session=session, device="cpu",
                 scope="filled filesystem programs, explicit CPU graph interpretation"),
        differential_comparator,
    )
    yield consumer, session
    session.close()


def declarations(events=None):
    events = [] if events is None else events
    result = {}
    for name, acceleration in host_entries().items():
        original = acceleration.implementation.prepare

        def prepare(binding, resources, *, name=name, original=original):
            events.append(name)
            return original(binding, resources)

        result[name] = replace(acceleration, implementation=replace(acceleration.implementation, prepare=prepare))
    return result


def test_each_actual_registration_executes_from_independent_logical_state(case):
    consumer, session = case
    calls = []
    entries = declarations(calls)
    report = run_registry(entries, consumer)
    assert report.status is Status.PASS, report.to_dict()
    assert report.complete
    assert set(calls) == set(entries)
    assert all(result.details["scope"] == "filled filesystem programs, explicit CPU graph interpretation"
               for result in report.results)
    assert session.prepared == [] and session.resource_sets == [] and session.bindings == []
    decoders = [result for result in report.results
                if entries[result.name].reference is WhisperDecoderSubmodule]
    witnesses = [item["solver"]["witness"] for result in decoders
                 for item in result.details["cases"] if item["executed"]]
    assert {(item["graph_walk"], tuple(item["token_lengths"])) for item in witnesses} == {
        ("prefill", (4,)), ("decode", (1,))}
    assert {item["name"] for result in decoders for item in result.checks} >= {
        "tokens", "immediate_clone", "decode.cross_state_unchanged", "commit.position",
        "commit.sampling", "kv_cache.cleanup_pages", "cross_kv_cache.sink",
    }


def test_exact_token_error_becomes_failed_evidence_without_skipping_case(case):
    consumer, _ = case
    greedy = {"resources": {SAMPLER: (sampling_policy_facts(SamplingReqConfig(temperature=0)),)}}
    # This negative control targets exact greedy token accuracy. Additional
    # categorical registrations may appear anywhere in discovery order.
    declaration = next(value for value in declarations().values()
                       if value.reference is WhisperDecoderSubmodule
                       and value.when.evaluate(greedy) is not Support.NO)
    original_prepare = declaration.implementation.prepare

    def wrong_prepare(binding, resources):
        prepared = original_prepare(binding, resources)

        def primitive(call, execution):
            result = prepared.primitive(call, execution)
            return DecoderResult(result.tokens + 1)

        return replace(prepared, primitive=primitive)

    bad = replace(declaration, implementation=replace(declaration.implementation, prepare=wrong_prepare))
    result = consumer("faulty.decoder", bad)
    assert result.status is Status.FAIL
    failed = {item["name"] for item in result.checks if not item["passed"]}
    assert failed == {"tokens", "immediate_clone"}
    assert next(item for item in result.checks if item["name"] == "kv_cache.cleanup_pages")["passed"]


def test_unavailable_device_never_prepares_a_cpu_substitute(case):
    consumer, session = case
    calls = []
    declaration = next(value for value in declarations(calls).values()
                       if value.reference is WhisperEncoderSubmodule)
    unsupported = replace(declaration, when=When(Eq("device_type", "mstar_pi"), Eq("batch_size", 1)))
    result = consumer("pi-declaration-on-cpu", unsupported)
    assert result.status is Status.SKIP
    assert calls == []
    assert session.prepared == []


def test_missing_optional_reference_is_diagnosed_before_preparation(case, monkeypatch):
    consumer, session = case
    calls = []
    monkeypatch.setattr(WhisperEncoderSubmodule, "make_reference_program", None)
    declaration = next(value for value in declarations(calls).values()
                       if value.reference is WhisperEncoderSubmodule)
    result = consumer("no-reference.encoder", declaration)
    assert result.status is Status.SKIP
    assert result.details["code"] == "missing_reference"
    assert calls == []
    assert session.prepared == []
    assert session.resource_sets == []


def test_unexpected_reference_error_propagates_without_native_preparation(case, monkeypatch):
    consumer, session = case
    calls = []

    def broken_reference(self, binding):
        raise RuntimeError("reference implementation bug")

    monkeypatch.setattr(WhisperEncoderSubmodule, "make_reference_program", broken_reference)
    with pytest.raises(RuntimeError, match="reference implementation bug"):
        declaration = next(value for value in declarations(calls).values()
                           if value.reference is WhisperEncoderSubmodule)
        consumer("broken-reference.encoder", declaration)
    assert calls == []
    assert session.prepared == []
    assert session.resource_sets == []
