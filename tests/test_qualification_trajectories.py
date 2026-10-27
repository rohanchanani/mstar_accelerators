"""Host mechanisms with tiny synthetic math, not Whisper/native qualification.

The real model shell, declarations, registry, resource resolver, call binder,
admission/planning/commit and central consumer are exercised. The test reference
and implementations deliberately compute simple cache-dependent arithmetic.
"""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.kv.config import KVSpec
from mstar.engine.resources.sequence import DenseSequenceView, PagedSequenceView, SequenceBudget
from mstar.model.whisper.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, SAMPLER, WhisperModelConfig
from mstar.model.whisper.submodules import WhisperDecoderSubmodule
from mstar.model.whisper.whisper_model import WhisperModel
from mstar_accelerators import Acceleration, Eq, Implementation, When
from mstar_accelerators.conditions import AllRows, Or
from mstar_accelerators.compiler import bind_default_call
from mstar_accelerators.definition import PreparedImplementation
from mstar_accelerators.testing.harness import DifferentialConsumer, Status, run_registry
from mstar_accelerators.testing.models import whisper as cases
from mstar_accelerators.testing.numerics import differential_comparator
from mstar_accelerators.testing.whisper import QualificationSession


@pytest.fixture
def tiny(monkeypatch):
    # These are CPU mechanism tests, independent of the host accelerator.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    config = WhisperModelConfig(
        d_model=2, decoder_layers=1, decoder_attention_heads=1, decoder_ffn_dim=4,
        encoder_layers=1, encoder_attention_heads=1, encoder_ffn_dim=4,
        num_mel_bins=2, max_target_positions=128, max_source_positions=128,
        vocab_size=8, decoder_start_token_id=0, eos_token_id=7, no_timestamps_token_id=3,
        lang_to_id={"<|en|>": 1}, task_to_id={"transcribe": 2})
    model = WhisperModel.__new__(WhisperModel)
    model.config = config
    events = []
    sampling_case = {"flat_logits": False}

    def specs(model):
        result = model.get_node_resources()
        for spec in result:
            if isinstance(spec, KVSpec):
                cfg = spec.config
                spec.budget = SequenceBudget(3 * cfg.num_layers * 2 * cfg.max_seq_len
                                             * cfg.num_kv_heads * cfg.head_dim * 4)
        return result

    monkeypatch.setattr(cases, "_resource_specs", specs)

    class Reference:
        def modules(self): return ()

        def run(self, walk, engine, ids, states=None, *, return_logits=False):
            total = 0.0
            for key, values in ((KV_CACHE, ids.float().reshape(-1, 1).expand(-1, 2)),
                                (CROSS_KV_CACHE, states)):
                resource = engine.resources[key]
                if values is not None:
                    resource.write_kv(values.reshape(-1, 1, 2),
                                      2 * values.reshape(-1, 1, 2), layer_idx=0)
                view = engine.step.ctx.plan_results[key]["main"].views[0]
                positions = torch.arange(view.length)
                pages = torch.tensor(view.page_idxs)[positions // resource.kv_cache.page_size]
                slots = positions % resource.kv_cache.page_size
                total += float(resource.kv_cache.tensor[0, pages, 0, slots].sum())
            token = int(total) % config.vocab_size
            logits = torch.zeros(1, config.vocab_size)
            if not sampling_case["flat_logits"]:
                logits[0, token] = 1
            return logits if return_logits else logits.argmax(-1)

    def reference(self, binding, *, dtype=torch.float32):
        if binding.node_name == "audio_encoder":
            return SimpleNamespace(run=lambda _: torch.ones(128, 2, dtype=dtype), modules=lambda: ())
        return Reference()

    monkeypatch.setattr(cases.WhisperCaseAdapter, "_reference", reference)

    def declaration(walk, request, *, wrong=False, cross_value_bias=0.0, mutate_cross=False,
                    mutate_other=False, temperature=0.0, fixed_argmax=False):
        def prepare(binding, resources):
            events.append(("prepare", walk, id(resources[KV_CACHE]), id(resources[CROSS_KV_CACHE])))
            default = bind_default_call(binding, resources)
            views = {key: resources[key].view(request) for key in (KV_CACHE, CROSS_KV_CACHE)}
            other_cross = None

            def acquire(facts):
                events.append(("acquire", walk, facts.row_walks))
                return default.acquire(facts)

            def primitive(call, execution):
                nonlocal other_cross
                events.append(("run", walk, id(resources[KV_CACHE])))
                values = {KV_CACHE: call.inputs.input_ids[0].float().reshape(-1, 1).expand(-1, 2),
                          CROSS_KV_CACHE: call.inputs.encoder_states[0] if call.inputs.encoder_states else None}
                total = 0.0
                for key, value in values.items():
                    value_bias = cross_value_bias if key == CROSS_KV_CACHE else 0.0
                    use = getattr(call.plan_results, key)["main"]
                    access = views[key]
                    if isinstance(request, DenseSequenceView):
                        row = access.for_step(use)[0]
                        if value is not None:
                            row.keys[0][row.positions] = value.reshape(-1, 1, 2)
                            row.values[0][row.positions] = 2 * value.reshape(-1, 1, 2) + value_bias
                        if key == CROSS_KV_CACHE and mutate_cross and walk == "decode":
                            row.values[0][0, 0, 0] += 1e-6
                        if key == CROSS_KV_CACHE and mutate_other and walk == "decode":
                            if use.request_ids[0] == 1:
                                other_cross = row.values[0]
                            elif use.request_ids[0] == 2:
                                other_cross[0, 0, 0] += 1e-6
                        # Reading the complete fixed slot makes safe-tail
                        # initialization necessary, including NaN reuse.
                        total += float(row.keys[0].sum())
                    else:
                        plan = access.for_step(use)
                        locations = plan.write_locations
                        tensor = access.layer(0)
                        if value is not None:
                            tensor[locations.token_to_page, 0, locations.token_to_cache] = value.reshape(-1, 1, 2)
                            tensor[locations.token_to_page, 1, locations.token_to_cache] = (
                                2 * value.reshape(-1, 1, 2)
                                + value_bias)
                        attention = getattr(call.plan_results, ATTN if key == KV_CACHE else CROSS_ATTN)["main"]
                        context = access.for_attention(attention).contexts[0]
                        positions = torch.arange(context.length)
                        pages = torch.tensor(context.page_idxs)[positions // access.config.page_size]
                        slots = positions % access.config.page_size
                        total += float(tensor[pages, 0, slots].sum())
                token = (int(total) + int(wrong)) % config.vocab_size
                logits = torch.zeros(config.vocab_size)
                if not sampling_case["flat_logits"]:
                    logits[token] = 1
                token = int(logits.argmax())
                if temperature > 0:
                    sampler = getattr(call.plan_results, SAMPLER)
                    if fixed_argmax:
                        token = int(logits.argmax())
                    else:
                        generator = torch.Generator().manual_seed(sampler.configs[0].seed)
                        token = int(torch.multinomial(
                            torch.softmax(logits / sampler.configs[0].temperature, 0),
                            sampler.offsets[0] + 1, replacement=True, generator=generator)[-1])
                call.outputs.tokens.fill_(token)
                return default.result(call)

            return PreparedImplementation(default.materialize, primitive, acquire, default.close)

        walks = ("prefill", "decode") if walk == "both" else (walk,)
        branches = Or(*(When(Eq("graph_walk", selected),
                             Eq("token_lengths", (4,) if selected == "prefill" else (1,)),
                             Eq("row_walks", (selected,))) for selected in walks))
        return Acceleration(WhisperDecoderSubmodule,
                            When(Eq("device_type", "cpu"), branches, Eq("batch_size", 1),
                                 AllRows(SAMPLER, Eq("temperature", temperature))),
                            Implementation(prepare, resources={KV_CACHE: request, CROSS_KV_CACHE: request}))

    def consumer(entries, *, flat_logits=False):
        sampling_case["flat_logits"] = flat_logits
        session = QualificationSession()
        refs = cases.adapters(model, torch.zeros(2, 256), session=session, device="cpu",
                              corpus=entries, scope="synthetic host lifecycle mechanism",
                              model_config={"autocast_dtype": "float32"})
        return DifferentialConsumer(refs, differential_comparator), session

    return declaration, consumer, events


def test_distinct_registered_programs_share_state_and_reuse_after_backpressure(tiny, monkeypatch):
    from mstar.engine.engine import Engine
    original = Engine.exec
    executions = []

    def execute(engine, batch):
        outputs = original(engine, batch)
        if any(option.name != "logical-reference" for option in engine.execution_capabilities(batch.node_name).options):
            executions.append((batch.graph_walk, batch.admit_error is not None))
        return outputs

    monkeypatch.setattr(Engine, "exec", execute)
    declaration, consumer, events = tiny
    # Deliberately decode first: trajectory evidence does not depend on a
    # previous registration having run or on corpus insertion order.
    entries = {"dense_decode": declaration("decode", DenseSequenceView()),
               "paged_prefill": declaration("prefill", PagedSequenceView())}
    consume, session = consumer(entries)
    report = run_registry(entries, consume)
    assert report.status is Status.PASS, report.to_dict()
    assert report.complete and len(report.results) == 2
    result = report.results[0]
    witness = result.details["cases"][0]["solver"]["witness"]
    assert witness["row_walks"] == ["decode"]
    assert witness["resources"][SAMPLER][0]["temperature"] == 0.0
    trajectory = result.details["trajectories"]["cases"][0]
    assert trajectory["prefill"] == "paged_prefill" and trajectory["decode"] == "dense_decode"
    assert trajectory["executed"] and trajectory["lifecycle"]["applicable"]
    assert all(item["layout"] == "contiguous" and item["blocks"] == 3
               for item in trajectory["resolved_resources"].values())
    names = {check["name"] for check in result.checks}
    assert names >= {"trajectory.third_request_backpressure", "trajectory.backpressure_is_retryable",
                     "trajectory.reuses_released_blocks", "trajectory.reassigned.decode.kv_cache",
                     "trajectory.retained_output_after_reassignment",
                     "trajectory.other_request_preserved.cross_kv_cache"}
    assert sum(not blocked for _, blocked in executions) == sum(event[0] == "run" for event in events)
    assert ("prefill", True) in executions  # Backpressure also goes through Engine.exec.
    prepared = [event for event in events if event[0] == "prepare"]
    # The two preparations in the joint case receive identical existing owners.
    assert prepared[1][2:] == prepared[2][2:] and prepared[1][1] != prepared[2][1]
    assert session.engines == [] and session.assemblies == []
    assert session.prepared == [] and session.resource_sets == [] and session.bindings == []


def test_incompatible_views_are_recorded_without_reclassifying_independent_cases(tiny):
    declaration, consumer, _ = tiny
    entries = {"decode": declaration("decode", DenseSequenceView()),
               "fixed_prefill": declaration("prefill", PagedSequenceView(page_size=64))}
    consume, _ = consumer(entries)
    result = consume("decode", entries["decode"])
    assert result.status is Status.PASS
    trajectory = result.details["trajectories"]["cases"][0]
    assert not trajectory["executed"]
    assert trajectory["considered"] == [{
        "registration": "fixed_prefill", "support": "YES", "compatible": False,
        "reason": "kv_cache: incompatible sequence block sizes [64, 128]"},
        {"registration": "decode", "support": "NO"}]
    assert all(not check["name"].startswith("trajectory.") for check in result.checks)


def test_unchanged_cross_state_preserves_legal_prefill_numerical_error(tiny):
    declaration, consumer, _ = tiny
    entries = {"prefill": declaration("prefill", PagedSequenceView(), cross_value_bias=1e-6),
               "decode": declaration("decode", DenseSequenceView())}
    consume, _ = consumer(entries)
    result = consume("decode", entries["decode"])
    assert result.status is Status.PASS, result.to_dict()
    assert result.details["trajectories"]["cases"][0]["executed"]


def test_cross_state_mutation_still_fails_exact_check_within_numerical_budget(tiny):
    declaration, consumer, _ = tiny
    entries = {"prefill": declaration("prefill", PagedSequenceView()),
               "decode": declaration("decode", DenseSequenceView(), mutate_cross=True)}
    consume, _ = consumer(entries)
    result = consume("decode", entries["decode"])
    assert result.status is Status.FAIL
    assert any(check["name"].endswith("decode.cross_state_unchanged") and not check["passed"]
               for check in result.checks)
    assert all(check["passed"] for check in result.checks
               if check["name"].endswith("cross_kv_cache"))


def test_other_live_request_mutation_fails_exact_preservation(tiny):
    declaration, consumer, _ = tiny
    entries = {"prefill": declaration("prefill", PagedSequenceView(), cross_value_bias=1e-6),
               "decode": declaration("decode", DenseSequenceView(), mutate_other=True)}
    consume, _ = consumer(entries)
    result = consume("decode", entries["decode"])
    assert result.status is Status.FAIL
    assert any(check["name"] == "trajectory.other_request_preserved.cross_kv_cache"
               and not check["passed"] and check["policy"] == "exact-v1"
               for check in result.checks)
    assert all(check["passed"] for check in result.checks
               if check["name"].startswith("trajectory.other_request_accuracy."))


def test_stochastic_declared_walks_use_seeded_distribution_and_one_shared_program(tiny):
    declaration, consumer, events = tiny
    entries = {"categorical": declaration("both", DenseSequenceView(), temperature=0.7)}
    consume, session = consumer(entries, flat_logits=True)
    result = consume("categorical", entries["categorical"])
    assert result.status is Status.PASS, result.to_dict()
    assert len(result.details["cases"]) == 2
    assert all(case["solver"]["witness"]["resources"][SAMPLER][0]["temperature"] == 0.7
               for case in result.details["cases"])
    distributions = [check for check in result.checks if check["name"] == "sampling.distribution"]
    assert len(distributions) == 2
    assert all(check["draws"] == 64 and check["distinct_tokens"] > 1 for check in distributions)
    assert sum(check["name"] == "sampling.seeded_replay" for check in result.checks) == 2
    trajectory, = result.details["trajectories"]["cases"]
    assert trajectory["executed"] and trajectory["selection_kind"] == "same_registration"
    assert trajectory["prefill"] == trajectory["decode"] == "categorical"
    assert trajectory["lifecycle"]["applicable"]
    # One preparation per independent witness, then one combined preparation
    # serving both walks and the two-live trajectory under one owner.
    assert len([event for event in events if event[0] == "prepare"]) == 3
    assert session.prepared == [] and session.resource_sets == []


def test_stochastic_fixed_argmax_is_not_qualified_by_lawful_tokens_alone(tiny):
    declaration, consumer, _ = tiny
    entries = {"fixed": declaration("both", DenseSequenceView(), temperature=1.0, fixed_argmax=True)}
    consume, _ = consumer(entries, flat_logits=True)
    result = consume("fixed", entries["fixed"])
    assert result.status is Status.FAIL
    failed = [check for check in result.checks if not check["passed"]]
    assert len(failed) == 2 and all(check["name"] == "sampling.distribution" for check in failed)
    assert all(check["passed"] for check in result.checks if check["policy"] == "categorical-support-v1")


def test_bad_decode_remains_a_failed_registration_with_trajectory_evidence(tiny):
    declaration, consumer, _ = tiny
    entries = {"prefill": declaration("prefill", PagedSequenceView()),
               "decode": declaration("decode", DenseSequenceView(), wrong=True)}
    consume, _ = consumer(entries)
    result = consume("decode", entries["decode"])
    assert result.status is Status.FAIL
    assert {check["name"] for check in result.checks if not check["passed"]} >= {
        "logits", "trajectory.decode.logits", "trajectory.reassigned.decode.logits"}
    assert result.details["trajectories"]["cases"][0]["executed"]


def test_invalid_resource_configuration_is_not_an_incompatible_pair_skip(tiny, monkeypatch):
    declaration, consumer, _ = tiny
    entries = {"prefill": declaration("prefill", PagedSequenceView()),
               "decode": declaration("decode", DenseSequenceView())}
    consume, _ = consumer(entries)

    def bad_config(*args, **kwargs):
        raise ValueError("malformed deployment budget")

    monkeypatch.setattr(cases, "resolve_bound_specs", bad_config)
    with pytest.raises(ValueError, match="malformed deployment budget"):
        consume("decode", entries["decode"])


def test_resource_construction_failure_keeps_completed_owners_for_managed_drain(monkeypatch):
    from mstar_accelerators.testing import whisper

    session = QualificationSession()
    first = object()
    specs = [SimpleNamespace(resource_key=name, depends_on=lambda: ()) for name in ("one", "two")]
    monkeypatch.setattr(whisper, "resolve_bound_specs", lambda values, *args, **kwargs: values)

    def build(spec, info, requests):
        if spec.resource_key == "two":
            raise RuntimeError("second resource construction failed")
        return first

    monkeypatch.setattr(whisper, "build_bound_resource", build)
    with pytest.raises(RuntimeError, match="second resource"):
        whisper._runtime_resources(None, torch.device("cpu"), {}, specs=specs, session=session)
    assert session.resource_sets == [{"one": first}]
    session.close()
    assert session.resource_sets == []
