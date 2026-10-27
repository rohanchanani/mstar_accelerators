"""Controller tests only: mocks establish no native execution or fence evidence."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from mstar import runtime_extensions

from mstar_accelerators.testing import hosting as pi
from mstar_accelerators.testing.harness import Result, Status, SuiteReport


class Owner:
    def __init__(self):
        self.state = "reserved"

    def status(self):
        return self.state


@pytest.fixture
def entries(monkeypatch):
    values = {"first": object(), "second": object(), "third": object()}
    monkeypatch.setattr(pi, "_entries", lambda source, job: values)
    return values


@pytest.mark.parametrize("verdict", [Status.PASS, Status.FAIL])
def test_publish_completed_verdict_only_after_managed_release(entries, tmp_path, monkeypatch, verdict):
    owner = Owner()
    path = tmp_path / "report.json"
    events = []

    def managed(worker_class, kwargs, extension, received_owner):
        assert received_owner is owner
        assert not path.exists()
        record = kwargs["record"]
        record["worker_closed"] = True
        results = [Result(name, verdict, "mock completed numerical evidence") for name in entries]
        record["results"].extend(results)
        record["report"] = SuiteReport(tuple(results)).to_dict()
        events.append("execution settled")
        owner.state = "released"
        events.append("native ownership released")

    monkeypatch.setattr(runtime_extensions, "run_managed_worker", managed)
    pi._managed("test:registry", SimpleNamespace(entries=tuple(entries), environment=SimpleNamespace(runtime=lambda source: object(), prepare_process=lambda: None, device="cpu")), "/unused", {}, 0, "unused", owner, path)
    events.append("report observed")
    report = json.loads(path.read_text())
    assert events == ["execution settled", "native ownership released", "report observed"]
    assert report["status"] == verdict.value
    assert report["complete"]
    assert report["runtime_released"]
    assert [item["name"] for item in report["accelerations"]] == list(entries)


def test_execution_failure_keeps_completed_results_and_diagnoses_aborted_entries(entries, tmp_path, monkeypatch):
    owner = Owner()
    path = tmp_path / "report.json"

    def managed(worker_class, kwargs, extension, received_owner):
        record = kwargs["record"]
        record["worker_closed"] = True
        record["results"].append(Result("first", Status.PASS, "completed before error"))
        record["active"] = "second"
        assert not path.exists()
        owner.state = "released"
        raise RuntimeError("execution failed; managed drain and release completed")

    monkeypatch.setattr(runtime_extensions, "run_managed_worker", managed)
    pi._managed("test:registry", SimpleNamespace(entries=tuple(entries), environment=SimpleNamespace(runtime=lambda source: object(), prepare_process=lambda: None, device="cpu")), "/unused", {}, 0, "unused", owner, path)
    report = json.loads(path.read_text())
    assert report["status"] == "FAIL" and not report["complete"]
    assert report["counts"] == {"PASS": 1, "FAIL": 1, "SKIP": 1}
    assert [(item["name"], item["status"]) for item in report["accelerations"]] == [
        ("first", "PASS"), ("second", "FAIL"), ("third", "SKIP"),
    ]
    assert "managed drain" in report["worker_error"]
    assert report["runtime_released"]


def test_early_driver_failure_preserves_registry_inventory(entries, tmp_path, monkeypatch):
    owner = Owner()
    path = tmp_path / "report.json"

    def managed(worker_class, kwargs, extension, received_owner):
        kwargs["record"]["worker_closed"] = True
        # No declaration or worker.run() entered; all names must already exist.
        assert kwargs["record"]["names"] == tuple(entries)
        owner.state = "released"
        raise RuntimeError("reference construction bug")

    monkeypatch.setattr(runtime_extensions, "run_managed_worker", managed)
    pi._managed("test:registry", SimpleNamespace(entries=tuple(entries), environment=SimpleNamespace(runtime=lambda source: object(), prepare_process=lambda: None, device="cpu")), "/unused", {}, 0, "unused", owner, path)
    report = json.loads(path.read_text())
    assert report["status"] == "FAIL" and not report["complete"]
    assert report["counts"] == {"PASS": 0, "FAIL": 0, "SKIP": 3}
    assert "reference construction bug" in report["worker_error"]
    assert [item["name"] for item in report["accelerations"]] == list(entries)


def test_uncertain_owner_exception_propagates_without_publishing_release(entries, tmp_path, monkeypatch):
    owner = Owner()
    path = tmp_path / "report.json"

    def managed(worker_class, kwargs, extension, received_owner):
        owner.state = "held"
        raise RuntimeError("completion is uncertain")

    monkeypatch.setattr(runtime_extensions, "run_managed_worker", managed)
    with pytest.raises(RuntimeError, match="uncertain"):
        pi._managed("test:registry", SimpleNamespace(entries=tuple(entries), environment=SimpleNamespace(runtime=lambda source: object(), prepare_process=lambda: None, device="cpu")), "/unused", {}, 0, "unused", owner, path)
    assert owner.state == "held" and not path.exists()


def test_missing_report_cannot_become_a_success(entries, tmp_path, monkeypatch):
    owner = Owner()
    path = tmp_path / "report.json"

    def managed(worker_class, kwargs, extension, received_owner):
        kwargs["record"]["worker_closed"] = True
        owner.state = "released"

    monkeypatch.setattr(runtime_extensions, "run_managed_worker", managed)
    pi._managed("test:registry", SimpleNamespace(entries=tuple(entries), environment=SimpleNamespace(runtime=lambda source: object(), prepare_process=lambda: None, device="cpu")), "/unused", {}, 0, "unused", owner, path)
    report = json.loads(path.read_text())
    assert report["status"] == "FAIL" and not report["complete"]
    assert "without producing a registry report" in report["worker_error"]


@pytest.mark.parametrize("settlement_failure", [False, True])
def test_registered_implementation_chooses_provider_through_the_actual_reference_adapter(monkeypatch, settlement_failure):
    """Exercise driver -> adapter -> registered prepare, with no native work."""
    from mstar.model.whisper import whisper_model
    from mstar.model.whisper.config import WhisperModelConfig
    from mstar.model.whisper.submodules import WhisperEncoderSubmodule

    from mstar_accelerators import Acceleration, Eq, Implementation, When
    from mstar_accelerators.backends.graph import (
        GraphProgram,
        NativeGraphProvider,
        prepare_graph_programs,
    )
    from mstar_accelerators.testing import whisper

    config = WhisperModelConfig(d_model=8, num_mel_bins=4, max_source_positions=3,
                                lang_to_id={"<|en|>": 1}, task_to_id={"transcribe": 2})
    features = torch.ones(4, 6)
    expected = torch.ones(3, 8)
    model = whisper_model.WhisperModel.__new__(whisper_model.WhisperModel)
    model.config = config
    model.create_submodule = lambda name: WhisperEncoderSubmodule(config)
    model.get_node_resources = list
    model.logical_weights = lambda: SimpleNamespace(specs=(), load=lambda names: {})
    monkeypatch.setattr(whisper_model, "WhisperModel", lambda checkpoint: model)
    monkeypatch.setattr(whisper, "audio_inputs", lambda *args, **kwargs: (
        {"audio_features": [features]}, {"scope": "synthetic host fixture"},
    ))
    monkeypatch.setattr(WhisperEncoderSubmodule, "make_reference_program", lambda self, binding:
                        SimpleNamespace(run=lambda value: expected.to(dtype=binding.dtype), modules=lambda: ()))
    backend = ModuleType("mstar_pi")
    def synchronize():
        raise RuntimeError("case completion is uncertain")

    backend.current_stream = lambda device: SimpleNamespace(synchronize=synchronize) if settlement_failure else None
    monkeypatch.setitem(sys.modules, "mstar_pi", backend)
    calls = []

    class Lease:
        def submit(self, inputs, outputs):
            assert torch.equal(inputs["audio_features"], features)
            outputs["encoder_states"].copy_(expected)
            calls.append("fake provider submission")

        def close(self):
            calls.append("lease close")

    class Program:
        def try_acquire(self):
            return Lease()

        def close(self):
            calls.append("program close")

    def provider_prepare(self, build, constants, *, slots):
        assert isinstance(self, NativeGraphProvider)
        assert constants == {} and slots == 1
        calls.append("explicit provider preparation")
        return Program()

    monkeypatch.setattr(NativeGraphProvider, "prepare", provider_prepare)

    def prepare(binding, resources):
        return prepare_graph_programs(binding, resources, {
            "prefill": GraphProgram(lambda: None, slots=1),
        }, provider=NativeGraphProvider())

    acceleration = Acceleration(WhisperEncoderSubmodule,
                                When(Eq("device_type", "cpu"), Eq("graph_walk", "prefill"), Eq("batch_size", 1)),
                                Implementation(prepare))
    monkeypatch.setattr(pi, "_entries", lambda source, job: {"encoder": acceleration})
    monkeypatch.setattr(pi, "load_accelerations", lambda source: {"encoder": acceleration})
    record = {"results": [], "active": None}
    from mstar_accelerators.testing.planning import Job
    case = SimpleNamespace(describe=lambda checkpoint: model, deployment_config={"autocast_dtype": "float32"}, identity=lambda: {})
    environment = SimpleNamespace(stream=backend.current_stream, runtime_options={})
    job = Job(case, environment, 1, ('audio_encoder',), ('encoder',), ())
    worker = pi.Worker(worker_id="controller-test", device=torch.device('cpu'),
        runtime_extension=SimpleNamespace(drain=lambda: calls.append("drain")),
        source="synthetic:registry", job=job, checkpoint="/unused", fixtures={'audio': '/unused'},
        record=record, rank=0, rendezvous='unused')
    if settlement_failure:
        with pytest.raises(RuntimeError, match="completion is uncertain"):
            worker.run()
        # Both independent reference and candidate engines remain rooted until drain.
        assert len(worker.session.engines) == 2 and worker.session.bindings
        assert calls == ["explicit provider preparation", "fake provider submission", "lease close"]
        # Simulate the existing owner's later successful drain. The failed case
        # itself did not drop roots, publish evidence, or submit another case.
        worker.close()
        assert calls[-2:] == ["drain", "program close"]
        assert worker.session.engines == [] and "report" not in record
        return
    worker.run()
    report = record["report"]
    assert report["status"] == "PASS" and report["complete"], [c for a in report["accelerations"] for c in a["checks"] if not c["passed"]]
    result = report["accelerations"][0]
    assert result["details"]["executed"] and result["details"]["cases"][0]["solver"]["status"] == "SAT"
    assert calls == ["explicit provider preparation", "fake provider submission", "lease close",
                     "program close", "drain"]
