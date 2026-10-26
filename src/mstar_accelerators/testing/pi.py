"""Pi qualification driver using M*'s existing managed worker ownership.

This is optimization-platform orchestration, not a serving adapter or a CPU
fallback. Target/reference absence produces SKIP before reservation. Accepted
native work stays under the established drain/fatal-hold/release mechanism.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import multiprocessing as mp
import tempfile
from pathlib import Path

import torch

from ..sources import load_accelerations, normalize_source
from .harness import Result, Status, SuiteReport, run_registry
from .numerics import DEFAULT
from .policy import consumer as central_consumer
from .configuration import environment


def _registry(spec):
    return load_accelerations(spec)


def _skip(entries, code, reason):
    return run_registry(entries, lambda name, acceleration: Result(
        name, Status.SKIP, reason, details={"code": code, "scope": "not executed"},
    )).to_dict() | {"native_owner_reserved": False, "policy": DEFAULT.name}


class _Worker:
    def __init__(self, *, worker_id, device, runtime_extension, registry_spec, options, record):
        del worker_id
        from .whisper import QualificationSession

        self.extension = runtime_extension
        self.session = QualificationSession()
        self.device = device
        self.registry_spec = registry_spec
        self.options = options
        self.record = record
        self.stop_requested = False
        torch.set_num_threads(options.get("threads", 4))

    def request_stop(self):
        self.stop_requested = True

    def run(self):
        import mstar_pi
        from mstar.model.whisper.whisper_model import WhisperModel

        from .whisper import audio_inputs

        entries = _registry(self.registry_spec)
        self.record["names"] = tuple(entries)
        model = WhisperModel(self.options["checkpoint"])
        inputs, identity = audio_inputs(model, self.options["audio"],
                                       language=self.options.get("language", "en"),
                                       task=self.options.get("task", "transcribe"))
        consumer = central_consumer(model, features=inputs["audio_features"][0], options=self.options,
                              session=self.session, device=self.device,
                              corpus=entries,
                              stream=mstar_pi.current_stream(self.device),
                              scope="fresh native execution of registered declaration")

        def consume(name, acceleration):
            if self.stop_requested:
                raise InterruptedError("qualification stopped before the next declaration")
            self.record["active"] = name
            result = consumer(name, acceleration)
            # A growing registry must not retain every tested program/state.
            # This is the qualification read/release boundary, outside serving.
            self.extension.drain()
            self.session.close()
            self.record["results"].append(result)
            self.record["active"] = None
            return result

        suite = run_registry(entries, consume)
        self.record["report"] = suite.to_dict() | {
            "policy": DEFAULT.name, "audio": identity,
            "checkpoint": str(Path(self.options["checkpoint"]).resolve()),
            "scope": "fresh native execution of registered declarations; consumer-selected cases",
        }
        if self.stop_requested:
            raise InterruptedError("qualification stopped after accepted work settled")

    def close(self):
        self.extension.drain()
        self.session.close()


def _managed(registry_spec, options, extension, owner, report_path):
    from mstar.runtime_extensions import run_managed_worker

    record = {"names": tuple(_registry(registry_spec)), "results": [], "active": None}
    failure = None
    try:
        run_managed_worker(_Worker, {
            "worker_id": "registry-qualification", "device": "mstar_pi:0",
            "registry_spec": registry_spec, "options": options, "record": record,
        }, extension, owner)
    except Exception as exc:
        # run_managed_worker returns an ordinary exception only after real
        # native release. An uncertain context remains in its existing hold.
        if owner.status() != "released":
            raise
        failure = f"{type(exc).__name__}: {exc}"
    if failure is None:
        report = record.get("report")
        if report is None:
            failure = "qualification stopped without producing a registry report"
    if failure is not None:
        completed = {result.name for result in record["results"]}
        results = list(record["results"])
        for name in record["names"]:
            if name not in completed:
                active = name == record["active"]
                results.append(Result(name, Status.FAIL if active else Status.SKIP,
                                      failure if active else "not executed after driver failure",
                                      details={"scope": "aborted qualification"}))
        report = SuiteReport(tuple(results)).to_dict()
        report.update(status="FAIL", driver_error=failure, policy=DEFAULT.name)
    report.update(native_owner_reserved=True, native_owner_released=True)
    # Only host scalars/metrics are retained here; worker close and native
    # shutdown have already proven release before this report is published.
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(registry_spec: str, options: dict) -> dict:
    """Run this driver's supported corpus, diagnosing unavailable prerequisites.

    Options describe assets and managed execution, never correctness policy.
    The central coordinator selects this environment host; policy.py selects
    the references and checks, and M* owns candidate execution.
    """
    options = environment("pi", options)
    registry_spec = normalize_source(registry_spec)
    entries = _registry(registry_spec)
    checkpoint, audio = options.get("checkpoint"), options.get("audio")
    if not checkpoint or not Path(checkpoint).is_dir() or not audio or not Path(audio).is_file():
        return _skip(entries, "reference_assets_unavailable",
                     "this driver needs an available checkpoint directory and audio fixture")
    if importlib.util.find_spec("transformers") is None:
        return _skip(entries, "reference_dependency",
                     "the optional Whisper reference requires transformers")
    if importlib.util.find_spec("mstar_pi") is None:
        return _skip(entries, "target_unavailable", "the optional mstar_pi backend is not installed")
    import mstar_pi

    if not mstar_pi.is_available():
        return _skip(entries, "target_unavailable", "the Pi target capability probe is unavailable")
    if options.get("threads", 4) <= 0:
        raise ValueError("threads must be positive")
    from mstar_accelerators.backends.pi.runtime import PiRuntime

    extension = PiRuntime({"accelerations": registry_spec, "native": {
        "wait_timeout_ms": options.get("wait_timeout_ms", 180000),
        "validation": options.get("validation", True),
    }})
    context = extension.admission_context()
    owner = extension.reserve_worker(context, "registry-qualification")
    with tempfile.TemporaryDirectory(prefix="mstar-qualification-") as directory:
        report_path = str(Path(directory) / "report.json")
        if options.get("in_process", False):
            _managed(registry_spec, options, extension, owner, report_path)
        else:
            process = mp.get_context("spawn").Process(
                target=_managed, args=(registry_spec, options, extension, owner, report_path),
                name="registry-qualification",
            )
            try:
                process.start()
            except BaseException:
                owner.unknown("qualification spawn failed without an acknowledged child release")
                raise
            process.join()
            if owner.status() != "released":
                owner.unknown("qualification child exited without a native release acknowledgement")
                raise RuntimeError("qualification child did not acknowledge release")
            if process.exitcode != 0:
                raise RuntimeError(f"qualification child exited with status {process.exitcode}")
        report = json.loads(Path(report_path).read_text())
    context.close()
    return report
