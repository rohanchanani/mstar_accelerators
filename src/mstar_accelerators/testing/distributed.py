"""CPU role environment for the centralized Whisper differential consumer.

The environment hosts the encoder on role zero and the decoder on both roles,
matching the CPU deployment. It uses M*'s groups, binding, selection and admission
mechanics. Ordinary M* service remains a separate integration gate.
"""
from __future__ import annotations

from datetime import timedelta
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import socket
import tempfile
import time
import traceback

import torch

from ..sources import load_accelerations, normalize_source
from .environments import combine
from .harness import Result, Status, SuiteReport, run_registry
from .numerics import DEFAULT
from .policy import consumer as central_consumer
from .configuration import environment


def _write(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _aborted(names, results, active, error):
    complete = {result.name for result in results}
    results = list(results)
    for name in names:
        if name not in complete:
            results.append(Result(name, Status.FAIL if name == active else Status.SKIP,
                                  error if name == active else "not executed after role failure",
                                  details={"scope": "aborted CPU qualification"}))
    return SuiteReport(tuple(results)).to_dict() | {"status": "FAIL", "driver_error": error}


def _role(rank, registry_spec, options, rendezvous, directory):
    from mstar.distributed.communication import CommGroup, WorkerParallelGroups
    from mstar.model.whisper.whisper_model import WhisperModel

    from ..backends.torch.runtime import TorchRuntime
    from .whisper import QualificationSession, audio_inputs

    path = Path(directory) / f"role-{rank}.json"
    entries = load_accelerations(registry_spec)
    session = QualificationSession()
    runtime = TorchRuntime({"accelerations": registry_spec, "threads": options.get("threads", 1)})
    groups = WorkerParallelGroups(num_workers=2, global_rank=rank, any_parallelism=True,
                                  world_parallel_groups=[(0, 1)])
    groups.add("decoder", CommGroup(rank, rank, [0, 1]))
    if rank == 0:
        groups.add("audio_encoder", CommGroup(rank, 0, [0]))
    active, results = None, []
    try:
        runtime.initialize(torch.device("cpu"))
        groups.init_dist(rendezvous, torch.device("cpu"), timeout=timedelta(
            seconds=options.get("collective_timeout_seconds", 180)))
        control = groups.get_joint_group_for_node("decoder")
        inventories = control.all_gather_object(tuple(entries))
        if any(value != inventories[0] for value in inventories[1:]):
            raise ValueError("qualification roles loaded different corpus inventories")
        model = WhisperModel(options["checkpoint"])
        inputs, audio = audio_inputs(model, options["audio"],
                                    language=options.get("language", "en"),
                                    task=options.get("task", "transcribe"))
        consumer = central_consumer(model, features=inputs["audio_features"][0], options=options, session=session,
                              device="cpu", corpus=entries, parallel_groups=groups,
                              scope="registered Torch execution in actual CPU role processes")

        def consume(name, acceleration):
            nonlocal active
            active = name
            result = consumer(name, acceleration)
            runtime.drain()
            session.close()
            # Keep every role on the same corpus entry even when only one role
            # hosts that reference, or a declaration is inapplicable here.
            finished = control.all_gather_object(name)
            if any(value != name for value in finished):
                raise ValueError("qualification roles left a corpus entry out of order")
            results.append(result)
            active = None
            return result

        report = run_registry(entries, consume).to_dict()
        runtime.drain()
        session.close()
        groups.close()
        runtime.close()
        report.update(role=rank, pid=mp.current_process().pid, audio=audio,
                      checkpoint=str(Path(options["checkpoint"]).resolve()),
                      groups_released=True, runtime_released=True, policy=DEFAULT.name)
        _write(path, report)
    except BaseException:
        groups.mark_failed()
        report = _aborted(entries, results, active, traceback.format_exc())
        report.update(role=rank, pid=mp.current_process().pid, groups_released=False,
                      runtime_released=False, policy=DEFAULT.name)
        _write(path, report)
        # The parent stops the other CPU role. Failed collective groups do not
        # enter an ordinary teardown collective or another implementation.
        raise SystemExit(1)


def run(registry_spec, options):
    options = environment("cpu", options)
    registry_spec = normalize_source(registry_spec)
    entries = load_accelerations(registry_spec)
    checkpoint, audio = options.get("checkpoint"), options.get("audio")
    reason = None
    if not checkpoint or not Path(checkpoint).is_dir() or not audio or not Path(audio).is_file():
        reason = ("reference_assets_unavailable", "CPU role qualification needs checkpoint and audio assets")
    elif importlib.util.find_spec("transformers") is None:
        reason = ("reference_dependency", "the optional Whisper reference requires transformers")
    elif not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        reason = ("target_unavailable", "this Torch build has no CPU Gloo backend")
    if reason is not None:
        return run_registry(entries, lambda name, acceleration: Result(
            name, Status.SKIP, reason[1], details={"code": reason[0], "scope": "not executed"},
        )).to_dict() | {"roles_started": False, "policy": DEFAULT.name}
    if options.get("tp_size", 2) != 2 or options.get("threads", 1) < 1:
        raise ValueError("this execution environment requires TP2 and positive CPU threads")
    deadline_seconds = options.get("deadline_seconds", 7200)
    if deadline_seconds <= 0 or options.get("collective_timeout_seconds", 180) <= 0:
        raise ValueError("CPU qualification deadlines must be positive")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        rendezvous = f"tcp://127.0.0.1:{listener.getsockname()[1]}"
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="mstar-cpu-qualification-") as directory:
        processes = [context.Process(target=_role,
                     args=(rank, registry_spec, options, rendezvous, directory),
                     name=f"registry-cpu-role-{rank}") for rank in range(2)]
        failed = None
        started = []
        try:
            for process in processes:
                process.start()
                started.append(process)
            deadline = time.monotonic() + deadline_seconds
            while any(process.is_alive() for process in started):
                if any(process.exitcode not in (None, 0) for process in started):
                    failed = "a CPU qualification role failed; peers were stopped without retry"
                    break
                if time.monotonic() >= deadline:
                    failed = "CPU role qualification exceeded its recorded deadline"
                    break
                for process in started:
                    process.join(timeout=0.1)
        except BaseException:
            failed = traceback.format_exc()
        finally:
            if failed is not None:
                # These processes initialize CPU execution only. This is not a
                # device settlement mechanism and never authorizes Pi reuse.
                for process in started:
                    if process.is_alive():
                        process.terminate()
                for process in started:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join()
        reports = {}
        for rank, process in enumerate(processes):
            path = Path(directory) / f"role-{rank}.json"
            if path.exists():
                reports[str(rank)] = json.loads(path.read_text())
            else:
                reports[str(rank)] = _aborted(entries, (), None,
                    f"role {rank} produced no report; exit={process.exitcode}; {failed}")
            if process.exitcode != 0:
                failed = failed or f"CPU role {rank} exited {process.exitcode}"
        rows = {rank: {result["name"]: result for result in report["accelerations"]}
                for rank, report in reports.items()}
        suite = run_registry(entries, lambda name, acceleration: combine(
            name, {rank: results[name] for rank, results in rows.items()},
            scope="all participating CPU roles; encoder TP1 and decoder TP2"))
        report = suite.to_dict() | {
            "roles_started": bool(started), "policy": DEFAULT.name,
            "roles": {rank: {key: value for key, value in result.items() if key != "accelerations"}
                      for rank, result in reports.items()},
            "exit_codes": [process.exitcode for process in processes],
            "scope": "centralized differential consumer in CPU processes; not ordinary service evidence",
        }
        if failed is not None or any(result["status"] == "FAIL" for result in reports.values()):
            report.update(status="FAIL", driver_error=failed or "CPU role report failed")
        return report
