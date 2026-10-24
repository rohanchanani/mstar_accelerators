#!/usr/bin/env python3
"""Exercise the ordinary M* API, conductor, scheduler and spawned worker.

Run the native deployment under backends/pi/tools/run_hardware.py --managed.
This client starts M* itself, records every response, then requests graceful
API shutdown. It never kills a worker or substitutes process exit for release.
--host-test selects a separate, explicit CPU graph-evaluator test deployment.
An explicitly CPU TP2 deployment uses the ordinary Torch runtime with no Pi lease.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import re
import subprocess
import sys
import time
import traceback
import uuid

import requests
import yaml


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/whisper"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def acceleration_identity(config):
    """Record the selected declarations without assuming their filenames."""
    from mstar_accelerators.sources import acceleration_files, load_accelerations, normalize_source
    source = normalize_source(config["runtime_extension"]["accelerations"])
    if ":" in source and not Path(source).exists():
        return {"source": source, "declarations": list(load_accelerations(source)),
                "sha256": None}
    files = acceleration_files(source)
    return {"source": source, "declarations": list(files), "sha256": {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_ready(process, url, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"M* exited during startup: {process.returncode}")
        try:
            response = requests.get(url + "/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(.25)
    raise TimeoutError("M* did not become ready before the recorded startup deadline")


def transcribe(url, audio, expected, output, name, timeout, *, stream=False, cancel=False,
               temperature=0.0, seed=None):
    started = time.monotonic()
    record = {"case": name, "audio": audio.name, "expected": expected,
              "stream": stream, "cancel_after_first_delta": cancel, "events": [],
              "temperature": temperature, "seed": seed}
    data = {"model": "whisper", "language": "en", "temperature": str(temperature),
            "response_format": "json", "stream": "true" if stream else "false"}
    if seed is not None:
        data["seed"] = str(seed)
    try:
        with audio.open("rb") as source, requests.post(
            url + "/v1/audio/transcriptions",
            files={"file": (audio.name, source, "audio/wav")},
            data=data,
            stream=stream, timeout=(5, timeout),
        ) as response:
            record["http_status"] = response.status_code
            if response.status_code != 200:
                record["body"] = response.text
                raise AssertionError(f"{name}: HTTP {response.status_code}: {response.text}")
            if not stream:
                record["response"] = response.json()
                if not isinstance(record["response"].get("text"), str) or not record["response"]["text"]:
                    raise AssertionError(f"{name}: no completed transcript")
                if expected is not None and record["response"]["text"] != expected:
                    raise AssertionError(f"{name}: transcript differs from the original reference")
            else:
                text, done, terminator = "", None, False
                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if not line.startswith("data:"):
                        continue
                    payload = line.removeprefix("data:").strip()
                    if payload == "[DONE]":
                        terminator = True
                        break
                    event = json.loads(payload)
                    record["events"].append(event)
                    if event["type"] == "error":
                        raise AssertionError(f"{name}: stream error: {event}")
                    if event["type"] == "transcript.text.delta":
                        text += event["delta"]
                        if cancel and text:
                            # Closing the client is the public cancellation
                            # operation; M* owns in-flight drain and KV cleanup.
                            record["cancelled_prefix"] = text
                            break
                    elif event["type"] == "transcript.text.done":
                        done = event["text"]
                if cancel:
                    if not record.get("cancelled_prefix") or not expected.startswith(text.lstrip()):
                        raise AssertionError("cancellation did not observe a valid live transcript prefix")
                elif not terminator or text.strip() != expected or done != expected:
                    raise AssertionError(f"{name}: incomplete or incorrect streaming transcript")
        record["passed"] = True
        return record
    except BaseException as error:
        record.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        # This is observed end-to-end time, not a speedup claim.
        save(output / f"{name}.json", record)


def reject_unsupported_sampling(url, audio, output, timeout):
    """The authored support union rejects canonical policy at request admission."""
    with audio.open("rb") as source:
        response = requests.post(url + "/v1/audio/transcriptions",
                                 files={"file": (audio.name, source, "audio/wav")},
                                 data={"model": "whisper", "language": "en", "temperature": "0.5",
                                       "top_p": "0.9"},
                                 timeout=(5, timeout))
    record = {"case": "06_unsupported_sampling", "http_status": response.status_code,
              "body": response.text, "temperature": .5, "top_p": .9,
              "passed": response.status_code == 400 and "request policy is unsupported" in response.text.lower()}
    save(output / "06_unsupported_sampling.json", record)
    if not record["passed"]:
        raise AssertionError("unsupported sampling did not receive the declared policy rejection")
    return record


def cpu_lifecycle(log):
    ready = re.findall(r"\[([^\]]+)\].*TORCH_RUNTIME_READY pid=(\d+)", log)
    released = re.findall(r"\[([^\]]+)\].*TORCH_RUNTIME_RELEASED pid=(\d+)", log)
    groups = re.findall(r"Process groups closed: rank=(\d+) failed=(True|False)", log)
    choices = re.findall(
        r"\[([^\]]+)\].*Bound execution: node=decoder walk=(\S+) implementation=(\S+) rows=(\d+)", log)
    per_worker = {worker: [] for worker, _ in ready}
    for worker, walk, implementation, rows in choices:
        per_worker.setdefault(worker, []).append(dict(walk=walk, implementation=implementation, rows=int(rows)))
    return {
        "ready": [dict(worker=worker, pid=int(pid)) for worker, pid in ready],
        "released": [dict(worker=worker, pid=int(pid)) for worker, pid in released],
        "process_groups": [dict(rank=int(rank), failed=failed == "True") for rank, failed in groups],
        "decoder_executions": per_worker,
    }


def process_running(pid):
    # The target is Linux. A zombie has stopped executing; the conductor reaps it.
    path = Path(f"/proc/{pid}/stat")
    try:
        state = path.read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


def observe_instance_failure(url, audio, output, request_timeout, shutdown_timeout):
    record = {"case": "controlled_instance_failure", "passed": False}
    try:
        try:
            with audio.open("rb") as source:
                response = requests.post(url + "/v1/audio/transcriptions",
                                         files={"file": (audio.name, source, "audio/wav")},
                                         data={"model": "whisper", "language": "en", "temperature": "0"},
                                         timeout=(5, request_timeout))
            record.update(http_status=response.status_code, body=response.text)
            if response.status_code != 503:
                raise AssertionError("instance-fatal request did not receive HTTP 503")
        except requests.ConnectionError as error:
            record["connection_closed"] = str(error)
        deadline = time.monotonic() + shutdown_timeout
        while time.monotonic() < deadline:
            log = (output / "server.log").read_text()
            workers = cpu_lifecycle(log)["ready"]
            if len(workers) == 2 and all(not process_running(worker["pid"]) for worker in workers):
                break
            time.sleep(.25)
        else:
            raise TimeoutError("instance-fatal CPU workers did not stop before harness cleanup")
        if "after_execution_failure" not in log or "INSTANCE_FATAL:" not in log or "stopping deployment" not in log:
            raise AssertionError("worker retirement lacks the injected failure and conductor consumption receipts")
        record.update(passed=True, workers=workers, stopped_before_harness_cleanup=True)
        return record
    finally:
        save(output / "controlled_instance_failure.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=FIXTURES / "reference.json")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--shutdown-timeout", type=float, default=180)
    parser.add_argument("--host-test", action="store_true")
    parser.add_argument("--stochastic-temperature", type=float,
                        help="also exercise the authored positive-temperature declaration with fixed seeds")
    parser.add_argument("--distributed-control", choices=("refuse_once", "fail_after_execution"),
                        help="run a separately recorded fault control through the ordinary CPU TP2 service")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    oracle = json.loads(args.oracle.read_text())
    for name, expected in oracle["cases"].items():
        if hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() != expected["audio_sha256"]:
            raise ValueError(f"fixture identity mismatch: {name}")
    config = yaml.safe_load(args.config.read_text())
    source_identity = acceleration_identity(config)
    cpu = config.get("device_type") == "cpu" and not args.host_test
    if args.distributed_control and not cpu:
        raise ValueError("distributed controls require an explicit ordinary CPU deployment")
    fatal_control = args.distributed_control == "fail_after_execution"
    if fatal_control and not sys.platform.startswith("linux"):
        raise RuntimeError("the CPU retirement control requires Linux /proc process observations")

    lock = None
    context = None
    admission_environment = {}
    if args.host_test:
        sys.path.insert(0, str(ROOT))
        from tests.whisper.cpu_runtime import AdmissionContext, ENVIRONMENT
        # A native-free cohort exclusively for this host integration test.
        # It is never used to authorize physical Pi execution.
        lock = (output / "cpu-test.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX)
        context = AdmissionContext.create(lock.fileno(), output / "cpu-admission", str(uuid.uuid4()))
        config["device_type"] = "cpu"
        source = config["runtime_extension"]["accelerations"]
        config["runtime_extension"] = {
            "factory": "tests.whisper.cpu_runtime:CpuRuntime",
            "accelerations": source,
        }
        config_path = output / "cpu-deployment.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        admission_environment[ENVIRONMENT] = context.environment()
    elif cpu:
        config_path = args.config.resolve()
        if args.distributed_control:
            config["runtime_extension"]["factory"] = (
                "mstar_accelerators.testing.distributed_controls:ControlledTorchRuntime")
            config["runtime_extension"]["distributed_control"] = args.distributed_control
            config_path = output / "controlled-cpu-deployment.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    else:
        from mstar_pi.admission import AdmissionContext, ENVIRONMENT
        context = AdmissionContext.from_environment()
        if context is None:
            raise RuntimeError("native demo requires the managed physical-device launcher")
        config_path = args.config.resolve()
        admission_environment[ENVIRONMENT] = context.environment()
    port = free_port()
    command = [sys.executable, "-m", "mstar.api_server.entrypoint", "--config", str(config_path),
               "--host", "127.0.0.1", "--port", str(port), "--tensor-comm-protocol", "SHM",
               "--socket-path-prefix", str(output / "sockets") + "/",
               "--upload-dir", str(output / "uploads"), "--timeout", str(args.request_timeout),
               "--log-level", "DEBUG", "--log-stats-file", str(output / "request-profiles.jsonl")]
    environment = {**os.environ, **admission_environment,
                   "MSTAR_RUST_ZMQ": "0", "MSTAR_RUST_GRAPH": "0",
                   "MSTAR_WIRE_CODEC": "msgpack", "MSTAR_PRE_PLAN_SPEC": "0",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                   "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    save(output / "launch.json", {"command": command, "host_test": args.host_test,
                                  "session": None if context is None else context.session, "port": port,
                                  "ordinary_cpu": cpu, "distributed_control": args.distributed_control,
                                  "config": str(config_path), "accelerations": source_identity})
    process = None
    failure = None
    report = {"gate": "mstar_whisper_api", "host_test": args.host_test, "cases": [],
              "ordinary_cpu": cpu, "distributed_control": args.distributed_control,
              "accelerations": source_identity}
    greedy_identities = set()
    try:
        with (output / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=environment,
                                       pass_fds=() if context is None else (context.fd,),
                                       stdout=log, stderr=subprocess.STDOUT)
        wait_ready(process, f"http://127.0.0.1:{port}", args.startup_timeout)
        url = f"http://127.0.0.1:{port}"
        full, tail = (FIXTURES / name for name in ("jfk16k.wav", "jfk_tail16k.wav"))
        expected_full = oracle["cases"][full.name]["text"].strip()
        expected_tail = oracle["cases"][tail.name]["text"].strip()
        def run(audio, expected, name, **kwargs):
            return transcribe(url, audio, expected, output, name, args.request_timeout, **kwargs)
        if fatal_control:
            report["cases"].append(observe_instance_failure(
                url, full, output, args.request_timeout, args.shutdown_timeout))
        else:
            report["cases"].append(run(full, expected_full, "01_full"))
            with ThreadPoolExecutor(max_workers=2) as clients:
                requests_in_flight = [clients.submit(run, full, expected_full, "02_concurrent_full"),
                                      clients.submit(run, tail, expected_tail, "03_concurrent_tail")]
                report["cases"].extend(future.result() for future in requests_in_flight)
            report["cases"].append(run(tail, expected_tail, "04_stream", stream=True))
            report["cases"].append(run(full, expected_full, "05_cancel", stream=True, cancel=True))
            report["cases"].append(reject_unsupported_sampling(url, full, output, args.request_timeout))
            report["cases"].append(run(full, expected_full, "07_after_cancel_and_rejection"))
            if args.stochastic_temperature is not None:
                # These first seven requests establish the actual greedy identities.
                # Later positive-policy requests must select additional programs.
                greedy_identities = set(re.findall(
                    r"Bound execution: node=(\S+) walk=(\S+) implementation=(\S+) rows=\d+",
                    (output / "server.log").read_text()))
                policy = {"temperature": args.stochastic_temperature, "seed": 20261023}
                stochastic = run(full, None, "08_stochastic", **policy)
                report["cases"].append(stochastic)
                sampled_text = stochastic["response"]["text"]
                report["cases"].append(run(full, sampled_text, "09_seeded_repeat", **policy))
                with ThreadPoolExecutor(max_workers=2) as clients:
                    mixed = [clients.submit(run, tail, expected_tail, "10_mixed_greedy"),
                             clients.submit(run, full, sampled_text, "11_mixed_stochastic", **policy)]
                    report["cases"].extend(future.result() for future in mixed)
                report["cases"].append(run(full, sampled_text, "12_stochastic_stream", stream=True, **policy))
                report["cases"].append(run(full, sampled_text, "13_stochastic_cancel", stream=True,
                                           cancel=True, **policy))
                report["cases"].append(run(full, sampled_text, "14_after_stochastic_cancel", **policy))
                report["sampling_scope"] = (
                    "Fixed-seed public transcript replay, including mixed requests and cancellation; "
                    "central qualification owns token/distribution and resource-effect assertions.")
    except BaseException as error:
        failure = error
        report["error"] = traceback.format_exc()
    finally:
        if process is not None and process.poll() is None:
            # Signal only M*'s API parent. Its existing shutdown reaches the
            # conductor, which asks managed workers to drain and release.
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=args.shutdown_timeout)
            except subprocess.TimeoutExpired as error:
                if context is not None:
                    context.quarantine("demo API did not finish graceful shutdown")
                failure = failure or error
                report["shutdown_error"] = "No process was killed; supervised recovery is required."
        owners = [] if context is None else context.owners()
        report["owners"] = owners
        report["server_exit_code"] = None if process is None else process.poll()
        if any(owner["state"] != "released" for owner in owners):
            context.quarantine("demo ended without every worker's release acknowledgement")
            failure = failure or RuntimeError("native owners were not released")
        if process is None or (process.returncode != 0 and not fatal_control):
            failure = failure or RuntimeError("M* API did not exit successfully")
        if context is not None and not owners:
            failure = failure or RuntimeError("M* never reserved a worker")
        log_path = output / "server.log"
        log = log_path.read_text() if log_path.exists() else ""
        choices = re.findall(r"Bound execution: node=(\S+) walk=(\S+) implementation=(\S+) rows=(\d+)", log)
        report["executions"] = [dict(node=node, walk=walk, implementation=implementation, rows=int(rows))
                                for node, walk, implementation, rows in choices]
        if args.stochastic_temperature is not None:
            positive_identities = {(node, walk, implementation) for node, walk, implementation, _ in choices}
            positive_identities -= greedy_identities
            report["policy_dispatch"] = {"greedy": sorted(greedy_identities),
                                         "additional_positive": sorted(positive_identities)}
            if not {("decoder", "prefill"), ("decoder", "decode")} <= {
                (node, walk) for node, walk, _ in positive_identities
            }:
                failure = failure or RuntimeError("positive requests did not select additional prefill/decode programs")
        required = {("audio_encoder", "prefill"), ("decoder", "prefill")}
        if not fatal_control:
            required.add(("decoder", "decode"))
        if not required <= {(choice["node"], choice["walk"]) for choice in report["executions"]}:
            failure = failure or RuntimeError("the server log lacks required implementation executions")
        if cpu:
            lifecycle = cpu_lifecycle(log)
            report["cpu_lifecycle"] = lifecycle
            ranks = {rank for group in config["node_groups"] for rank in group["ranks"]}
            if len({row["pid"] for row in lifecycle["ready"]}) != len(ranks):
                failure = failure or RuntimeError("ordinary CPU service did not initialize every placed worker")
            decoder_sequences = list(lifecycle["decoder_executions"].values())
            if len(decoder_sequences) != 2 or not decoder_sequences[0] or decoder_sequences[0] != decoder_sequences[1]:
                failure = failure or RuntimeError("CPU TP2 workers did not execute one agreed decoder sequence")
            if fatal_control and any(len(sequence) != 1 for sequence in decoder_sequences):
                failure = failure or RuntimeError("an instance-fatal decoder was retried locally")
            if not fatal_control:
                if sorted(lifecycle["ready"], key=lambda row: row["pid"]) != sorted(lifecycle["released"], key=lambda row: row["pid"]):
                    failure = failure or RuntimeError("ordinary CPU workers did not acknowledge runtime release")
                closed = {row["rank"] for row in lifecycle["process_groups"] if not row["failed"]}
                if closed != ranks:
                    failure = failure or RuntimeError("ordinary CPU process groups did not close normally")
            controls = [json.loads(line.split("DISTRIBUTED_CONTROL ", 1)[1])
                        for line in log.splitlines() if "DISTRIBUTED_CONTROL " in line]
            report["control_receipts"] = controls
            report["agreement_receipts"] = [line for line in log.splitlines() if "Execution agreement:" in line]
            if args.distributed_control == "refuse_once":
                refused = [row for row in controls if row["event"] == "workspace_refusal"]
                alternatives = {row["implementation"] for row in report["executions"] if row["node"] == "decoder"}
                if len(refused) != 1 or refused[0]["rank"] != 1 or len(alternatives) < 2:
                    failure = failure or RuntimeError("rank-local refusal did not exercise the alternate execution")
                else:
                    peer_release = re.findall(
                        r"Execution agreement: node=decoder stage=workspace implementation=" +
                        re.escape(refused[0]["implementation"]) +
                        r" outcome=refused released=(True|False) refused_ranks=\[1\]", log)
                    if sorted(peer_release) != ["False", "True"] or any(
                        not sequence or sequence[0]["implementation"] == refused[0]["implementation"]
                        for sequence in decoder_sequences
                    ):
                        failure = failure or RuntimeError("peer rollback and common alternate selection were not recorded")
        elif not args.host_test:
            releases = [json.loads(line.split("PI_RUNTIME_RELEASED ", 1)[1])
                        for line in log.splitlines() if "PI_RUNTIME_RELEASED " in line]
            report["native_release"] = releases
            if not releases or any(not health["closed"] or health["live_storages"] or
                                   health["retirement_failures"] or health["poisoned"] for health in releases):
                failure = failure or RuntimeError("native shutdown did not establish clean storage retirement")
        report["passed"] = failure is None
        save(output / "result.json", report)
        if context is not None:
            context.close()
        if lock is not None:
            lock.close()
    if failure is not None:
        raise failure
    print(json.dumps({"gate": report["gate"], "passed": True, "host_test": args.host_test,
                      "cases": len(report["cases"]), "report": str(output / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
