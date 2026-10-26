#!/usr/bin/env python3
"""One real M* startup, multimodal smoke, and c1/c8 speech baseline.

Uses M*'s ordinary service and speech benchmark. Every response and benchmark
failure is retained. The eight-prompt run is a bring-up sample, not a quality
study or a reliable tail-latency estimate.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def wait_ready(process: subprocess.Popen, url: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup: {process.returncode}")
        try:
            if requests.get(url + "/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("startup deadline exceeded; see server.log (cold ROCm compilation can be slow)")


def check_benchmark(report: dict, *, expected_concurrency=None, expected_requests=None) -> dict:
    if expected_concurrency is not None and report["concurrency"] != expected_concurrency:
        raise RuntimeError("speech benchmark concurrency differs from the requested run")
    if expected_requests is not None and report["num_sentences"] != expected_requests:
        raise RuntimeError("speech benchmark request count differs from the requested run")
    if report["errors_total"] or not report["repeats_detail"]:
        raise RuntimeError("speech benchmark contains failed requests")
    for repeat in report["repeats_detail"]:
        if (repeat["errors"] or repeat["requests"] != report["num_sentences"]
                or len(repeat["results"]) != repeat["requests"]):
            raise RuntimeError("incomplete speech repeat")
        for row in repeat["results"]:
            if row["error"] is not None or row["audio_s"] <= 0 or row["ttfa_s"] is None:
                raise RuntimeError(f"incomplete speech result: {row}")
    median = report["median_over_repeats"]
    return {"concurrency": report["concurrency"], "completed": report["num_sentences"],
            "errors": report["errors_total"], "ttfa_p50_ms": median["ttfa_p50_ms"],
            "audio_s_per_wall_s": median["audio_s_per_wall_s"], "rtf_mean": median["rtf_mean"]}


def media_smoke(url: str, mstar_root: Path, output: Path) -> list[dict]:
    from PIL import Image, ImageDraw

    image_path = output / "red-square.png"
    image = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(image).rectangle((40, 40, 184, 184), fill="red")
    image.save(image_path)
    audio_path = mstar_root / "test/qwen3-omni/audio.wav"
    image_uri = "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode()
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()
    cases = [
        ("vision", [{"type": "text", "text": "What color is the square? Answer briefly."},
                    {"type": "image_url", "image_url": {"url": image_uri}}]),
        ("audio", [{"type": "text", "text": "Briefly describe what is said in this recording."},
                   {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}]),
    ]
    results = []
    for name, content in cases:
        response = requests.post(url + "/v1/chat/completions", json={
            "model": "qwen3_omni", "messages": [{"role": "user", "content": content}],
            "temperature": 0.0, "max_tokens": 96, "seed": 17,
        }, timeout=(5, 600))
        record = {"case": name, "status_code": response.status_code, "response": response.text}
        save(output / f"{name}.json", record)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"empty {name} response")
        results.append({"case": name, "text": text, "scope": "completed media execution; inspect response for quality"})
    return results


def source_identity(path: Path) -> dict:
    path = path.resolve()

    def source_file(candidate):
        return (candidate.is_file() and not candidate.is_symlink()
                and not candidate.name.endswith((".env", ".pem", ".key"))
                and not candidate.name.startswith((".env", "id_rsa", "id_ed25519")))

    def digest(candidate):
        with candidate.open("rb") as content:
            return hashlib.file_digest(content, "sha256").hexdigest()

    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    try:
        own_checkout = Path(git("rev-parse", "--show-toplevel")).resolve() == path
        head = git("rev-parse", "HEAD") if own_checkout else None
    except (FileNotFoundError, subprocess.CalledProcessError):
        head = None
    if head is not None:
        identity = {"kind": "git", "path": str(path), "head": head,
                    "status": git("status", "--short"),
                    "tracked_diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD").encode()).hexdigest()}
        names = filter(None, git("ls-files", "--others", "--exclude-standard", "-z").split("\0"))
        identity["untracked_files_sha256"] = {name: digest(path / name) for name in names
                                              if source_file(path / name)}
    else:
        # Delivered source archives need no synthetic Git history. Exclude
        # generated build/cache trees and credential-shaped files from identity.
        excluded = {".git", ".hg", ".svn", ".ssh", ".aws", ".venv", "venv", ".cache",
                    "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist", "target"}
        files = {}
        for directory, dirs, names in os.walk(path):
            dirs[:] = sorted(name for name in dirs
                             if name not in excluded and not name.endswith(".egg-info")
                             and not (Path(directory) / name).is_symlink())
            for name in sorted(names):
                source = Path(directory) / name
                if source_file(source):
                    files[source.relative_to(path).as_posix()] = digest(source)
        ordered = dict(sorted(files.items()))
        tree = "".join(f"{name}\0{value}\n" for name, value in ordered.items())
        identity = {"kind": "source_archive", "path": str(path), "files_sha256": ordered,
                    "file_count": len(ordered), "source_tree_sha256": hashlib.sha256(tree.encode()).hexdigest(),
                    "scope": "regular source files excluding generated trees and credential-shaped names"}
        for manifest in (path / "SOURCE_MANIFEST.json", path.parent / "SOURCE_MANIFEST.json",
                         path / "MANIFEST.json", path.parent / "MANIFEST.json"):
            if manifest.is_file():
                identity["export_manifest"] = {"path": str(manifest), "sha256": digest(manifest)}
                break
    provenance = path / ".amd-a-source-provenance.json"
    if provenance.is_file():
        identity["source_snapshot_provenance"] = json.loads(provenance.read_text())
    return identity


def check_shutdown(log: str) -> dict:
    ready = set(re.findall(r"DEVICE_TORCH_RUNTIME_READY pid=(\d+)", log))
    released = set(re.findall(r"DEVICE_TORCH_RUNTIME_RELEASED pid=(\d+)", log))
    if not ready or ready != released:
        raise RuntimeError(f"worker release receipts do not match startup: ready={ready}, released={released}")
    return {"ready_pids": sorted(ready), "released_pids": sorted(released)}


def check_execution_receipts(log: str) -> list[str]:
    receipts = re.findall(r"ENGINE_EXECUTION node=Talker .*", log)
    for row in receipts:
        fields = dict(re.findall(r"(\w+)=([^\s]+)", row))
        if (fields.get("walk") == "talker_decode" and fields.get("path") == "captured"
                and fields.get("runner") == "tensor_forward" and fields.get("implementation")
                and fields.get("rows", "0").isdecimal() and int(fields.get("rows", "0")) > 0):
            return receipts
    raise RuntimeError("no captured tensor-forward Talker decode receipt; AMD deployment is incomplete")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--mstar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=5400)
    parser.add_argument("--shutdown-timeout", type=float, default=120)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.mstar_root = args.mstar_root.resolve()
    deployment = args.deployment.resolve()
    config = yaml.safe_load(deployment.read_text())
    if config.get("runtime_extension", {}).get("nodes") != ["Talker"]:
        raise ValueError("this baseline requires the explicit external Talker selection")
    # Refuse an occupied port instead of accidentally probing a different service.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    url = f"http://127.0.0.1:{args.port}"
    command = [sys.executable, "-m", "mstar.cli.main", "serve", "qwen3_omni",
               "--config", str(deployment), "--gpus", args.gpu, "--host", "127.0.0.1",
               "--port", str(args.port), "--tensor-comm-protocol", "SHM"]
    record = {"status": "FAIL", "scope": "AMD deployment service bring-up and baseline sample",
              "mstar": source_identity(args.mstar_root), "accelerators": source_identity(ROOT),
              "command": command, "deployment": config,
              "deployment_sha256": hashlib.sha256(deployment.read_bytes()).hexdigest()}
    save(args.output / "run.json", record)
    process = None
    try:
        with (args.output / "server.log").open("w") as log:
            process = subprocess.Popen(command, cwd=args.mstar_root, stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            wait_ready(process, url, args.startup_timeout)
            record["startup_seconds"] = time.monotonic() - started
            record["media"] = media_smoke(url, args.mstar_root, args.output)
            record["speech"] = []
            record["benchmark_commands"] = []
            for label, concurrency, count in (("warmup", 1, 1), ("c1", 1, 8), ("c8", 8, 8)):
                report_path = args.output / f"speech-{label}.json"
                bench = [sys.executable, "-m", "benchmark.tts_speech_bench", "--engine", "mstar",
                         "--url", url, "--model", "qwen3_omni", "--sentences",
                         str(Path(__file__).with_name("prompts.txt")), "--concurrency", str(concurrency),
                         "--num-sentences", str(count), "--repeats", "1", "--warmup", "0",
                         "--voice", "Ethan", "--seed", "17",
                         "--extra", "max_output_tokens=128", "--extra", "talker_max_output_tokens=512",
                         "--out", str(report_path), "--save-audio-dir",
                         str(args.output / f"audio-{label}"), "--timeout", "600", "--label", "A"]
                record["benchmark_commands"].append({"label": label, "command": bench})
                subprocess.run(bench, cwd=args.mstar_root, check=True)
                summary = check_benchmark(json.loads(report_path.read_text()),
                                          expected_concurrency=concurrency, expected_requests=count)
                if label == "warmup":
                    record["warmup"] = summary
                else:
                    record["speech"].append(summary)
            log.flush()
            log_text = (args.output / "server.log").read_text()
            record["execution_receipts"] = check_execution_receipts(log_text)
            if process.poll() is not None:
                raise RuntimeError(f"server exited before requested shutdown: {process.returncode}")
            record["status"] = "PASS"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=args.shutdown_timeout)
            except subprocess.TimeoutExpired:
                record["status"] = "FAIL"
                record["shutdown_error"] = f"server pid {process.pid} did not shut down; inspect before reuse"
        record["server_exit_code"] = process.poll() if process else None
        if record["status"] == "PASS":
            try:
                record["worker_release"] = check_shutdown((args.output / "server.log").read_text())
                if record["server_exit_code"] not in (0, -signal.SIGTERM):
                    raise RuntimeError(f"unexpected server exit: {record['server_exit_code']}")
            except RuntimeError as error:
                record.update(status="FAIL", shutdown_error=str(error))
        save(args.output / "run.json", record)
    print(json.dumps(record, indent=2))
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
