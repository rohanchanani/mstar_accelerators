#!/usr/bin/env python3
"""Execute AMD target gates and the real demo from an installed environment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent


def save(path, record):
    path.write_text(json.dumps(record, indent=2) + "\n")


def run_stage(name, command, output, record, *, cwd):
    """A stage cannot hand control to its successor after a failed subprocess."""
    log_path = output / f"{name}.log"
    stage = {"name": name, "command": command, "log": str(log_path)}
    record["stages"].append(stage)
    save(output / "foundation.json", record)
    print(f"[{name}] {log_path}", flush=True)
    start = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        stage["returncode"] = process.wait()
    stage["seconds"] = time.monotonic() - start
    save(output / "foundation.json", record)
    if stage["returncode"]:
        raise RuntimeError(f"{name} failed with exit {stage['returncode']}; see {log_path}")


def check_qualification_report(report):
    if (report.get("status") != "PASS" or report.get("complete") is not True
            or not report.get("accelerations")
            or any(row.get("status") != "PASS" for row in report["accelerations"])):
        raise RuntimeError("central qualification must report complete PASS for every selected acceleration")
    return {"status": "PASS", "complete": True, "counts": report["counts"]}


def qualification_options(deployment):
    if deployment.get("runtime_extension", {}).get("nodes") != ["Talker"]:
        raise ValueError("the AMD campaign requires the external Talker deployment")
    return {"checkpoint": deployment["model_kwargs"]["model_path_hf"],
            "device": "cuda:0", "threads": 4}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mstar-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="reuse and verify the pinned local snapshot")
    parser.add_argument("--offline", action="store_true", help="use existing checkpoint/cache files only")
    parser.add_argument("--output", type=Path, help="fresh run directory; default: WORK/runs/<UTC timestamp>")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=5400)
    parser.add_argument("--shutdown-timeout", type=float, default=120)
    args = parser.parse_args(argv)
    work = args.work_dir.resolve()
    output = (args.output or work / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    mstar = args.mstar_root.resolve()
    record = {"status": "FAIL", "scope": "AMD target preflight, central qualification and real service baseline",
              "python": sys.executable, "mstar_root": str(mstar), "accelerators_root": str(ROOT),
              "output": str(output), "stages": []}
    preflight = output / "preflight.json"
    qualification = output / "qualification.json"
    deployment = output / "qwen3_omni_amd.yaml"
    try:
        # Refuse an unavailable target before downloading or hashing the large checkpoint.
        run_stage("preflight", [sys.executable, str(TOOLS / "preflight.py"),
                  "--mstar-root", str(mstar), "--report", str(preflight)], output, record, cwd=ROOT)
        preflight_result = json.loads(preflight.read_text())
        if preflight_result.get("status") != "PASS" or not preflight_result.get("kernels_executed"):
            raise RuntimeError("target preflight did not execute its required backend tests")
        assets = [sys.executable, str(TOOLS / "prepare_assets.py"), "--config", str(deployment)]
        if args.checkpoint:
            assets += ["--checkpoint", str(args.checkpoint.resolve())]
        else:
            assets += ["--cache-dir", str(work / "hf")]
        if args.offline:
            assets.append("--offline")
        run_stage("assets", assets, output, record, cwd=ROOT)
        config = yaml.safe_load(deployment.read_text())
        options = qualification_options(config) | {"deployment": str(deployment)}
        options_path = output / "qualification-config.json"
        save(options_path, options)
        registry = config["runtime_extension"]["accelerations"]
        run_stage("qualification", [sys.executable, "-m", "mstar_accelerators.testing",
                  "--registry", registry, "--driver", "mstar_accelerators.testing.qwen3_omni:run",
                  "--config", str(options_path), "--report", str(qualification), "--require-all-pass"],
                  output, record, cwd=ROOT)
        record["qualification"] = check_qualification_report(json.loads(qualification.read_text()))
        record["qualification"]["report"] = str(qualification)
        record["qualification"]["sha256"] = hashlib.sha256(qualification.read_bytes()).hexdigest()
        run_stage("demo", [sys.executable, str(TOOLS / "run_demo.py"), "--deployment", str(deployment),
                  "--mstar-root", str(mstar), "--output", str(output / "demo"), "--port", str(args.port),
                  "--startup-timeout", str(args.startup_timeout), "--shutdown-timeout", str(args.shutdown_timeout)],
                  output, record, cwd=ROOT)
        demo = json.loads((output / "demo/run.json").read_text())
        if demo.get("status") != "PASS":
            raise RuntimeError("service demo did not produce a passing receipt")
        record.update(status="PASS", speech=demo["speech"],
                      execution_receipts=demo["execution_receipts"], worker_release=demo["worker_release"])
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        # An unavailable target/reference is missing evidence, not a numerical failure.
        for report_path in (preflight, qualification):
            if report_path.exists():
                try:
                    result = json.loads(report_path.read_text())
                except (OSError, ValueError):
                    continue
                if result.get("status") in ("BLOCKED", "SKIP"):
                    record["status"] = "BLOCKED"
    finally:
        save(output / "foundation.json", record)
    print(json.dumps({"status": record["status"], "report": str(output / "foundation.json")}), flush=True)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
