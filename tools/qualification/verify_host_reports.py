#!/usr/bin/env python3
"""Verify generated-file host receipts and the source-owned numerical policy.

This checks report completeness and identity, not the underlying tensor values.
Diagnostic FAILs are retained and never become acceptance requirements.
Declaration identities come from the selected source corpus. Program retirement
is checked against the programs actually prepared by the host engine.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_policy(source):
    from mstar_accelerators.testing import numerics
    require(Path(numerics.__file__).resolve() ==
            (source / "src/mstar_accelerators/testing/numerics.py").resolve(),
            "policy import differs from selected source")
    return numerics.DEFAULT


def check_policy(record, policy, dtype, reference_dtype, *, required):
    name = record["name"]
    require(record["policy"] == policy.name, f"{name}: wrong policy")
    require(record["actual_dtype"] == record["required_dtype"] == str(dtype),
            f"{name}: wrong output dtype")
    require(record["reference_dtype"] == str(reference_dtype), f"{name}: wrong reference dtype")
    require(type(record["passed"]) is bool, f"{name}: missing boolean verdict")
    if required:
        require(record["passed"] is True, f"{name}: required observation failed")


def verify_case(source, path, fixture, oracle, default, declarations):
    report = json.loads(path.read_text())
    require(report["gate"] == "generated_files_engine_algebra_v1", f"{fixture}: wrong gate")
    require(report["passed"] is True, f"{fixture}: gate did not pass")
    fixture_path = source / "tests/fixtures/whisper" / fixture
    require(hashlib.sha256(fixture_path.read_bytes()).hexdigest() == oracle["audio_sha256"],
            f"{fixture}: source audio differs from frozen reference")
    require(report["audio"]["sha256"] == oracle["audio_sha256"], f"{fixture}: wrong input audio")
    require(report["text"] == oracle["text"] and report["tokens"] == oracle["tokens"],
            f"{fixture}: frozen text/token mismatch")
    count = len(oracle["tokens"])
    expected_count = {"jfk16k.wav": 24, "jfk_tail16k.wav": 17}[fixture]
    require(count == expected_count, f"{fixture}: changed frozen trajectory length")
    require(report["declarations"] == declarations,
            f"{fixture}: registrations differ from the selected source corpus")
    require(report["programs_prepared"] > 0 and report["programs_closed"] == report["programs_prepared"]
            and report["resource_pages_released"] is True,
            f"{fixture}: missing program/resource release")
    require(oracle["stopped_by"] == "eos" and oracle["generated_tokens"] == count
            and report["generated_tokens"] == count, f"{fixture}: incomplete EOS trajectory")
    require(report["encoder_trajectory"] ==
            "unchanged CPU FP32 graph output; no FP64/reference substitution",
            f"{fixture}: changed encoder trajectory")
    names = ["encoder.algebra_fp64"] + [
        f"{'prefill' if step == 0 else 'decode'}.{step}.{kind}"
        for step in range(count) for kind in ("logits", "kv_cache", "cross_kv_cache")]
    observations = report["observations"]
    require([record["name"] for record in observations] == names,
            f"{fixture}: required observations missing, duplicated or reordered")
    for index, record in enumerate(observations):
        dtype = torch.float64 if index == 0 else torch.float32
        check_policy(record, default, dtype, dtype, required=True)
    diagnostics = {}
    for field, name, policy, reference_dtype in (
        ("cpu_fp32_encoder_diagnostic", "encoder.cpu_fp32", default, torch.float32),
        ("composed_internal_cache_diagnostic", "different_boundary.cross_cache", default, torch.float32),
    ):
        record = report[field]
        require(record["name"] == name, f"{fixture}: wrong {field}")
        check_policy(record, policy, torch.float32, reference_dtype, required=False)
        diagnostics[field] = {"passed": record["passed"], "policy": record["policy"]}
    return {"fixture": fixture, "report": str(path.resolve()),
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "declarations": report["declarations"], "programs_closed": report["programs_closed"],
            "resource_pages_released": report["resource_pages_released"], "generated_tokens": count,
            "required_observations": len(observations), "diagnostics": diagnostics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True,
                        help="identity.stdout from this source's run_host_gate.py")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        source = args.source.resolve()
        from mstar_accelerators.sources import acceleration_files
        identity = json.loads(args.identity.read_text())
        files = acceleration_files(identity["acceleration_sources"])
        declarations = list(files)
        require(bool(declarations) and identity["declarations"] == declarations,
                "selected acceleration corpus is empty or changed")
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
        require(hashes == identity["acceleration_sha256"], "acceleration sources changed during the host campaign")
        default = load_policy(source)
        reference = json.loads((source / "tests/fixtures/whisper/reference.json").read_text())
        cases = [verify_case(source, path, fixture, reference["cases"][fixture], default, declarations)
                 for path, fixture in ((args.full, "jfk16k.wav"), (args.tail, "jfk_tail16k.wav"))]
        result = {"passed": True, "gate": "generated_files_engine_algebra_v1",
                  "scope": "report metadata verification; no tensor replay or native execution",
                  "acceleration_sources": identity["acceleration_sources"], "acceleration_sha256": hashes,
                  "policy_source_sha256": hashlib.sha256(
                      (source / "src/mstar_accelerators/testing/numerics.py").read_bytes()).hexdigest(),
                  "cases": cases}
    except (KeyError, ValueError, OSError) as exc:
        result = {"passed": False, "error": str(exc)}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
