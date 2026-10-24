"""Reevaluate stored encoder observations under named qualification policies.

This consumes evidence, not an acceleration. It never initializes a device or
submits native work. A fresh target harness run is a separate qualification step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mstar_accelerators.testing.numerics import DEFAULT, LEGACY_POINTWISE, compare


def replay(paths: list[Path]) -> dict:
    cases = []
    for path in paths:
        path = Path(path).resolve()
        capture = torch.load(path, map_location="cpu", weights_only=True)
        native = capture["native"]
        reference32 = capture["reference_f32"]
        reference64 = capture["reference_f64"]
        if reference32.dtype != torch.float32 or reference64.dtype != torch.float64:
            raise ValueError(f"{path}: expected distinct FP32 and FP64 reference captures")
        cases.append({
            "capture": str(path),
            "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "current": compare(path.stem, native, reference64, dtype=torch.float32),
            "legacy": compare(path.stem, native, reference32, dtype=torch.float32,
                              policy=LEGACY_POINTWISE),
            "reference32_against_reference64": compare(
                path.stem, reference32, reference64, dtype=torch.float32),
        })
    passed = bool(cases) and all(case["current"]["passed"] for case in cases)
    return {
        "scope": "stored native capture numerical reevaluation; no new native execution",
        "status": "PASS" if passed else "FAIL",
        "policy": DEFAULT.name,
        "reference": "captured CPU HF FP64 on the same logical FP32 inputs and weights",
        "legacy_policy": LEGACY_POINTWISE.name,
        "legacy_status": "PASS" if cases and all(case["legacy"]["passed"] for case in cases) else "FAIL",
        "historical_verdicts_changed": False,
        "fresh_acceleration_execution": False,
        "cases": cases,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path,
                        help="original full-comparison.pt / tail-comparison.pt captures")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    report = replay(args.captures)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.report.write_text(text)
    print(text, end="", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
