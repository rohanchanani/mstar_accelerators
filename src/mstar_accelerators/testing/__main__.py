"""Run the platform-owned qualification policy using M* execution machinery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .environments import run


def exit_status(report: dict, *, require_all_pass: bool = False) -> int:
    if report["status"] == "FAIL":
        return 1
    if require_all_pass and (report["status"] != "PASS" or not report["complete"]):
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="mstar_accelerators.catalog:accelerations")
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-all-pass", action="store_true",
                        help="make missing coverage nonzero for this requested gate")
    args = parser.parse_args(argv)
    report = run(args.registry)
    report.update(registry=args.registry)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return exit_status(report, require_all_pass=args.require_all_pass)


if __name__ == "__main__":
    raise SystemExit(main())
