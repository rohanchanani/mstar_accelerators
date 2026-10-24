"""Run a registry through an optimization-platform qualification driver.

The driver owns its test environment and executable-reference adapters; the
registry-consumption loop and numerical policies are ordinary reusable code.
A new backend or non-differential consumer needs no M* serving change.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def load_symbol(spec: str):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def exit_status(report: dict, *, require_all_pass: bool = False) -> int:
    if report["status"] == "FAIL":
        return 1
    if require_all_pass and (report["status"] != "PASS" or not report["complete"]):
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="mstar_accelerators.catalog:accelerations")
    parser.add_argument("--driver", default="mstar_accelerators.testing.pi:run")
    parser.add_argument("--config", help="driver-owned JSON options; omitted means discovery without model assets")
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-all-pass", action="store_true",
                        help="make missing coverage nonzero for this requested gate")
    args = parser.parse_args(argv)
    options = {} if args.config is None else json.loads(Path(args.config).read_text())
    report = load_symbol(args.driver)(args.registry, options)
    report.update(registry=args.registry, driver=args.driver)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return exit_status(report, require_all_pass=args.require_all_pass)


if __name__ == "__main__":
    raise SystemExit(main())
