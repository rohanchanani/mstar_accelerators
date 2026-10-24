"""Separately gated postacceptance containment test; intentionally does not exit.

Run ONLY after preserving all normal-path receipts and arranging the supervised
candidate-boot recovery procedure. A real ggml submission occurs. A test-only
wrapper retains its genuine receipt but gives Runtime an invalid receipt.
Runtime must close admission and retain the complete invocation. A physically
finished healthy kernel does not restore a context whose completion contract was
broken. No hung shader, reset, cancellation, or failed kernel fence is induced.
"""

from __future__ import annotations

if not __debug__:
    raise SystemExit("containment oracles require assertions; remove -O and PYTHONOPTIMIZE")

import argparse
import json
import os
from pathlib import Path
import traceback

import torch
import mstar_pi

from consumer_chain import hold_uncertain_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shader", required=True)
    parser.add_argument("--render-node", default="/dev/dri/renderD128")
    parser.add_argument("--wait-timeout-ms", type=int, default=30_000)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--i-understand-this-requires-supervised-recovery", action="store_true")
    args = parser.parse_args()
    if not args.i_understand_this_requires_supervised_recovery:
        parser.error("this gated test quarantines a real accepted invocation; supervised recovery is required")

    mstar_pi.initialize(copy_shader=Path(args.shader).resolve(strict=True), render_node=args.render_node,
                        wait_timeout_ms=args.wait_timeout_ms, validation=args.validation)
    report = {
        "test": "ggml_missing_receipt", "pid": os.getpid(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "scope": "native postacceptance receipt containment; no device/kernel fault induced",
    }
    try:
        with torch.inference_mode():
            stream = mstar_pi.Stream()
            with stream:
                weights = torch.eye(384, dtype=torch.float32).to("mstar_pi", non_blocking=True)
                program = mstar_pi.prepare_linear(weights, token_counts=[1], slots=1)
                x = torch.arange(384, dtype=torch.float32).view(1, 384).to("mstar_pi", non_blocking=True)
                out = torch.empty_like(x)
                print(json.dumps({**report, "status": "ARMED", "health": mstar_pi.health()}, sort_keys=True),
                      flush=True)
                try:
                    program._test_submit_missing_receipt(
                        x, out, acknowledge="quarantine until supervised recovery")
                except (RuntimeError, mstar_pi.FatalError) as error:
                    report["submission_error"] = str(error)
                else:
                    raise AssertionError("invalid postacceptance receipt unexpectedly succeeded")
                health = mstar_pi.health()
                assert health["poisoned"], health
                assert program.info()["active_slots"] == 1
                rejected = []
                for name, operation in (
                    ("clone", out.clone),
                    ("new_native_program", lambda: mstar_pi.prepare_csd([0])),
                    ("provider_resubmit", lambda: program.submit(x, out)),
                ):
                    try:
                        operation()
                    except (RuntimeError, mstar_pi.FatalError, mstar_pi.Rejected):
                        rejected.append(name)
                    else:
                        raise AssertionError(f"{name} admitted new work after fatal receipt failure")
                report.update({
                    "status": "EXPECTED_FATAL_HELD", "health": mstar_pi.health(),
                    "admission_rejected": rejected, "program": program.info(),
                    "ordinary_shutdown_attempted": False, "normal_exit_is_not_recovery": True,
                })
                # All local tensors and program objects also remain in scope.
                # The authoritative native quarantine independently owns them.
                hold_uncertain_context(report)
    except BaseException as error:
        report.update({"status": "FAIL", "error": repr(error),
                       "traceback": traceback.format_exc(), "health": mstar_pi.health()})
        if report["health"]["poisoned"]:
            hold_uncertain_context(report)
        try:
            mstar_pi.synchronize()
        except BaseException as drain_error:
            report["drain_error"] = repr(drain_error)
            report["health"] = mstar_pi.health()
            hold_uncertain_context(report)
        print(json.dumps({**report, "accepted_work_settled": True}, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
