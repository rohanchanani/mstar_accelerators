#!/usr/bin/env python3
"""Verify exact patch application and run extracted-function branch tests.

This is intentionally not called a kernel build or a hardware test. It executes
selected unmodified bodies from the patched source with deterministic kernel
boundary substitutes. Full-kernel/KASAN/hardware gates are in ../README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
KERNEL = HERE.parent


def function(text: str, name: str) -> str:
    """Extract a file-scope function; kernel style keeps its } at column 0."""
    m = re.search(r"(?m)^(?:static )?(?:struct dma_fence \*|int|void)\s*\n?" +
                  re.escape(name) + r"\([^;]*?\n\{", text)
    if not m:
        raise AssertionError(f"function not found: {name}")
    end = text.index("\n}", m.end()) + 2
    return text[m.start():end] + "\n"


def check_structure(root: Path) -> int:
    submit = (root / "drivers/gpu/drm/v3d/v3d_submit.c").read_text()
    sched = (root / "drivers/gpu/drm/v3d/v3d_sched.c").read_text()
    drv = (root / "drivers/gpu/drm/v3d/v3d_drv.c").read_text()
    irq = (root / "drivers/gpu/drm/v3d/v3d_irq.c").read_text()
    checks = 0
    for name, first_push, reservations in (
        ("v3d_submit_csd_ioctl", "v3d_push_job(&job->base);", 1),
        ("v3d_submit_cpu_ioctl", "v3d_push_job(&cpu_job->base);", 2),
    ):
        body = function(submit, name)
        before, accepted = body.split(first_push, 1)
        accepted = accepted[:accepted.index("\n\treturn 0;")]
        assert before.count("v3d_reserve_chain_dependency(") == reservations
        assert "drm_sched_job_add_dependency(" not in accepted
        assert "goto fail" not in accepted
        checks += 3
    for name in ("v3d_submit_cl_ioctl", "v3d_submit_tfu_ioctl",
                 "v3d_submit_csd_ioctl", "v3d_submit_cpu_ioctl"):
        body = function(submit, name)
        assert body.count("READ_ONCE(v3d->csd_error)") >= 2
        checks += 1
    init = function(submit, "v3d_job_init")
    free = function(submit, "v3d_job_free")
    for take, release in (("__module_get(THIS_MODULE)", "module_put(THIS_MODULE)"),
                          ("drm_dev_get", "drm_dev_put"),
                          ("get_device", "put_device")):
        assert take in init and release in free
        checks += 1
    assert ".suppress_bind_attrs = true" in drv
    assert "v3d_file_priv_put(v3d_priv)" in function(drv, "v3d_postclose")
    assert "v3d_perfmon_close_file(v3d_priv)" in function(drv, "v3d_file_priv_free")
    assert "kref_get(&v3d_priv->refcount)" in init and "v3d_file_priv_put(job->file_priv)" in free
    assert "v3d_csd_poison_device(v3d, -EFAULT)" in irq
    checks += 5
    reset_start = sched.index("v3d_gpu_reset_for_timeout(")
    reset_end = sched.index("\n}", reset_start)
    reset = sched[reset_start:reset_end]
    assert reset.index("v3d_job_quarantine(job, -ETIMEDOUT)") < reset.index("v3d_reset(v3d)")
    assert "v3d_priv = job->file_priv" not in reset
    checks += 2
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    manifest = json.loads((KERNEL / "source_manifest.json").read_text())
    result = {"source_package": manifest["source_package"], "full_kernel_build": False,
              "hardware_test": False}
    with tempfile.TemporaryDirectory(prefix="v3d-contract-") as temp:
        work = Path(temp)
        for relative, hashes in manifest["files"].items():
            data = (source / relative).read_bytes()
            assert hashlib.sha256(data).hexdigest() == hashes["before_sha256"], relative
            destination = work / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i",
                        str(KERNEL / "0001-v3d-csd-completion-contract.patch")],
                       cwd=work, check=True)
        for relative, hashes in manifest["files"].items():
            assert hashlib.sha256((work / relative).read_bytes()).hexdigest() == hashes["after_sha256"], relative
        result["exact_source_files"] = len(manifest["files"])
        result["structural_checks"] = check_structure(work)
        bodies = []
        for relative, names in (
            ("drivers/gpu/drm/v3d/v3d_submit.c", ("v3d_reserve_chain_dependency", "v3d_set_chain_dependency", "v3d_submit_csd_ioctl")),
            ("drivers/gpu/drm/v3d/v3d_gem.c", ("v3d_clean_caches",)),
            ("drivers/gpu/drm/v3d/v3d_sched.c", ("v3d_csd_poison_device", "v3d_job_quarantine", "v3d_job_error", "v3d_sched_job_free", "v3d_csd_job_run", "v3d_cache_clean_job_run")),
        ):
            text = (work / relative).read_text()
            for name in names:
                bodies.append(function(text, name))
        generated = work / "extracted.inc"
        generated.write_text("\n".join(bodies))
        exe = work / "failure_branches"
        subprocess.run([args.cc, "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                        "-Wno-unused-parameter", "-O1", "-g", "-fsanitize=undefined,address",
                        "-I", str(work), str(HERE / "failure_branches.c"), "-o", str(exe)], check=True)
        test_env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
        tested = subprocess.run([str(exe)], capture_output=True, text=True, env=test_env)
        print(tested.stdout, end="")
        if tested.returncode:
            raise RuntimeError(tested.stderr)
        result["extracted_function_tests"] = json.loads(tested.stdout)
        result["sanitizers"] = ["address (leak detection disabled for sandbox)", "undefined"]
    result["status"] = "PASS (source and host branch tests only)"
    report = json.dumps(result, indent=2) + "\n"
    print(report, end="")
    if args.report:
        args.report.write_text(report)


if __name__ == "__main__":
    main()
