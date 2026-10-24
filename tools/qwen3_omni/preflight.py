#!/usr/bin/env python3
"""Check the pinned AMD environment and run focused backend conformance tests."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {
    "torch": "2.12.1+rocm7.2", "torchvision": "0.27.1+rocm7.2",
    "torchaudio": "2.11.0+rocm7.2", "triton-rocm": "3.7.1",
    "pybind11": "3.0.4", "flydsl": "0.3.4.1", "transformers": "5.10.2",
    "numpy": "2.2.6", "amd-flashinfer": "0.6.18+amd.1",
}
# Backend and generic capture-adapter mechanisms, independent of an acceleration's verdict.
KERNEL_TESTS = [
    "test/rocm/test_attention_aiter.py::test_capture_once_replay_new_lengths",
    "test/rocm/test_attention_aiter.py::test_preplanned_slot_replays_while_the_live_one_is_untouched",
    "test/rocm/test_attention_aiter.py::test_dense_aiter_varlen_aligns_causal_bottom_right",
    "test/rocm/test_norm_kernels.py::test_forward_residual_under_graph_capture",
    "test/rocm/test_rope.py::test_rope_under_graph_capture",
    "test/rocm/test_sampler.py::test_captured_sampler_replays_the_eager_draw",
    "test/rocm/test_sampler.py::test_draws_repeat_per_seed_and_offset",
    "test/rocm/test_fused_moe.py::test_backend_capture_replay",
    "test/rocm/test_skinny_gemm.py::test_captured_bucket_replays",
    "test/rocm/test_triton_kernels.py::test_sampling_prep_kernels",
    "test/rocm/test_triton_kernels.py::test_attention_helpers",
    "test/modular/test_tensor_forward_execution.py::test_separate_program_real_engine_and_stable_logical_outputs[gpu]",
]


class EnvironmentUnavailable(RuntimeError):
    """The requested target cannot execute here; this is not model qualification."""


def check_source_imports(imports: dict, mstar_root: Path) -> None:
    for name, root in (("mstar", mstar_root), ("mstar_accelerators", ROOT)):
        if not Path(imports[name]).is_relative_to(root.resolve()):
            raise RuntimeError(f"wrong {name} import: {imports[name]}")


def check_kernel_report(path: Path) -> dict:
    cases = ET.parse(path).getroot().findall(".//testcase")
    skipped = [case.attrib.get("name") for case in cases if case.find("skipped") is not None]
    failed = [case.attrib.get("name") for case in cases
              if case.find("failure") is not None or case.find("error") is not None]
    if not cases or skipped or failed:
        raise RuntimeError(f"backend preflight must execute every selected case: {len(cases)} cases, "
                           f"skipped={skipped}, failed={failed}")
    return {"passed": len(cases), "skipped": 0, "failed": 0}


def inspect_environment(mstar_root: Path) -> dict:
    import torch
    from mstar.platforms import current_platform

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("the pinned ROCm environment requires Linux x86_64")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("the pinned AITER wheel requires Python 3.12")
    if not current_platform.is_rocm() or not torch.cuda.is_available():
        raise EnvironmentUnavailable("no usable ROCm GPU; target preflight cannot execute on this host")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    differences = {name: {"expected": expected, "actual": versions[name]}
                   for name, expected in VERSIONS.items() if versions[name] != expected}
    if differences:
        raise RuntimeError(f"environment differs from the pinned recipe: {differences}")
    aiter = importlib.metadata.version("amd-aiter")
    if aiter.split("+")[0] != "0.1.25":
        raise RuntimeError(f"expected amd-aiter 0.1.25, found {aiter}")
    versions["amd-aiter"] = aiter
    properties = torch.cuda.get_device_properties(0)
    arch = current_platform.arch()
    if arch != "gfx942":
        raise RuntimeError(f"this deployment is prepared for gfx942, found {arch}")
    if properties.total_memory < 90 * 1024**3:
        raise RuntimeError("the colocated checkpoint needs an MI300X/MI325X-class memory budget")
    imports = {}
    for name in ("torchvision", "torchaudio", "mstar", "mstar_accelerators", "mstar_rust", "aiter", "flashinfer.rocm"):
        module = importlib.import_module(name)
        imports[name] = str(Path(module.__file__).resolve())
    check_source_imports(imports, mstar_root)
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        raise RuntimeError("hipcc is unavailable; source kernels cannot compile")
    if os.environ.get("MIOPEN_FIND_MODE") == "FAST":
        raise RuntimeError("unset MIOPEN_FIND_MODE=FAST; it changes the donor's vocoder tuning policy")
    if os.environ.get("AITER_JIT_DIR") and not list(Path(os.environ["AITER_JIT_DIR"]).glob("*.so")):
        raise RuntimeError("AITER_JIT_DIR hides the prebuilt wheel kernels; unset it or seed the .so files")
    # Exercise a real allocation, HIP GEMM, and synchronization before the JIT tests.
    x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    if not torch.isfinite(y).all().item():
        raise RuntimeError("non-finite result from the initial HIP GEMM")
    return {"versions": versions, "imports": imports, "hipcc": hipcc,
            "gpu": properties.name, "arch": arch, "memory_bytes": properties.total_memory,
            "torch_hip": torch.version.hip}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mstar-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--imports-only", action="store_true", help="defer kernel tests; report is environment-only")
    args = parser.parse_args(argv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "FAIL", "scope": "AMD environment and backend conformance; no model quality claim",
              "kernels_executed": False}
    try:
        report.update(inspect_environment(args.mstar_root))
        if not args.imports_only:
            kernel_report = args.report.with_suffix(".kernels.xml").resolve()
            command = [sys.executable, "-m", "pytest", "-q", *KERNEL_TESTS,
                       "--junitxml=" + str(kernel_report)]
            report["kernel_command"] = command
            subprocess.run(command, cwd=args.mstar_root, check=True)
            report["kernel_results"] = check_kernel_report(kernel_report)
        report["status"] = "PASS"
        report["kernels_executed"] = not args.imports_only
    except EnvironmentUnavailable as error:
        report.update(status="BLOCKED", error=f"{type(error).__name__}: {error}")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
