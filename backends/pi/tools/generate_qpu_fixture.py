#!/usr/bin/env python3
"""Reassemble pinned qualification kernels; not a serving dependency."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembler-root", type=Path, required=True,
                        help="py-videocore7 checkout at 4707e67fcba62217293a5bb155bfd594d97d8eb3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("r2", "copy"), default="r2")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "tests/fixtures/qpu" / (
        "source.py" if args.kind == "r2" else "copy_source.py")
    assembler_file = args.assembler_root / "src/_videocore7/assembler.py"
    if not assembler_file.is_file():
        parser.error("--assembler-root must contain src/_videocore7/assembler.py")
    assembler_hash = hashlib.sha256(assembler_file.read_bytes()).hexdigest()
    if assembler_hash != "3b34b28a80e0c2f3e3b5c27067e4f9a0c4ccde7470e4110e0eedfc1a2b2de279":
        parser.error("assembler bytes do not match the pinned witness revision")
    sys.path.insert(0, str(args.assembler_root.resolve() / "src"))
    from videocore7.assembler import assemble
    spec = importlib.util.spec_from_file_location("r2_qpu_fixture", source)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    if args.kind == "r2":
        programs = {
            "producer": assemble(module.qpu_tmu_single_write),
            "consumer": assemble(module.qpu_tmu_single_read),
        }
    else:
        programs = {"copy_words": assemble(module.qpu_copy_words)}
    result = {
        **programs,
        "domain": {"vector_width": 16, "iterations_min": 1, "iterations_max": 4096},
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "assembler_revision": "4707e67fcba62217293a5bb155bfd594d97d8eb3",
        "assembler_sha256": assembler_hash,
        "program_sha256": {
            name: hashlib.sha256(b"".join(struct.pack("<Q", word) for word in words)).hexdigest()
            for name, words in programs.items()
        },
    }
    if args.kind == "copy":
        result["program_sha256"] = result["program_sha256"]["copy_words"]
        result["domain"]["overlap"] = "identical or disjoint byte ranges"
        result["hardware_observed"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["program_sha256"], sort_keys=True))


if __name__ == "__main__":
    main()
