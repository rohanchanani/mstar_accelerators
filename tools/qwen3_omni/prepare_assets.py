#!/usr/bin/env python3
"""Fetch the pinned Qwen checkpoint, verify once, and render absolute paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "requirements/qwen3-omni-assets.json"


def verify_file(path: Path, record: dict) -> dict:
    size = path.stat().st_size
    if size != record["bytes"]:
        raise ValueError(f"wrong size for {path}: {size} != {record['bytes']}")
    if "sha256" in record:
        algorithm, expected = "sha256", record["sha256"]
        digest = hashlib.sha256()
    else:
        algorithm, expected = "git_blob_sha1", record["git_blob_sha1"]
        digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"{algorithm} mismatch for {path}")
    return {"path": str(path.resolve()), "bytes": size, algorithm: expected}


def render_deployment(template: Path, checkpoint: Path, destination: Path) -> dict:
    from mstar_accelerators.sources import normalize_source

    config = yaml.safe_load(template.read_text())
    config.setdefault("model_kwargs", {})["model_path_hf"] = str(checkpoint.resolve())
    extension = config["runtime_extension"]
    extension["accelerations"] = normalize_source(extension["accelerations"], relative_to=ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache, reusing existing shards")
    parser.add_argument("--checkpoint", type=Path, help="verify an existing local checkpoint instead")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=ROOT / "deployments/qwen3_omni_amd_a.yaml")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
    else:
        from huggingface_hub import snapshot_download

        checkpoint = Path(snapshot_download(
            repo_id=manifest["model"], revision=manifest["revision"],
            allow_patterns=[row["path"] for row in manifest["files"]],
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=args.offline,
        ))
    started = time.monotonic()
    records = []
    for row in manifest["files"]:
        print(f"Verifying {row['path']}", flush=True)
        records.append(verify_file(checkpoint / row["path"], row))
    render_deployment(args.template, checkpoint, args.config)
    receipt = {
        "status": "PASS", "scope": "checkpoint bytes and rendered deployment, no model execution",
        "model": manifest["model"], "revision": manifest["revision"],
        "checkpoint": str(checkpoint.resolve()), "config": str(args.config.resolve()),
        "files": records, "verification_seconds": time.monotonic() - started,
    }
    receipt_path = args.config.with_suffix(".assets.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"config": str(args.config.resolve()), "receipt": str(receipt_path.resolve())}))


if __name__ == "__main__":
    main()
