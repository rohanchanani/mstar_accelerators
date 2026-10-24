#!/usr/bin/env python3
"""Fetch the pinned checkpoint, verify its bytes, and render the deployment.

This is asset preparation. It does not initialize the Pi or build a reference
compute stack. Checkpoints and generated deployments live outside source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/fixtures/whisper/assets.json"


def verify(path, record):
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
        raise ValueError(f"asset identity mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deployment-template", type=Path,
                        default=ROOT / "deployments/whisper_tiny_pi.yaml")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    checkpoint = args.destination.resolve() / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=manifest["model"], revision=manifest["revision"],
                          allow_patterns=[entry["path"] for entry in manifest["files"]],
                          local_dir=str(checkpoint))
    for entry in manifest["files"]:
        verify(checkpoint / entry["path"], entry)
    audio = ROOT / "tests/fixtures/whisper/jfk16k.wav"
    verify(audio, manifest["audio"]["derived"])
    verify(ROOT / "tests/fixtures/whisper/jfk_tail16k.wav", manifest["audio"]["tail"])
    from mstar_accelerators.sources import normalize_source
    config = yaml.safe_load(args.deployment_template.read_text())
    config["runtime_extension"]["accelerations"] = normalize_source(
        config["runtime_extension"]["accelerations"], relative_to=ROOT)
    config["model_kwargs"]["model_path_hf"] = str(checkpoint)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint),
                      "revision": manifest["revision"], "files_verified": len(manifest["files"]),
                      "audio": str(audio), "config": str(args.config.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
