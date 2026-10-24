"""Stored evidence can receive a new policy verdict without rewriting history."""
import hashlib
import json

import pytest
import torch

from tools.qualification.replay_encoder import main, replay


def capture(tmp_path, *, corrupt=False):
    # A sparse coordinate difference passes the complete-component RMS gate
    # while retaining a failing legacy per-coordinate comparison.
    reference = torch.ones(1024, dtype=torch.float64)
    reference[0] = 0.
    actual = reference.float()
    actual[0] = .01 if corrupt else 4e-4
    path = tmp_path / "capture.pt"
    torch.save({"native": actual, "reference_f32": reference.float(),
                "reference_f64": reference}, path)
    return path


def test_replay_retains_distinct_legacy_verdict_and_capture_identity(tmp_path):
    path = capture(tmp_path)
    raw = path.read_bytes()
    report = replay([path])
    assert report["status"] == "PASS"
    assert report["legacy_status"] == "FAIL"
    assert report["historical_verdicts_changed"] is False
    assert report["fresh_acceleration_execution"] is False
    assert report["cases"][0]["capture_sha256"] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw


def test_corrupt_output_fails_replay_and_cli_exit(tmp_path):
    path = capture(tmp_path, corrupt=True)
    output = tmp_path / "report.json"
    assert main([str(path), "--report", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "FAIL"
    assert not report["cases"][0]["current"]["passed"]


def test_replay_does_not_fabricate_high_precision_reference(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"native": torch.zeros(4), "reference_f32": torch.zeros(4),
                "reference_f64": torch.zeros(4)}, path)
    with pytest.raises(ValueError, match="distinct FP32 and FP64"):
        replay([path])


def test_empty_replay_is_not_pass():
    assert replay([])["status"] == "FAIL"
