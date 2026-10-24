"""Source receipts follow the selected files, including intentionally split books."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mstar
from mstar_accelerators.sources import load_accelerations
from tools.qualification import run_host_gate, verify_host_reports
from tools.whisper import demo


def test_split_source_identity_and_campaign_change_detection(tmp_path):
    corpus = tmp_path / "authored"
    corpus.mkdir()
    (corpus / "stages").mkdir()
    for name in ("encoder", "stages/prefill", "stages/contiguous_decode"):
        (corpus / (name + ".py")).write_text("acceleration = object()\n")
    source = Path(__file__).resolve().parents[1]
    mstar_source = Path(mstar.__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-B", "-c", run_host_gate.IDENTITY,
                             str(source), str(mstar_source), str(corpus)],
                            check=True, text=True, capture_output=True)
    identity = json.loads(result.stdout)
    assert identity["declarations"] == list(load_accelerations(corpus))
    assert set(identity["declarations"]) == {"encoder", "stages/prefill", "stages/contiguous_decode"}
    assert demo.acceleration_identity({"runtime_extension": {"accelerations": str(corpus)}})["sha256"] == identity["acceleration_sha256"]
    receipt = tmp_path / "identity.json"
    receipt.write_text(result.stdout)
    (corpus / "stages/contiguous_decode.py").write_text("acceleration = object()  # edited\n")
    output = tmp_path / "verification.json"
    assert verify_host_reports.main(["--source", str(source), "--identity", str(receipt),
                                     "--full", str(tmp_path / "full.json"),
                                     "--tail", str(tmp_path / "tail.json"),
                                     "--output", str(output)]) == 1
    assert json.loads(output.read_text())["error"] == "acceleration sources changed during the host campaign"


def test_demo_recognizes_the_support_consumers_policy_rejection(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    monkeypatch.setattr(demo.requests, "post", lambda *args, **kwargs: SimpleNamespace(
        status_code=400, text="request policy is unsupported by node decoder: temperature=0.5"))
    result = demo.reject_unsupported_sampling("http://unused", audio, tmp_path, 1)
    assert result["passed"]
