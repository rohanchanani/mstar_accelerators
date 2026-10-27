"""Service-client mechanics, not acceleration accuracy qualification."""
import json

import pytest

from tools.whisper.demo import transcribe


class Response:
    status_code = 200

    def __init__(self, text, events):
        self.text, self.events = text, events

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def json(self):
        return {"text": self.text}

    def iter_lines(self, **_):
        yield from self.events


def request(monkeypatch, tmp_path, text, *, expected="fixture reference", events=(), **kwargs):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"client fixture")
    monkeypatch.setattr("tools.whisper.demo.requests.post", lambda *a, **k: Response(text, events))
    return transcribe("http://localhost", audio, expected, tmp_path, "case", 1, **kwargs)


def stream(done="generated text", terminator=True):
    events = ["data: " + json.dumps({"type": "transcript.text.delta", "delta": " generated text"}),
              "data: " + json.dumps({"type": "transcript.text.done", "text": done})]
    return events + (["data: [DONE]"] if terminator else [])


@pytest.mark.parametrize("streaming", [False, True])
def test_measurement_retains_transcript_without_accuracy_verdict(monkeypatch, tmp_path, streaming):
    row = request(monkeypatch, tmp_path, "generated text", stream=streaming, events=stream())
    assert row["passed"] and row["accuracy_qualification"] == "not_run"
    assert row["reference_transcript"] == "fixture reference"
    assert "repeatability_matched" not in row


def test_seeded_replay_still_checks_exact_text(monkeypatch, tmp_path):
    with pytest.raises(AssertionError, match="seeded transcript replay"):
        request(monkeypatch, tmp_path, "different", exact_text=True, seed=12)
    assert request(monkeypatch, tmp_path, "same", expected="same", exact_text=True, seed=12)["repeatability_matched"]


@pytest.mark.parametrize("events", [stream(done="inconsistent"), stream(terminator=False)])
def test_stream_integrity_remains_required(monkeypatch, tmp_path, events):
    with pytest.raises(AssertionError, match="incomplete or inconsistent"):
        request(monkeypatch, tmp_path, "unused", stream=True, events=events)


def test_cancellation_checks_live_prefix_not_reference_accuracy(monkeypatch, tmp_path):
    row = request(monkeypatch, tmp_path, "unused", stream=True, cancel=True, events=stream())
    assert row["cancelled_prefix"] == " generated text"
    with pytest.raises(AssertionError, match="seeded transcript replay"):
        request(monkeypatch, tmp_path, "unused", stream=True, cancel=True, events=stream(), exact_text=True)
