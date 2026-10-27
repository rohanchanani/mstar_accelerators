"""Public measurements retain their frozen bytes, split and full context."""
import hashlib
import json
import wave
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.whisper.demo import public_benchmark_inputs


def workload(tmp_path, durations=(1, 2)):
    rows = []
    for index, seconds in enumerate(durations):
        path = tmp_path / f"clip{index}.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(bytes(2 * 16000 * seconds))
        rows.append(dict(id=f"clip{index}", filename=path.name,
                         role="development" if index == 0 else "holdout",
                         transcript=f"Original transcript {index}.", duration_s=seconds,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({"files": rows}))
    return manifest


class PublicBenchmarkInputsTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_split_preserves_audio_and_supplied_text(self):
        manifest = workload(self.directory)
        selected = public_benchmark_inputs(manifest, "development")
        self.assertEqual([(row.name, row.transcript, row.duration_seconds) for row in selected],
                         [("clip0", "Original transcript 0.", 1)])
        self.assertEqual([row.name for row in public_benchmark_inputs(manifest, "holdout")], ["clip1"])

    def test_audio_tampering_is_rejected(self):
        manifest = workload(self.directory)
        (self.directory / "clip0.wav").write_bytes(b"changed input")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            public_benchmark_inputs(manifest, "development")

    def test_cannot_silently_truncate_long_audio(self):
        manifest = workload(self.directory, durations=(31,))
        with self.assertRaisesRegex(ValueError, "full 30-second context"):
            public_benchmark_inputs(manifest, "development")
