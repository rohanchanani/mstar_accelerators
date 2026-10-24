"""Host-only tests of real lease/marker behavior; no GPU simulation claims."""
from __future__ import annotations

import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_hardware", ROOT / "tools/run_hardware.py")
guard = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(guard)
BOOT = "8cc01d34-f310-4ad8-98c1-f08ba570f108"
OLD_BOOT = "66899387-f3b2-4cfa-97ac-c15270257611"


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.state = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_success_releases_inflight(self):
        self.assertEqual(guard.admit([sys.executable, "-c", "pass"], 5, self.state, BOOT), 0)
        self.assertFalse((self.state / "inflight.json").exists())

    def test_same_boot_quarantine_rejects_before_launch(self):
        guard.atomic_json(self.state / "quarantine.json", {"boot_id": BOOT})
        with self.assertRaisesRegex(RuntimeError, "same-boot"):
            guard.admit(["this-command-must-not-run"], 5, self.state, BOOT)

    def test_old_boot_record_is_preserved(self):
        marker = {"boot_id": OLD_BOOT, "reason": "old hardware failure"}
        guard.atomic_json(self.state / "quarantine.json", marker)
        self.assertEqual(guard.admit([sys.executable, "-c", "pass"], 5, self.state, BOOT), 0)
        self.assertEqual(json.loads((self.state / "history" / OLD_BOOT / "quarantine.json").read_text()), marker)

    def test_other_owner_lease_is_respected(self):
        with (self.state / "lease.lock").open("a+") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already leased"):
                guard.admit(["this-command-must-not-run"], 5, self.state, BOOT)

    def test_deadline_keeps_child_alive_and_quarantines(self):
        output = self.state / "normal-exit.txt"
        code = "import pathlib,sys,time; time.sleep(.3); pathlib.Path(sys.argv[1]).write_text('settled')"
        result = guard.admit([sys.executable, "-c", code, str(output)], .01, self.state, BOOT)
        self.assertEqual(result, 124)
        self.assertEqual(output.read_text(), "settled")
        self.assertEqual(json.loads((self.state / "quarantine.json").read_text())["boot_id"], BOOT)
        self.assertTrue((self.state / "inflight.json").exists())

    def test_nonzero_status_does_not_infer_settlement(self):
        self.assertEqual(guard.admit([sys.executable, "-c", "raise SystemExit(7)"], 5, self.state, BOOT), 7)
        self.assertEqual(json.loads((self.state / "quarantine.json").read_text())["status"], 7)

    def test_marker_failure_after_spawn_keeps_lease_until_child_exits(self):
        output = self.state / "normal-exit.txt"
        code = "import pathlib,sys,time; time.sleep(.2); pathlib.Path(sys.argv[1]).write_text('settled')"
        original = guard.atomic_json
        calls = 0

        def fail_after_initial_marker(path, value):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise OSError("controlled marker write failure")
            return original(path, value)

        with mock.patch.object(guard, "atomic_json", side_effect=fail_after_initial_marker):
            with self.assertRaisesRegex(OSError, "controlled marker"):
                guard.admit([sys.executable, "-c", code, str(output)], 5, self.state, BOOT)
        self.assertEqual(output.read_text(), "settled")
        self.assertTrue((self.state / "inflight.json").exists())


if __name__ == "__main__":
    unittest.main()
