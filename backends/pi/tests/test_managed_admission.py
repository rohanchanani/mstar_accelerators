"""Real host descriptor/roster tests; these processes perform no device work."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
# admission.py has no Torch/native dependencies. Import its actual module by
# its source directory so a spawned host process can resolve the same classes.
sys.path.insert(0, str(ROOT / "torch/mstar_pi"))
import admission

spec = importlib.util.spec_from_file_location("managed_admission_guard", ROOT / "tools/run_hardware.py")
guard = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(guard)
BOOT = "8cc01d34-f310-4ad8-98c1-f08ba570f108"


def _spawn_owner(owner, connection):
    owner.activate()
    connection.send("active")
    assert connection.recv() == "release"
    owner.release()
    connection.send("released")
    connection.close()


def _lock_is_held(path):
    with path.open("a+") as contender:
        try:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


@pytest.fixture
def lease(tmp_path):
    with (tmp_path / "lease.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        context = admission.AdmissionContext.create(lock.fileno(), tmp_path, BOOT)
        try:
            yield context, lock
        finally:
            context.close()


def test_acknowledged_release_closes_only_its_own_descriptor(lease):
    context, lock = lease
    owner = context.reserve("native-free-owner")
    assert owner.status() == "reserved"
    owner.activate()
    owner.release()
    assert owner.status() == "released"
    assert owner.context.fd == -1
    assert _lock_is_held(Path(lock.name))


def test_spawn_inherits_shared_lease_until_the_worker_acknowledges_release(lease):
    context, lock = lease
    owner = context.reserve("spawned-host-owner")
    start = multiprocessing.get_context("spawn")
    parent, child = start.Pipe()
    process = start.Process(target=_spawn_owner, args=(owner, child))
    process.start()
    child.close()
    try:
        assert parent.poll(10), "spawned owner did not activate"
        assert parent.recv() == "active"
        assert owner.status() == "active"
        # Every parent copy closes. LOCK_UN here would also unlock the child's
        # open-file description; the real competing flock detects that bug.
        owner.context.close()
        context.close()
        lock.close()
        assert _lock_is_held(Path(lock.name))
        parent.send("release")
        assert parent.poll(10), "spawned owner did not release"
        assert parent.recv() == "released"
        process.join(10)
        assert process.exitcode == 0
        assert owner.status() == "released"
        assert not _lock_is_held(Path(lock.name))
    finally:
        parent.close()
        if process.is_alive():
            # Test cleanup only: this child never loaded a native backend.
            process.terminate()
            process.join(10)


def test_held_owner_cannot_release_and_first_quarantine_is_preserved(lease):
    context, lock = lease
    owner = context.reserve("uncertain-host-owner")
    owner.activate()
    owner.hold("first failure")
    owner.hold("later diagnosis")
    with pytest.raises(RuntimeError, match="held owner"):
        owner.release()
    assert owner.status() == "held"
    assert owner.context.fd >= 0
    assert json.loads((context.state / "quarantine.json").read_text())["reason"] == "first failure"
    assert _lock_is_held(Path(lock.name))
    owner.context.close()


def test_roster_sealing_prevents_late_untracked_native_owners(lease):
    context, _ = lease
    owner = context.reserve("known-owner")
    context.seal()
    with pytest.raises(RuntimeError, match="sealed"):
        context.reserve("late-owner")
    owner.activate()
    owner.release()
    assert [record["state"] for record in context.owners()] == ["released"]


def test_a_released_reservation_cannot_activate_late(lease):
    context, _ = lease
    owner = context.reserve("delayed-native-free-child")
    # A child holds a different descriptor wrapper but the same reservation.
    child_owner = admission.WorkerOwner(
        admission.AdmissionContext(os.dup(context.fd), context.state, context.boot, context.session),
        owner.owner_id,
    )
    owner.activate()
    owner.release()
    try:
        with pytest.raises(RuntimeError, match="reserved"):
            child_owner.activate()
        assert owner.status() == "released"
    finally:
        child_owner.context.close()


def test_an_existing_reservation_can_activate_after_roster_sealing(lease):
    context, _ = lease
    owner = context.reserve("known-delayed-owner")
    context.seal()
    owner.activate()
    assert owner.status() == "active"
    owner.release()


def test_environment_borrow_has_independent_close_but_the_same_lease(lease, monkeypatch):
    context, lock = lease
    monkeypatch.setenv(admission.ENVIRONMENT, context.environment())
    borrowed = admission.AdmissionContext.from_environment()
    assert borrowed.fd != context.fd
    assert borrowed.session == context.session
    context.close()
    lock.close()
    assert _lock_is_held(Path(lock.name))
    borrowed.close()
    assert not _lock_is_held(Path(lock.name))


def test_managed_launcher_passes_the_real_descriptor_and_observes_release(tmp_path, monkeypatch):
    # Supply the real pure-Python admission module without loading the target
    # native extension in this host test process.
    package = ModuleType("mstar_pi")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "mstar_pi", package)
    monkeypatch.setitem(sys.modules, "mstar_pi.admission", admission)
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from admission import AdmissionContext
context = AdmissionContext.from_environment()
owner = context.reserve('native-free child')
owner.activate()
owner.release()
context.close()
"""
    result = guard.admit([sys.executable, "-c", code, str(ROOT / "torch/mstar_pi")],
                         10, tmp_path, BOOT, managed=True)
    assert result == 0
    assert not (tmp_path / "inflight.json").exists()
    assert not (tmp_path / "quarantine.json").exists()
    records = [json.loads(path.read_text()) for path in tmp_path.glob("sessions/*/owner-*.json")]
    assert len(records) == 1 and records[0]["state"] == "released"
    assert not _lock_is_held(tmp_path / "lease.lock")


def test_managed_launcher_preserves_a_cohort_failure_after_safe_release(tmp_path, monkeypatch, capsys):
    package = ModuleType("mstar_pi")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "mstar_pi", package)
    monkeypatch.setitem(sys.modules, "mstar_pi.admission", admission)
    children = []
    popen = guard.subprocess.Popen

    def observe_child(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(guard.subprocess, "Popen", observe_child)
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from admission import AdmissionContext
context = AdmissionContext.from_environment()
owner = context.reserve('native-free late-release child')
owner.activate()
owner.unknown('shutdown exceeded its bounded grace')
marker = context.state / 'quarantine.json'
(context.state / 'first-quarantine.json').write_bytes(marker.read_bytes())
# Completion can settle later without making the earlier run successful.
owner.release()
context.close()
"""
    result = guard.admit([sys.executable, "-c", code, str(ROOT / "torch/mstar_pi")],
                         10, tmp_path, BOOT, managed=True)
    assert len(children) == 1 and children[0].returncode == 0
    assert result != 0
    assert '"status": "PASS"' not in capsys.readouterr().out
    assert (tmp_path / "inflight.json").exists()
    assert (tmp_path / "quarantine.json").read_bytes() == (tmp_path / "first-quarantine.json").read_bytes()
    records = [json.loads(path.read_text()) for path in tmp_path.glob("sessions/*/owner-*.json")]
    assert len(records) == 1 and records[0]["state"] == "released"
    assert not _lock_is_held(tmp_path / "lease.lock")
    with pytest.raises(RuntimeError, match="same-boot unresolved"):
        guard.admit([sys.executable, "-c", "raise AssertionError('must not enter')"],
                    10, tmp_path, BOOT, managed=True)
