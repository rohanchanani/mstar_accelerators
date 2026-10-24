#!/usr/bin/env python3
"""Pi 5 qualification admission using the existing shared lease and quarantine.

Unlike the old diagnostic runner, a deadline or signal never kills a child
that may own submitted GPU work. The lease stays held while it is alive. A
same-boot failure marker blocks later launchers, including the old launcher.
This is a test-admission tool, not the backend's completion mechanism.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def board_identity() -> tuple[str, str]:
    model = Path("/proc/device-tree/model").read_bytes().rstrip(b"\0").decode()
    if not model.startswith("Raspberry Pi 5 "):
        raise RuntimeError(f"Pi 5 qualification requires the Pi 5; found {model!r}")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if str(uuid.UUID(boot)) != boot:
        raise RuntimeError("noncanonical boot identity")
    return model, boot


def state_directory() -> Path:
    # Same physical-device admission domain as the existing investigation.
    # There is deliberately no CLI/environment override creating another lock.
    return Path.home() / ".local/state/mstar-accelerators/pi5-gpu"


def admit(command: list[str], deadline_seconds: float, state: Path, boot: str, *, managed: bool = False) -> int:
    state.mkdir(parents=True, exist_ok=True)
    with (state / "lease.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Pi 5 GPU is already leased") from exc

        for name in ("quarantine.json", "inflight.json"):
            path = state / name
            if not path.exists():
                continue
            # Malformed state fails closed. Do not erase history or guess a boot.
            marker = json.loads(path.read_text())
            if not isinstance(marker, dict) or not isinstance(marker.get("boot_id"), str):
                raise RuntimeError(f"invalid existing device marker: {path}")
            if str(uuid.UUID(marker["boot_id"])) != marker["boot_id"]:
                raise RuntimeError(f"invalid boot identity in device marker: {path}")
            if marker["boot_id"] == boot:
                raise RuntimeError(f"same-boot unresolved device marker: {path}; recovery required")
            # Preserve previous boots instead of overwriting their only record.
            history = state / "history" / marker["boot_id"]
            history.mkdir(parents=True, exist_ok=True)
            saved = history / name
            if saved.exists() and json.loads(saved.read_text()) != marker:
                saved = history / f"{time.time_ns()}-{name}"
            atomic_json(saved, marker)

        record = {
            "boot_id": boot, "pid": os.getpid(), "command": command,
            "started_unix": time.time(), "runner": "mstar-pi-stage2",
        }
        lock.seek(0)
        lock.truncate()
        json.dump(record, lock)
        lock.flush()
        os.fsync(lock.fileno())
        atomic_json(state / "inflight.json", record)

        context = None
        if managed:
            from mstar_pi.admission import AdmissionContext, ENVIRONMENT
            context = AdmissionContext.create(lock.fileno(), state, boot)
            record["session"] = context.session
            atomic_json(state / "inflight.json", record)

        interrupted: int | None = None
        previous_handlers = {}

        def on_signal(signum: int, _frame: object) -> None:
            nonlocal interrupted
            interrupted = signum

        child = None
        quarantine_reason: str | None = None
        deadline = time.monotonic() + deadline_seconds
        try:
            # Install ordinary interruption handling before the child can own
            # GPU work. The durable inflight record also predates the child.
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                previous_handlers[signum] = signal.signal(signum, on_signal)
            try:
                launch = {}
                if context is not None:
                    launch = {
                        "env": {**os.environ, ENVIRONMENT: context.environment()},
                        "pass_fds": (context.fd,),
                    }
                child = subprocess.Popen(command, start_new_session=True, **launch)
            except BaseException:
                # No child was returned to this launcher. Popen's exec-error
                # protocol has reaped its failed child before raising.
                (state / "inflight.json").unlink()
                raise
            record["child_pid"] = child.pid
            # Marker updates can fail too. They belong in the same containment
            # scope as the wait loop, rather than releasing a live child's lease.
            atomic_json(state / "inflight.json", record)
            sealed = False
            while True:
                status = child.poll()
                if status is not None:
                    if context is None:
                        break
                    if not sealed:
                        context.seal()
                        sealed = True
                    if all(owner["state"] == "released" for owner in context.owners()):
                        break
                reason = None
                if interrupted:
                    reason = f"admission process received signal {interrupted}"
                elif time.monotonic() >= deadline:
                    reason = "qualification deadline expired"
                elif status is not None:
                    reason = "direct child exited before every native owner acknowledged release"
                if reason and quarantine_reason is None:
                    quarantine_reason = reason
                    atomic_json(state / "quarantine.json", {**record, "reason": reason})
                    print(
                        f"GPU quarantined: {reason}. Child {child.pid} was NOT killed; "
                        "lease remains held until every owner releases or the board reboots.",
                        file=sys.stderr, flush=True,
                    )
                time.sleep(0.1)
            status = child.wait()
        except BaseException as exc:
            if child is not None:
                try:
                    atomic_json(state / "quarantine.json", {
                        **record, "reason": f"admission failure: {type(exc).__name__}",
                    })
                finally:
                    # Even a second filesystem error while recording failure
                    # cannot release a live child's lease. The initial inflight
                    # marker remains the durable fail-closed record.
                    while child.poll() is None:
                        time.sleep(0.1)
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            if context is not None:
                context.close()

        # A managed owner can report uncertainty before eventually draining.
        # Its later release settles ownership, not the qualification failure.
        if managed and quarantine_reason is None:
            marker_path = state / "quarantine.json"
            if marker_path.exists():
                marker = json.loads(marker_path.read_text())
                if marker["boot_id"] == boot:
                    quarantine_reason = marker["reason"]

        # Any failed qualification is preserved as a same-boot quarantine.
        # We deliberately do not infer safe settlement from arbitrary exit codes.
        if status != 0 or quarantine_reason is not None:
            reason = quarantine_reason or f"qualification exited with status {status}"
            marker_path = state / "quarantine.json"
            if not marker_path.exists() or json.loads(marker_path.read_text())["boot_id"] != boot:
                atomic_json(marker_path, {**record, "status": status, "reason": reason})
            return 124 if quarantine_reason else (128 - status if status < 0 else status)

        (state / "inflight.json").unlink()
        print(json.dumps({"status": "PASS", "boot_id": boot, "child_status": status}), flush=True)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=float, default=120.0)
    parser.add_argument("--managed", action="store_true", help="require release acknowledgements from the native-owner process tree")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.deadline <= 0:
        parser.error("a command and a positive deadline are required")
    _, boot = board_identity()
    return admit(command, args.deadline, state_directory(), boot, managed=args.managed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"admission rejected: {error}", file=sys.stderr)
        raise SystemExit(2)
