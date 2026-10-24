"""A boot-scoped physical-device lease shared by the processes that own work.

The launcher lends the same flock open-file description to its process tree.
An owner is reserved before spawn, and acknowledges release only after native
shutdown.  Process exit and a healthy status are not release acknowledgements.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
from multiprocessing.reduction import DupFd
import os
from pathlib import Path
import tempfile
import time
import uuid


ENVIRONMENT = "MSTAR_PI_ADMISSION"


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
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _restore_context(fd, path, boot, session):
    return AdmissionContext(fd.detach(), Path(path), boot, session)


@dataclass
class AdmissionContext:
    fd: int
    state: Path
    boot: str
    session: str

    def __reduce__(self):
        # DupFd must run during Process.start's spawning pickler.  Creating it
        # when reserving an owner would involve a disposable parent's resource
        # sharer instead of transferring the descriptor through spawn itself.
        return _restore_context, (DupFd(self.fd), str(self.state), self.boot, self.session)

    @property
    def directory(self):
        return self.state / "sessions" / self.session

    @contextmanager
    def metadata(self):
        # Cohort members share the device flock's open-file description; it
        # cannot serialize them against each other.  Metadata has its own lock.
        with (self.directory / "metadata.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def reserve(self, name: str) -> "WorkerOwner":
        with self.metadata():
            session = json.loads((self.directory / "session.json").read_text())
            if session["sealed"]:
                raise RuntimeError("the admission owner roster is sealed")
            owner_id = uuid.uuid4().hex
            atomic_json(self.directory / f"owner-{owner_id}.json", {
                "owner": owner_id, "name": name, "state": "reserved",
                "boot_id": self.boot, "session": self.session,
                "reserved_by": os.getpid(), "updated_unix": time.time(),
            })
        return WorkerOwner(AdmissionContext(os.dup(self.fd), self.state, self.boot, self.session), owner_id)

    def owners(self) -> list[dict]:
        with self.metadata():
            return [json.loads(path.read_text()) for path in sorted(self.directory.glob("owner-*.json"))]

    def seal(self) -> None:
        with self.metadata():
            path = self.directory / "session.json"
            session = json.loads(path.read_text())
            session["sealed"] = True
            atomic_json(path, session)

    def quarantine(self, reason: str) -> None:
        with self.metadata():
            path = self.state / "quarantine.json"
            if path.exists():
                existing = json.loads(path.read_text())
                if existing.get("boot_id") == self.boot:
                    return  # preserve the first failure
            atomic_json(path, {
                "boot_id": self.boot, "session": self.session,
                "pid": os.getpid(), "reason": reason,
                "updated_unix": time.time(),
            })

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        # Never LOCK_UN: it would unlock the whole shared open-file description.

    def __del__(self):
        self.close()

    def environment(self) -> str:
        return json.dumps({"fd": self.fd, "state": str(self.state), "boot": self.boot, "session": self.session})

    @classmethod
    def from_environment(cls):
        value = os.environ.get(ENVIRONMENT)
        if value is None:
            return None
        data = json.loads(value)
        fd = os.dup(data["fd"])
        return cls(fd, Path(data["state"]), data["boot"], data["session"])

    @classmethod
    def create(cls, fd: int, state: Path, boot: str):
        context = cls(os.dup(fd), state, boot, uuid.uuid4().hex)
        context.directory.mkdir(parents=True)
        atomic_json(context.directory / "session.json", {
            "boot_id": boot, "session": context.session, "sealed": False,
        })
        return context


@dataclass
class WorkerOwner:
    context: AdmissionContext
    owner_id: str

    @property
    def path(self):
        return self.context.directory / f"owner-{self.owner_id}.json"

    def status(self) -> str:
        return json.loads(self.path.read_text())["state"]

    def _set(self, state: str, reason: str | None = None) -> None:
        with self.context.metadata():
            record = json.loads(self.path.read_text())
            if state == "active" and record["state"] != "reserved":
                raise RuntimeError("only a reserved owner may activate")
            if record["state"] == "held" and state != "held":
                raise RuntimeError("a held owner cannot acknowledge ordinary release")
            record.update(state=state, pid=os.getpid(), updated_unix=time.time())
            if reason is not None:
                record["reason"] = reason
            atomic_json(self.path, record)

    def activate(self) -> None:
        self._set("active")

    def release(self) -> None:
        self._set("released")
        self.context.close()

    def hold(self, reason: str) -> None:
        self.context.quarantine(reason)
        self._set("held", reason)

    def unknown(self, reason: str) -> None:
        self.context.quarantine(reason)
