"""Register the real provider API without creating a native or ggml device."""

from __future__ import annotations

if not __debug__:
    raise SystemExit("preflight oracles require assertions; remove -O and PYTHONOPTIMIZE")

import json
import multiprocessing as mp

import torch
import mstar_pi


def preflight() -> dict:
    assert not mstar_pi.is_initialized()
    before = mstar_pi.health()
    assert callable(mstar_pi.prepare_linear)
    assert hasattr(mstar_pi.PreparedLinear, "submit")
    assert hasattr(mstar_pi.PreparedLinear, "info")
    rejected = []
    for name, operation in (
        ("wrapper_before_initialize", lambda: mstar_pi.prepare_linear(
            torch.eye(16), token_counts=[1, 16])),
        ("constructor_before_initialize", lambda: mstar_pi.PreparedLinear(
            torch.eye(16), [1], 1)),
    ):
        try:
            operation()
        except RuntimeError:
            rejected.append(name)
        else:
            raise AssertionError(f"{name} unexpectedly initialized a device")
    assert mstar_pi.health() == before
    return {
        "status": "PASS", "torch": torch.__version__,
        "scope": "ggml provider registration and no implicit initialization only",
        "rejected": rejected, "health": before,
    }


def spawned(connection) -> None:
    try:
        connection.send(preflight())
    finally:
        connection.close()


def main() -> None:
    print(json.dumps(preflight(), sort_keys=True), flush=True)
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=spawned, args=(writer,))
    child.start()
    writer.close()
    result = reader.recv()
    reader.close()
    child.join()
    assert child.exitcode == 0
    print(json.dumps({"spawned_preflight": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
