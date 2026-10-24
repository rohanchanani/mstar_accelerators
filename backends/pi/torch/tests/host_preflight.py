"""Torch ABI/registration preflight; no native context or GPU work is created.

This runs on a build host or the Pi. It deliberately does not initialize the
native provider. Passing it establishes import/schema/device-guard integration,
not working hardware, tensor execution, or the completion contract.
"""

from __future__ import annotations

if not __debug__:
    raise SystemExit("preflight oracles require assertions; remove -O and PYTHONOPTIMIZE")

import json
import multiprocessing as mp

import torch
import mstar_pi


def preflight() -> dict:
    assert torch.__version__.split("+")[0] == "2.9.1"
    assert str(torch.device("mstar_pi:0")) == "mstar_pi:0"
    assert not mstar_pi.is_initialized()
    before = mstar_pi.health()
    assert before == {
        "initialized": False, "closed": False, "poisoned": False,
        "reason": "", "live_storages": 0,
        "retained_storage_owners": 0, "retirement_failures": 0, "retirement_reason": "",
    }, before

    operators = (
        "aten::empty.memory_format", "aten::empty_strided", "aten::copy_",
        "aten::_copy_from", "aten::clone", "aten::fill_.Scalar", "aten::zero_",
        "aten::_local_scalar_dense", "aten::record_stream", "aten::as_strided",
        "aten::view", "aten::_reshape_alias",
    )
    for operator in operators:
        assert torch._C._dispatch_has_kernel_for_dispatch_key(operator, "PrivateUse1"), operator
    assert not torch._C._dispatch_has_backend_fallback("PrivateUse1")

    # Generic Torch event construction is lazy: no native context is needed
    # for an unrecorded event, and its empty dependency is already satisfied.
    event = torch.Event(device="mstar_pi", enable_timing=False)
    assert event.query()
    event.synchronize()
    assert torch.mstar_pi is mstar_pi
    assert mstar_pi.Event().query()
    with mstar_pi.stream(None):
        pass

    rejected = []
    for name, operation in (
        ("tensor_before_initialize", lambda: torch.empty(4, dtype=torch.int32, device="mstar_pi")),
        ("stream_before_initialize", lambda: torch.Stream(device="mstar_pi")),
        ("program_before_initialize", lambda: mstar_pi.prepare_csd([0])),
        ("shutdown_before_initialize", mstar_pi.shutdown),
    ):
        try:
            operation()
        except RuntimeError:
            rejected.append(name)
        else:
            raise AssertionError(f"{name} unexpectedly succeeded")
    torch.manual_seed(123)
    assert mstar_pi.health() == before
    return {
        "status": "PASS", "torch": torch.__version__,
        "scope": "Torch ABI/registration only; no native context or GPU work",
        "read_only_target_probe": mstar_pi.probe(),
        "registered_operators": list(operators), "rejected": rejected,
        "unrecorded_event": "complete", "health": before,
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
