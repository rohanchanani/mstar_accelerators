"""Real Pi hardware test for the native -> retained Torch consumer boundary.

Run under the Pi's existing exclusive admission/lease mechanism. No intentional
device fault, reset, hang, or OOM is induced. A fatal/uncertain context is held
alive for supervised recovery; a timeout/exception is never called a drain.
"""

from __future__ import annotations

if not __debug__:
    raise SystemExit("hardware oracles require assertions; remove -O and PYTHONOPTIMIZE")

import argparse
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import torch
import mstar_pi


def run_cases(kernels: dict) -> list[dict]:
    records = []
    producer = mstar_pi.prepare_csd(kernels["producer"])
    # Distinct immutable code allocations keep the two-prerequisite test from
    # acquiring a transitive ownership dependency through a shared code BO.
    producer_right = mstar_pi.prepare_csd(kernels["producer"])
    consumer = mstar_pi.prepare_csd(kernels["consumer"])
    iterations = 4096
    count = iterations * 16
    guard = 16
    sentinel = 0x5A5A5A5A
    expected = torch.arange(count, dtype=torch.int32)

    def produce(dst, program=producer):
        assert dst.numel() == count
        program.submit([iterations, mstar_pi.address(dst)])

    def increment(src, dst):
        assert src.numel() == dst.numel() == count
        consumer.submit([iterations, mstar_pi.address(src), mstar_pi.address(dst)])

    with torch.inference_mode():
        stream = torch.Stream(device="mstar_pi:0")
        with stream:
            assert torch.accelerator.current_stream(torch.device("mstar_pi:0")) == stream
            assert mstar_pi.current_stream() == stream
            # Nonzero-offset subviews have guard regions in the same Storage.
            raw = torch.full((count + guard * 2,), sentinel, dtype=torch.int32,
                             device="mstar_pi:0")
            x = raw[guard:guard + count]
            produce(x)
            # This immediate clone is the relevant M* consumption pattern.
            # It must wait on CSD in the device dependency system, not read the
            # still-pending mapping as if it were an ordinary CPU tensor.
            retained = x.clone()
            z = torch.empty_like(x)
            increment(retained, z)
            stage = torch.empty((count,), dtype=torch.int32, pin_memory=True)
            assert stage.is_pinned()
            stage.copy_(z, non_blocking=True)
            done = torch.Event(device="mstar_pi", enable_timing=False)
            done.record(stream)
            ready_on_query = done.query()
        done.synchronize()
        torch.testing.assert_close(stage, expected + 1, rtol=0, atol=0)
        raw_cpu = raw.cpu()
        assert torch.all(raw_cpu[:guard] == sentinel)
        assert torch.all(raw_cpu[-guard:] == sentinel)
        records.append({
            "case": "csd_vulkan_clone_csd_pinned_cpu_event",
            "elements": count,
            "ready_on_query": ready_on_query,
            "exact": True,
            "guards": True,
            "physical_overlap_measured": False,
        })

        # Retaining A's result and ordinary CPU snapshot must survive B's use
        # of its own result/workspace. Reusing x is explicit and stream ordered.
        owned_cpu = stage.clone()
        with stream:
            x.fill_(7)
            increment(x, z)
            next_stage = torch.empty_like(stage, pin_memory=True)
            next_stage.copy_(z, non_blocking=True)
            next_done = torch.Event(device="mstar_pi")
            next_done.record(stream)
        next_done.synchronize()
        torch.testing.assert_close(next_stage, torch.full_like(expected, 8), rtol=0, atol=0)
        torch.testing.assert_close(retained.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(owned_cpu, expected + 1, rtol=0, atol=0)
        records.append({"case": "retained_result_and_owned_cpu_copy", "exact": True})

        # M* also stages completed execution results on a separate D2H stream.
        # The wait inserts a dependency; the host waits only at CPU consumption.
        side = mstar_pi.Stream()
        with mstar_pi.stream(side):
            side.wait_event(next_done)
            worker_stage = torch.empty_like(stage, pin_memory=True)
            worker_stage.copy_(z, non_blocking=True)
            worker_done = mstar_pi.Event()
            worker_done.record(side)
        worker_done.synchronize()
        torch.testing.assert_close(worker_stage, torch.full_like(expected, 8), rtol=0, atol=0)
        records.append({"case": "worker_side_stream_d2h", "exact": True})

        # A native submission keeps its inputs alive if Python releases all
        # Tensor references before final completion. record_stream additionally
        # tests allocator bookkeeping; native declared uses are also retained.
        with stream:
            transient = torch.arange(count, dtype=torch.int32).to("mstar_pi", non_blocking=True)
            transient.record_stream(stream)
            increment(transient, z)
            del transient
            gc.collect()
            stage.copy_(z, non_blocking=True)
            done.record(stream)
        done.synchronize()
        torch.testing.assert_close(stage, expected + 1, rtol=0, atol=0)
        records.append({"case": "release_storage_while_declared_use_pending", "exact": True})

        # A native no-op-free state transition: persistent output serves as
        # next invocation's input, with only the final CPU observation waiting.
        with stream:
            state = expected.to("mstar_pi", non_blocking=True)
            for _ in range(4):
                increment(state, state)
            stage.copy_(state, non_blocking=True)
            done.record(stream)
        done.synchronize()
        torch.testing.assert_close(stage, expected + 4, rtol=0, atol=0)
        records.append({"case": "persistent_state", "steps": 4, "exact": True})

        # The wait on right must preserve left's existing prefix. There is no
        # claim that independent branches overlap physically on this hardware.
        left = torch.Stream(device="mstar_pi:0")
        right = torch.Stream(device="mstar_pi:0")
        with left:
            a = torch.empty_like(x)
            produce(a)
        with right:
            b = torch.empty_like(x)
            produce(b, producer_right)
            right_done = torch.Event(device="mstar_pi")
            right_done.record(right)
        with left:
            left.wait_event(right_done)
            a_copy = a.clone()
            b_copy = b.clone()
            joined = torch.Event(device="mstar_pi")
            joined.record(left)
        joined.synchronize()
        torch.testing.assert_close(a_copy.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(b_copy.cpu(), expected, rtol=0, atol=0)
        records.append({"case": "existing_prefix_and_waited_event", "exact": True})

        # A wait consumes the recorded version, not a mutable outer Event.
        # Waiting on the first snapshot and then re-recording that outer Event
        # must not rewrite earlier dependency objects into a cycle.
        # The CSD and Vulkan branches also both read a; any extra foreign-
        # ownership ordering needed for that valid read/read use is supplied
        # by the native allocation provider, not by an invented caller fence.
        with left:
            produce(a)
            mutable_event = torch.Event(device="mstar_pi")
            mutable_event.record(left)
        with right:
            right.wait_event(mutable_event)
            old_prefix = torch.Event(device="mstar_pi")
            old_prefix.record(right)
        with left:
            left.wait_event(old_prefix)
            increment(a, b)
            mutable_event.record(left)
        with right:
            snapshot_result = a.clone()
            snapshot_done = torch.Event(device="mstar_pi")
            snapshot_done.record(right)
        snapshot_done.synchronize()
        mutable_event.synchronize()
        torch.testing.assert_close(snapshot_result.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(b.cpu(), expected + 1, rtol=0, atol=0)
        records.append({"case": "event_rerecord_preserves_prior_wait_snapshot", "exact": True})

        # Bit-preserving copy of both supported floating and 64-bit storage.
        # No native arithmetic support is inferred from this copy test.
        for dtype in (torch.float32, torch.int64):
            cpu = torch.arange(64, dtype=dtype).reshape(4, 16)
            got = cpu.to("mstar_pi").view(64).clone().cpu().view(4, 16)
            torch.testing.assert_close(got, cpu, rtol=0, atol=0)
        records.append({"case": "fp32_int64_storage_and_metadata", "exact": True})

        # M* token/sampling paths eventually read scalar results on the CPU.
        # item() must establish readiness before dereferencing that value.
        token = torch.tensor([42], dtype=torch.int64).to("mstar_pi")
        assert token.item() == 42
        records.append({"case": "int64_scalar_cpu_read", "exact": True})

        # Unsupported operations reject before native acceptance, not through
        # a hidden CPU computation/transfer fallback.
        rejected = []
        for name, operation in (
            ("dtype", lambda: torch.empty(16, dtype=torch.float16, device="mstar_pi")),
            ("strided_copy", lambda: a[::2].clone()),
            ("int64_nonrepeated_fill", lambda: torch.full((16,), 1, dtype=torch.int64, device="mstar_pi")),
        ):
            try:
                operation()
            except (RuntimeError, mstar_pi.Rejected):
                assert not mstar_pi.health()["poisoned"]
                rejected.append(name)
            else:
                raise AssertionError(f"{name} unexpectedly succeeded")
        records.append({"case": "explicit_unsupported_rejection", "rejected": rejected})

        # An ordinary host exception after accepted submission is followed by
        # a real wait, while owners remain live. It is not fallback/retry.
        with stream:
            increment(x, z)
            try:
                raise ValueError("controlled post-submit host exception")
            except ValueError:
                stream.synchronize()
        records.append({"case": "host_exception_after_submit", "settled": True})

        mstar_pi.synchronize()
    return records


def hold_uncertain_context(report: dict) -> None:
    print(json.dumps(report, sort_keys=True), flush=True)
    print(
        "NATIVE CONTEXT IS UNCERTAIN. Do not release the device lease, retry, "
        "or interpret process termination as a drain. This process is retaining "
        "native owners pending supervised device recovery.",
        file=sys.stderr, flush=True,
    )
    # A normal Ctrl-C must not turn containment into implicit destruction.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        signal.pause()


def one_process(config: dict, connection=None) -> None:
    started = time.monotonic()
    try:
        mstar_pi.initialize(copy_shader=config["shader"], render_node=config["render_node"],
                            wait_timeout_ms=config["wait_timeout_ms"], validation=config["validation"])
        kernels = json.loads(Path(config["kernels"]).read_text())
        records = run_cases(kernels)
        gc.collect()
        mstar_pi.collect()
        health = mstar_pi.health()
        assert health["live_storages"] == 0, health
        assert health["retirement_failures"] == 0, health
        assert health["retained_storage_owners"] == 0, health
        mstar_pi.shutdown()
        result = {
            "status": "PASS", "pid": os.getpid(), "records": records,
            "torch": torch.__version__, "elapsed_seconds": time.monotonic() - started,
            "scope": "native/Torch consumer boundary; not complete M* or Whisper",
            "health_before_shutdown": health,
        }
        if connection is not None:
            connection.send(result)
        else:
            print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException as error:
        result = {"status": "FAIL", "error": repr(error), "traceback": traceback.format_exc(),
                  "pid": os.getpid(), "health": mstar_pi.health()}
        if result["health"]["poisoned"]:
            if connection is not None:
                connection.send(result)
            hold_uncertain_context(result)
        # A logical/test failure is still allowed to have accepted native work.
        # Drain it before releasing the ordinary failure result.
        if mstar_pi.is_initialized():
            try:
                mstar_pi.synchronize()
            except BaseException as drain_error:
                result["drain_error"] = repr(drain_error)
                result["health"] = mstar_pi.health()
                if connection is not None:
                    connection.send(result)
                hold_uncertain_context(result)
        result["accepted_work_settled"] = True
        if connection is not None:
            connection.send(result)
        else:
            print(json.dumps(result, sort_keys=True), flush=True)
        raise
    finally:
        if connection is not None:
            connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shader", required=True)
    parser.add_argument("--kernels", required=True)
    parser.add_argument("--render-node", default="/dev/dri/renderD128")
    parser.add_argument("--wait-timeout-ms", type=int, default=30_000)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--skip-spawn", action="store_true")
    args = parser.parse_args()
    config = {
        "shader": str(Path(args.shader).resolve(strict=True)),
        "kernels": str(Path(args.kernels).resolve(strict=True)),
        "render_node": args.render_node,
        "wait_timeout_ms": args.wait_timeout_ms,
        "validation": args.validation,
    }
    one_process(config)
    if not args.skip_spawn:
        context = mp.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        child = context.Process(target=one_process, args=(config, writer))
        child.start()
        writer.close()
        # Deliberately no kill-on-deadline recovery. The caller's lease owner
        # must distinguish a slow test from an uncertain device context.
        result = reader.recv()
        print(json.dumps({"spawned_worker": result}, sort_keys=True), flush=True)
        reader.close()
        if result.get("health", {}).get("poisoned") or "drain_error" in result:
            hold_uncertain_context({"status": "CHILD_HELD", "child_pid": child.pid, "child": result})
        child.join()
        if child.exitcode != 0 or result["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
