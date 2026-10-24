"""Real ggml FP32 linear execution through the Pi/Torch storage boundary.

This is an actual provider test, not a Whisper model or a performance claim.
No reset, failed fence, hung kernel, or OOM is induced. Run only through the
existing shared hardware lease. The frozen Stage 1 consumer test remains a
separate required regression. CPU arithmetic below supplies test oracles only.
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
import time
import traceback

import torch
import mstar_pi

# Reuse the established containment procedure, not its compute path or oracles.
from consumer_chain import hold_uncertain_context


def run_cases(copy_fixture: dict) -> list[dict]:
    records = []
    width = 384
    guard = 32  # Updated from the provider's actual alignment before ggml view tests.
    sentinel = -12345.0
    copier = mstar_pi.prepare_csd(copy_fixture["copy_words"])

    def native_copy(source: torch.Tensor, destination: torch.Tensor) -> None:
        assert source.dtype == destination.dtype == torch.float32
        assert source.numel() == destination.numel()
        assert source.numel() % 16 == 0
        iterations = source.numel() // 16
        assert 1 <= iterations <= copy_fixture["domain"]["iterations_max"]
        copier.submit([iterations, mstar_pi.address(source), mstar_pi.address(destination)])

    def stage_and_record(tensor: torch.Tensor, stream: torch.Stream):
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        host.copy_(tensor, non_blocking=True)
        event = mstar_pi.Event()
        event.record(stream)
        return host, event

    with torch.inference_mode():
        work = mstar_pi.Stream()
        # The new direct-QPU fixture has not inherited Stage 1's hardware
        # qualification. Witness it independently, before relying on it in the
        # provider chain. Compare raw words, including NaNs/subnormals/signed zero.
        bit_patterns = torch.tensor([
            0, -2147483648, 0x3F800000, -1082130432,
            0x7FC00001, 1, 0x7F800000, -1,
        ], dtype=torch.int32).repeat(16)
        with work:
            copy_source = bit_patterns.view(torch.float32).to("mstar_pi", non_blocking=True)
            copy_raw = torch.full((bit_patterns.numel() + 2 * guard,), sentinel,
                                  dtype=torch.float32, device="mstar_pi")
            copy_target = copy_raw[guard:guard + bit_patterns.numel()]
            native_copy(copy_source, copy_target)
            copy_host, copy_done = stage_and_record(copy_target, work)
        copy_done.synchronize()
        torch.testing.assert_close(copy_host.view(torch.int32), bit_patterns, rtol=0, atol=0)
        copy_guards = copy_raw.cpu()
        assert torch.all(copy_guards[:guard] == sentinel)
        assert torch.all(copy_guards[-guard:] == sentinel)
        records.append({"case": "independent_direct_qpu_copy_fixture", "exact_bits": True,
                        "guards": True, "words": bit_patterns.numel()})

        # The program retains a native logical View, not the original Tensor or
        # another Torch StorageOwner.
        diagonal = torch.where(torch.arange(width) % 2 == 0, 2.0, -0.5)
        weight_cpu = torch.diag(diagonal)
        with work:
            weight_tensor = weight_cpu.to("mstar_pi", non_blocking=True)
            program = mstar_pi.prepare_linear(weight_tensor, token_counts=[1, 16], slots=2)
        del weight_tensor
        gc.collect()
        info = program.info()
        assert info["token_counts"] == [1, 16]
        assert info["input_width"] == info["output_width"] == width
        assert info["implicit_host_copies"] is False
        alignment = info["buffer_offset_alignment"]
        assert alignment > 0 and alignment % 4 == 0
        guard = max(32, alignment // 4)
        # This is the actual provider limit; no driver/architecture-name guess
        # is used to make a view silently become valid through a copy.
        assert (guard * 4) % alignment == 0
        assert (width * 4) % alignment == 0

        saved_result = None
        saved_cpu = None
        old_events = []
        for tokens in (1, 16):
            count = tokens * width
            expected_x = ((torch.arange(count, dtype=torch.float32) % 31) - 15).view(tokens, width) / 16
            with work:
                source = expected_x.to("mstar_pi", non_blocking=True)
                # Input and output are disjoint nonzero-offset views of one BO.
                # They must import once, share one ownership frontier, and keep
                # all three guard ranges unchanged.
                raw = torch.full((2 * count + 3 * guard,), sentinel, device="mstar_pi", dtype=torch.float32)
                x = raw[guard:guard + count].view(tokens, width)
                out_start = 2 * guard + count
                out = raw[out_start:out_start + count].view(tokens, width)
                native_copy(source, x)
                program.submit(x, out)
                # No explicit intermediate event or host wait. This immediate
                # retained Torch consumer is the bridge the old pointer wrapper
                # could not establish.
                retained = out.clone()
                final = torch.empty_like(retained)
                native_copy(retained, final)
                host, event = stage_and_record(final, work)
                ready_on_query = event.query()
            event.synchronize()
            mstar_pi.collect()
            expected = expected_x * diagonal
            torch.testing.assert_close(host, expected, rtol=0, atol=0)
            full = raw.cpu()
            for region in (full[:guard], full[guard + count:out_start], full[out_start + count:]):
                assert torch.all(region == sentinel)
            records.append({
                "case": "native_ggml_immediate_clone_native_pinned_event",
                "tokens": tokens, "width": width, "exact": True, "guards": True,
                "same_bo_input_output": True, "nonzero_offsets": True,
                "ready_on_query": ready_on_query,
                "physical_overlap_measured": False,
            })
            old_events.append(event)
            if saved_result is None:
                saved_result = retained
                saved_cpu = expected.clone()

        assert saved_result is not None
        torch.testing.assert_close(saved_result.cpu(), saved_cpu, rtol=0, atol=0)
        records.append({"case": "result_retained_across_next_shape", "exact": True})

        # Consecutive graph invocations use the prepared slot pool. The second
        # reads the first's real state without an intermediate host wait. A fast
        # first invocation may retire before the second leases a slot; record
        # observed lease counts instead of inventing simultaneous occupancy.
        state_cpu = torch.arange(width, dtype=torch.float32).view(1, width) / 64
        with work:
            state = state_cpu.to("mstar_pi", non_blocking=True)
            state_next = torch.empty_like(state)
            state_last = torch.empty_like(state)
            program.submit(state, state_next)
            leased_after_first = program.info()["active_slots"]
            program.submit(state_next, state_last)
            leased_after_second = program.info()["active_slots"]
            state_host, state_done = stage_and_record(state_last, work)
        state_done.synchronize()
        mstar_pi.collect()
        torch.testing.assert_close(state_host, state_cpu * diagonal * diagonal, rtol=0, atol=0)
        assert program.info()["active_slots"] == 0
        # Keep every prior event object live while recycling those same slots.
        # Events own immutable completion tails, not mutable graph leases.
        old_events.append(state_done)
        with work:
            program.submit(state, state_next)
            again_host, again_done = stage_and_record(state_next, work)
        again_done.synchronize()
        mstar_pi.collect()
        torch.testing.assert_close(again_host, state_cpu * diagonal, rtol=0, atol=0)
        assert all(event.query() for event in old_events)
        records.append({
            "case": "persistent_state_slot_reuse_and_old_event_snapshots", "exact": True,
            "slots_per_shape": 2,
            "leased_slots_after_submissions": [leased_after_first, leased_after_second],
            "two_leases_observed": max(leased_after_first, leased_after_second) == 2,
            "physical_overlap_measured": False,
        })

        # A dense non-diagonal FP32 operation tests the actual provider beyond
        # its exact guard oracle. The tolerance is explicit, not inferred from
        # nominal dtype. The CPU float64 result is only an independent oracle.
        generator = torch.Generator().manual_seed(20261022)
        dense_w_cpu = torch.randn((width, width), generator=generator) * 0.03
        dense_x_cpu = torch.randn((16, width), generator=generator) * 0.2
        expected_dense = (dense_x_cpu.double() @ dense_w_cpu.double().T).float()
        with work:
            dense_w = dense_w_cpu.to("mstar_pi", non_blocking=True)
            dense_program = mstar_pi.prepare_linear(dense_w, token_counts=[1, 16], slots=2)
            dense_source = dense_x_cpu.to("mstar_pi", non_blocking=True)
            dense_x = torch.empty_like(dense_source)
            dense_y = torch.empty_like(dense_x)
            native_copy(dense_source, dense_x)
            dense_program.submit(dense_x, dense_y)
            dense_retained = dense_y.clone()
            dense_host, dense_done = stage_and_record(dense_retained, work)
        dense_done.synchronize()
        mstar_pi.collect()
        torch.testing.assert_close(dense_host, expected_dense, rtol=3e-5, atol=3e-5)
        records.append({
            "case": "dense_fp32_linear", "tokens": 16, "width": width,
            "rtol": 3e-5, "atol": 3e-5,
            "max_absolute_error": (dense_host - expected_dense).abs().max().item(),
            "oracle": "CPU float64 matmul cast to float32",
        })

        # F32 accumulation alone does not exclude an implementation that first
        # rounds its operands to F16 in shared memory. This exactly checkable
        # single-product witness uses values F16 would round to 1.0, and thus
        # distinguishes the admitted full-F32 shader from that weaker policy.
        precision_value = 1.0 + 2.0 ** -12
        precision_w_cpu = torch.eye(width, dtype=torch.float32) * precision_value
        precision_x_cpu = torch.zeros((1, width), dtype=torch.float32)
        precision_x_cpu[0, 7] = precision_value
        with work:
            precision_w = precision_w_cpu.to("mstar_pi", non_blocking=True)
            precision_program = mstar_pi.prepare_linear(precision_w, token_counts=[1], slots=1)
            precision_x = precision_x_cpu.to("mstar_pi", non_blocking=True)
            precision_y = torch.empty_like(precision_x)
            precision_program.submit(precision_x, precision_y)
            precision_host, precision_done = stage_and_record(precision_y, work)
        precision_done.synchronize()
        mstar_pi.collect()
        torch.testing.assert_close(precision_host, precision_x_cpu * precision_value, rtol=0, atol=0)
        assert precision_host[0, 7].item() != 1.0
        records.append({"case": "full_fp32_operand_precision_witness", "exact": True,
                        "operand": precision_value, "observed": precision_host[0, 7].item()})

        # Two independently submitted producers must both be consumed. Joining
        # one event cannot overwrite the other prerequisite. Fresh code objects
        # avoid creating an accidental transitive dependency via a shared code BO.
        independent_copy = mstar_pi.prepare_csd(copy_fixture["copy_words"])
        weights_stream, inputs_stream = mstar_pi.Stream(), mstar_pi.Stream()
        with weights_stream:
            joined_w = weight_cpu.to("mstar_pi", non_blocking=True)
            joined_program = mstar_pi.prepare_linear(joined_w, token_counts=[1], slots=1)
            weights_ready = mstar_pi.Event()
            weights_ready.record(weights_stream)
        with inputs_stream:
            joined_source = state_cpu.to("mstar_pi", non_blocking=True)
            joined_x = torch.empty_like(joined_source)
            independent_copy.submit([width // 16, mstar_pi.address(joined_source), mstar_pi.address(joined_x)])
            inputs_ready = mstar_pi.Event()
            inputs_ready.record(inputs_stream)
        with work:
            work.wait_event(weights_ready)
            work.wait_event(inputs_ready)
            joined_y = torch.empty_like(joined_x)
            joined_program.submit(joined_x, joined_y)
            joined_host, joined_done = stage_and_record(joined_y, work)
        joined_done.synchronize()
        mstar_pi.collect()
        torch.testing.assert_close(joined_host, state_cpu * diagonal, rtol=0, atol=0)
        records.append({"case": "ggml_two_producer_prerequisites", "exact": True})

        # Drop weights, input tensors, and the Python program while a complete
        # native invocation is pending. The Submission owns the graph and all
        # borrowed allocations until checked retirement. Output remains ordinary
        # caller-owned Torch storage and survives its producer's destruction.
        with work:
            temporary_w = weight_cpu.to("mstar_pi", non_blocking=True)
            temporary_program = mstar_pi.prepare_linear(temporary_w, token_counts=[1], slots=1)
            temporary_x = state_cpu.to("mstar_pi", non_blocking=True)
            temporary_y = torch.empty_like(temporary_x)
            temporary_program.submit(temporary_x, temporary_y)
            del temporary_w, temporary_x, temporary_program
            gc.collect()
            released_result = temporary_y.clone()
            released_host, released_done = stage_and_record(released_result, work)
        released_done.synchronize()
        mstar_pi.collect()
        torch.testing.assert_close(released_host, state_cpu * diagonal, rtol=0, atol=0)
        records.append({"case": "release_python_program_and_inputs_while_pending", "exact": True})

        # Every rejection is before provider acceptance. A prior independent
        # native producer may still be pending; rejection is not its cancellation.
        with work:
            native_copy(state, state_next)
            invalid_shape_x = torch.empty((3, width), device="mstar_pi", dtype=torch.float32)
            invalid_shape_y = torch.empty_like(invalid_shape_x)
            misaligned_storage = torch.empty(width + 1, device="mstar_pi", dtype=torch.float32)
            misaligned = misaligned_storage[1:].view(1, width)
            strided_storage = torch.empty((1, width * 2), device="mstar_pi", dtype=torch.float32)
            dtype_x = torch.empty((1, width), device="mstar_pi", dtype=torch.int32)
            rejected = []
            for name, operation in (
                ("unprepared_shape", lambda: program.submit(invalid_shape_x, invalid_shape_y)),
                ("destructive_alias", lambda: program.submit(state, state)),
                ("noncontiguous", lambda: program.submit(strided_storage[:, ::2], state_last)),
                ("misaligned_offset", lambda: program.submit(misaligned, state_last)),
                ("dtype", lambda: program.submit(dtype_x, state_last)),
            ):
                before = program.info()["submissions"]
                try:
                    operation()
                except (RuntimeError, mstar_pi.Rejected):
                    assert not mstar_pi.health()["poisoned"]
                    assert program.info()["submissions"] == before
                    rejected.append(name)
                else:
                    raise AssertionError(f"{name} unexpectedly succeeded")
            rejection_done = mstar_pi.Event()
            rejection_done.record(work)
        rejection_done.synchronize()
        torch.testing.assert_close(state_next.cpu(), state_cpu, rtol=0, atol=0)
        records.append({"case": "preaccept_rejection_and_prior_producer_drain", "rejected": rejected})

        # This tests ordinary host exception cleanup after a successful real
        # provider submission. It does not induce device loss or pretend that a
        # missing receipt is recoverable; that has its own explicitly gated test.
        with work:
            program.submit(state, state_last)
            try:
                raise ValueError("controlled host exception after ggml acceptance")
            except ValueError:
                work.synchronize()
        mstar_pi.collect()
        torch.testing.assert_close(state_last.cpu(), state_cpu * diagonal, rtol=0, atol=0)
        records.append({"case": "host_exception_after_ggml_submit", "settled": True, "exact": True})

        mstar_pi.synchronize()
        mstar_pi.collect()
        assert program.info()["active_slots"] == 0
        assert program.info()["retirements"] == program.info()["submissions"]
        records.append({"case": "prepared_program_reused_and_retired", "info": program.info()})
    return records


def one_process(config: dict, connection=None) -> None:
    started = time.monotonic()
    try:
        mstar_pi.initialize(copy_shader=config["shader"], render_node=config["render_node"],
                            wait_timeout_ms=config["wait_timeout_ms"], validation=config["validation"])
        fixture = json.loads(Path(config["copy_fixture"]).read_text())
        records = run_cases(fixture)
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
            "scope": "real ggml provider and retained Torch consumer bridge; not full Whisper or M*",
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
    parser.add_argument("--copy-fixture", required=True)
    parser.add_argument("--render-node", default="/dev/dri/renderD128")
    parser.add_argument("--wait-timeout-ms", type=int, default=30_000)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--skip-spawn", action="store_true")
    args = parser.parse_args()
    config = {
        "shader": str(Path(args.shader).resolve(strict=True)),
        "copy_fixture": str(Path(args.copy_fixture).resolve(strict=True)),
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
