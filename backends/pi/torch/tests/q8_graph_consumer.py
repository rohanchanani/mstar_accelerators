"""Explicit packed-format conformance using the existing graph gate lifecycle.

The dot comparator reconstructs the explicitly quantized operands. It does not
replace or relax the model's F32 reference/qualification policy.
"""
from __future__ import annotations

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
from mstar_pi.ggml import Graph, prepare_graph
from consumer_chain import hold_uncertain_context


EXPECTED_REJECTIONS = (mstar_pi._C.Rejected, RuntimeError, ValueError)


def rejection_exception_control():
    """Host-only check of the actual extension exception, before initialization."""
    caught = []
    for exception_type in EXPECTED_REJECTIONS:
        try:
            raise exception_type("expected rejection control")
        except EXPECTED_REJECTIONS as error:
            caught.append(type(error).__name__)
    return {"passed": len(caught) == 3, "caught": caught,
            "native_mro": [kind.__name__ for kind in mstar_pi._C.Rejected.__mro__]}


def run_cases():
    records = []
    work = mstar_pi.Stream()
    generator = torch.Generator().manual_seed(20261024)

    def close_check(actual, expected):
        try:
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
        except AssertionError as error:
            return {"passed": False, "detail": str(error)}
        return {"passed": True}

    def stage(tensor):
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        host.copy_(tensor, non_blocking=True)
        event = mstar_pi.Event(); event.record(work)
        return host, event

    with torch.inference_mode():
        for m, k, n in ((34, 32, 1), (33, 128, 1), (17, 384, 4), (51865, 384, 1)):
            weight_q = torch.randint(-127, 128, (m, k // 32, 32), generator=generator, dtype=torch.int8)
            delta = torch.full((m, k // 32, 1), 1 / 256, dtype=torch.float16)
            raw = torch.cat((delta.view(torch.uint8), weight_q.view(torch.uint8)), dim=-1)
            weight_bits = raw.contiguous().view(-1).view(torch.int32)
            weight_values = (weight_q.float() * delta.float()).reshape(m, k)
            words = ((k * n + 127) // 128) * 36

            def factory():
                g = Graph(max_nodes=32)
                w = g.tensor("w", (k, m), "q8_0")
                source = g.tensor("source", (k, n + 1))
                # A nonzero aligned view checks real source-window binding.
                x = g.view(source, (k, n), (4, 4 * k), offset=4 * k)
                packed = g.quantize_q8_1_x4(x)
                g.output("packed", packed)
                g.output("y", g.q8_0_dot_x4(w, packed))
                return g

            with work:
                guard = torch.full((8,), 0x12345678, dtype=torch.int32)
                whole_weight = torch.cat((guard, weight_bits, guard)).to("mstar_pi", non_blocking=True)
                w = whole_weight[8:-8]
                program = prepare_graph(factory, {"w": w}, slots=2)
                assert program.info()["source_precision"] == "explicit-q8_0/q8_1_x4-with-f32-output"
            previous = None
            for repeat in range(2):
                # Every block has exact amax127/64, so quantizer d=1/64 and
                # integer rounding is exact, with no halfway-rounding ambiguity.
                input_q = torch.randint(-127, 128, (n, k // 32, 32), generator=generator, dtype=torch.int8)
                input_q[..., 0] = 127; input_q[..., -1] = -127
                source_cpu = torch.cat((torch.full((1, k), 17.0), input_q.float().reshape(n, k) / 64))
                with work:
                    source = source_cpu.to("mstar_pi", non_blocking=True)
                    packed = torch.empty((words,), dtype=torch.int32, device="mstar_pi")
                    y = torch.empty((n, m), device="mstar_pi")
                    lease = program.try_acquire(); assert lease is not None
                    lease.submit({"source": source}, {"packed": packed, "y": y})
                    retained = y.clone()
                    packed_host, packed_done = stage(packed)
                    result_host, result_done = stage(retained)
                result_done.synchronize(); packed_done.synchronize(); mstar_pi.collect()
                packed_bytes = packed_host.view(torch.uint8)
                decoded = torch.empty((n * k // 32, 32), dtype=torch.float32)
                format_pass = True
                for block in range(n * k // 32):
                    outer, inner = divmod(block, 4)
                    pair = packed_bytes[outer * 144 + inner * 4:outer * 144 + inner * 4 + 4].view(torch.float16)
                    quants = packed_bytes[outer * 144 + 16 + inner * 32:outer * 144 + 16 + (inner + 1) * 32].view(torch.int8)
                    original = input_q.reshape(-1, 32)[block]
                    format_pass = format_pass and torch.equal(quants, original) and pair[0].item() == 1 / 64
                    expected_sum = (original.float().sum() / 64).to(torch.float16)
                    format_pass = format_pass and pair[1].item() == expected_sum.item()
                    decoded[block] = quants.float() * pair[0].float()
                expected = (decoded.reshape(n, k).double() @ weight_values.double().T).float()
                comparison = close_check(result_host, expected)
                retained_pass = True
                if previous is not None:
                    old_tensor, old_expected = previous
                    with work:
                        old_host, old_done = stage(old_tensor)
                    old_done.synchronize()
                    retained_pass = torch.equal(old_host.view(torch.int32), old_expected.view(torch.int32))
                previous = retained, result_host.clone()
                records.append({"case": "explicit_q8_formats_and_f32_dot", "m": m, "k": k, "n": n,
                                "repeat": repeat, "weight_bytes": weight_bits.numel() * 4,
                                "weight_storage_offset_bytes": 32,
                                "format_pass": format_pass, "comparison": comparison, "retained_pass": retained_pass,
                                "passed": format_pass and comparison["passed"] and retained_pass,
                                "packed_bytes": words * 4,
                                "max_absolute_error": (result_host - expected).abs().max().item(),
                                "rtol": 3e-5, "atol": 3e-5,
                                "oracle": "FP64 dot of explicitly reconstructed quantized operands; F32 output"})
            program.close()

        rejection_checks = []
        def rejected(make):
            try:
                value = make()
            except EXPECTED_REJECTIONS:
                rejection_checks.append(True)
                return
            if hasattr(value, "close"):
                value.close()
            rejection_checks.append(False)

        rejected(lambda: Graph().tensor("bad", (33, 2), "q8_0"))
        g = Graph(); q = g.tensor("q", (128, 33), "q8_0")
        rejected(lambda: g.scale(q, 1.0))
        fake = g.tensor("fake", (36,), "i32")
        rejected(lambda: g.q8_0_dot_x4(q, fake))
        with work:
            rejected(lambda: prepare_graph(factory, {}, slots=1))
            wrong_extent = torch.empty((weight_bits.numel() + 1,), dtype=torch.int32, device="mstar_pi")
            rejected(lambda: prepare_graph(factory, {"w": wrong_extent}, slots=1))
        records.append({"case": "q8_rejects_bad_block_shape_arithmetic_fake_producer_missing_constant_and_extent",
                        "exact": True, "checks": rejection_checks, "passed": len(rejection_checks) == 5 and all(rejection_checks)})
        mstar_pi.synchronize(); mstar_pi.collect()
    return records


EXPECTED = [(m, k, n, repeat) for m, k, n in ((34, 32, 1), (33, 128, 1), (17, 384, 4), (51865, 384, 1))
            for repeat in range(2)]


def one_process(config, connection=None):
    started = time.monotonic()
    try:
        rejection_control = rejection_exception_control()
        if not rejection_control["passed"]:
            raise RuntimeError("expected rejection exception control incomplete")
        mstar_pi.initialize(copy_shader=config["shader"], render_node=config["render_node"],
                            wait_timeout_ms=config["wait_timeout_ms"], validation=config["validation"])
        records = run_cases()
        gc.collect(); mstar_pi.collect(); health = mstar_pi.health()
        assert not health["poisoned"], health
        assert health["live_storages"] == health["retirement_failures"] == health["retained_storage_owners"] == 0, health
        mstar_pi.shutdown()
        observed = [(r["m"], r["k"], r["n"], r["repeat"]) for r in records if r["case"] == "explicit_q8_formats_and_f32_dot"]
        complete = observed == EXPECTED and len(records) == len(EXPECTED) + 1
        compared = complete and all(r["passed"] for r in records)
        result = {"status": "PASS" if compared else "FAIL", "execution_status": "PASS" if complete else "FAIL",
                  "comparison_status": "PASS" if compared else "FAIL" if complete else "INCOMPLETE",
                  "native_context_released": True, "complete": complete, "records": records,
                  "rejection_exception_control": rejection_control,
                  "health_before_shutdown": health, "pid": os.getpid(),
                  "elapsed_seconds": time.monotonic() - started,
                  "scope": "explicit packed-format conformance; reconstructed quantized-operand dot; no model oracle",
                  "launcher_exit_semantics": "zero requires complete execution and released context; settled numerical FAIL stays in report"}
        if connection: connection.send(result)
        else: print(json.dumps(result, sort_keys=True), flush=True)
        return result
    except BaseException as error:
        result = {"status": "FAIL", "execution_status": "FAIL", "comparison_status": "INCOMPLETE",
                  "complete": False, "native_context_released": False, "error": repr(error),
                  "traceback": traceback.format_exc(), "health": mstar_pi.health(), "pid": os.getpid()}
        if result["health"]["poisoned"]:
            if connection: connection.send(result)
            hold_uncertain_context(result)
        if mstar_pi.is_initialized():
            try: mstar_pi.synchronize()
            except BaseException as drain_error:
                result["drain_error"] = repr(drain_error); result["health"] = mstar_pi.health()
                if connection: connection.send(result)
                hold_uncertain_context(result)
        result["accepted_work_settled"] = True
        if connection: connection.send(result)
        else: print(json.dumps(result, sort_keys=True), flush=True)
        raise
    finally:
        if connection: connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shader", required=True)
    parser.add_argument("--render-node", default="/dev/dri/renderD128")
    parser.add_argument("--wait-timeout-ms", type=int, default=30000)
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    config = {"shader": str(Path(args.shader).resolve(strict=True)), "render_node": args.render_node,
              "wait_timeout_ms": args.wait_timeout_ms, "validation": args.validation}
    parent = one_process(config)
    ctx = mp.get_context("spawn"); reader, writer = ctx.Pipe(duplex=False)
    child = ctx.Process(target=one_process, args=(config, writer)); child.start(); writer.close()
    result = reader.recv(); reader.close()
    print(json.dumps({"spawned_worker": result}, sort_keys=True), flush=True)
    if result.get("health", {}).get("poisoned") or "drain_error" in result:
        hold_uncertain_context({"status": "CHILD_HELD", "child_pid": child.pid, "child": result})
    child.join()
    if child.exitcode or any(r["execution_status"] != "PASS" or not r["complete"] or not r["native_context_released"]
                             for r in (parent, result)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
