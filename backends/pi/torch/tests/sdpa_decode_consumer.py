"""Initial bounded Q1-attention mechanism probe, not model qualification.

Uses the retained graph_consumer initialization/completion/health/shutdown and
parent/spawn lifecycle. A wrapper classifies completed numerical rejection only
after that existing lifecycle has returned with an actually closed context.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import torch
import mstar_pi
from mstar_pi.ggml import Graph, prepare_graph
import graph_consumer as retained_gate


HEADS, TOKENS, BATCH, PADDING = 2, 17, 2, 8
SCALE = .125


def outcome(records, final_health):
    complete = [r.get("repeat") for r in records] == [0, 1, 2, 3, 4, 5]
    released = (final_health.get("closed") is True and final_health.get("initialized") is False and
                final_health.get("poisoned") is False and all(final_health.get(key) == 0 for key in (
                    "live_storages", "retained_storage_owners", "retirement_failures")))
    compared = complete and all(r["passed"] for r in records)
    return {"status": "PASS" if compared and released else "FAIL", "complete": complete,
            "execution_status": "PASS" if complete and released else "FAIL",
            "comparison_status": "PASS" if compared else "FAIL" if complete else "INCOMPLETE",
            "native_context_released": released}


def run_cases():
    # Same bounded primitive cases; additionally observe the new preparation-only
    # predicate on the actual current Runtime, including strict metadata refusal.
    supported = [(2,17,2), (6,448,1), (6,1500,1)]
    for heads,tokens,batch in supported:
        assert Graph.supports_sdpa_decode_f32(heads,tokens,batch), (heads,tokens,batch)
    try:
        Graph.supports_sdpa_decode_f32(2,1537,1)
    except mstar_pi._C.Rejected:
        invalid_query_rejected = True
    else:
        raise AssertionError("invalid SDPA geometry was treated as a capability result")
    records, held, saved = [], [], []
    work = mstar_pi.Stream()
    generator = torch.Generator().manual_seed(20261024)
    width, row_width = 64 * HEADS, 64 * HEADS + PADDING

    def factory():
        g = Graph(max_nodes=64)
        q = g.reshape(g.tensor("q", (width, BATCH)), (64, HEADS, 1, BATCH))
        k = g.view(g.tensor("k", (row_width, TOKENS, BATCH)),
                   (64, HEADS, TOKENS, BATCH), (4, 256, 4 * row_width, 4 * row_width * TOKENS))
        v = g.view(g.tensor("v", (row_width, TOKENS, BATCH)),
                   (64, HEADS, TOKENS, BATCH), (4, 256, 4 * row_width, 4 * row_width * TOKENS))
        mask = g.view(g.tensor("mask", (TOKENS, BATCH)), (TOKENS, 1, 1, BATCH),
                      (4, 4 * TOKENS, 4 * TOKENS, 4 * TOKENS))
        output = g.sdpa_decode_f32(q, k, v, mask, SCALE)
        g.output("y", g.reshape(output, (width, BATCH)))
        return g

    def stage(tensor):
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        host.copy_(tensor, non_blocking=True)
        event = mstar_pi.Event(); event.record(work)
        return host, event

    def upload(tensor):
        raw_cpu = torch.full((tensor.numel() + 16,), 31, dtype=tensor.dtype)
        raw_cpu[8:-8].view(tensor.shape).copy_(tensor)
        raw = raw_cpu.to("mstar_pi", non_blocking=True)
        return raw, raw[8:-8].view(tensor.shape), raw_cpu

    with torch.inference_mode():
        with work:
            program = prepare_graph(factory, {}, slots=1)
        assert program.info()["managed_sdpa_decode_nodes_per_slot"] == 1
        assert program.info()["managed_q8_dot_nodes_per_slot"] == 0
        for repeat in range(2):
            q = torch.randn((BATCH, width), generator=generator) / 8
            k = torch.full((BATCH, TOKENS, row_width), 31.)
            v = torch.full_like(k, 31.)
            k[..., :width] = torch.randn((BATCH, TOKENS, width), generator=generator) / 8
            v[..., :width] = torch.randn((BATCH, TOKENS, width), generator=generator) / 8
            mask = torch.zeros((BATCH, TOKENS))
            mask[0, 1:] = -float("inf")  # Different histories, no trimming by the operation.
            q64 = q.double().reshape(BATCH, HEADS, 64)
            k64 = k[..., :width].double().reshape(BATCH, TOKENS, HEADS, 64)
            v64 = v[..., :width].double().reshape(BATCH, TOKENS, HEADS, 64)
            scores = torch.einsum("bhd,bthd->bht", q64, k64) * SCALE + mask[:, None, :]
            expected = torch.einsum("bht,bthd->bhd", scores.softmax(-1), v64).float().reshape(BATCH, width)
            with work:
                uploads = {name: upload(value) for name, value in (("q", q), ("k", k), ("v", v), ("mask", mask))}
                raw_output = torch.full((BATCH * width + 16,), 47., device="mstar_pi")
                output = raw_output[8:-8].view(BATCH, width)
                lease = program.try_acquire()
                if lease is None: raise RuntimeError("settled attention probe did not return its slot")
                lease.submit({name: item[1] for name, item in uploads.items()}, {"y": output})
                host, done = stage(raw_output)
                input_observations = {name: stage(item[0]) for name, item in uploads.items()}
            done.synchronize()
            for _, event in input_observations.values(): event.synchronize()
            lease.close(); mstar_pi.collect()
            actual = host[8:-8].view(BATCH, width)
            rejection = None
            try: torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
            except AssertionError as error: rejection = str(error)
            guards = bool(torch.all(host[:8] == 47) and torch.all(host[-8:] == 47))
            readonly = all(torch.equal(value[0].view(torch.int32), uploads[name][2].view(torch.int32))
                           for name, value in input_observations.items())
            retained = True
            for old, snapshot in zip(held, saved):
                with work: old_host, old_done = stage(old)
                old_done.synchronize()
                retained = retained and torch.equal(old_host.view(torch.int32), snapshot.view(torch.int32))
            independent = all(output.data_ptr() != old.data_ptr() for old in held)
            held.append(output); saved.append(actual.clone())
            error = (actual - expected).abs().max().item()
            records.append({"case": "q1_f32_strided_odd_tokens_two_rows", "repeat": repeat,
                "shape": {"head_dim": 64, "heads": HEADS, "tokens": TOKENS, "batch": BATCH},
                "token_row_padding_floats": PADDING, "input_output_offset_bytes": 32,
                "comparison_passed": rejection is None, "comparison_error": rejection,
                "max_abs_error": error if math.isfinite(error) else None, "rtol": 3e-5, "atol": 3e-5,
                "input_readonly": readonly, "output_sentinels": guards,
                "retained_outputs_exact": retained, "independent_output_storage": independent,
                "passed": rejection is None and readonly and guards and retained and independent,
                "scope": "FP64 CPU operators for this primitive only; no model oracle or serving fallback"})
        program.close()

        # Bounded integration controls use the same explicit CPU equation and
        # completion/retention envelope. SET_ROWS aliases remain actual graph
        # predecessors; nonfinite cases pin full-domain masked-value behavior.
        for repeat, label, case_tokens, append, special in (
            (2, "set_rows_view_dependency", 17, True, None),
            (3, "maximum_token_domain", 1536, False, None),
            (4, "all_masked_explicit_equation_nan", 17, False, "all_masked"),
            (5, "masked_nonfinite_full_domain", 17, False, "masked_nonfinite"),
        ):
            def extra_factory():
                g = Graph(max_nodes=64)
                qg = g.reshape(g.tensor("q", (width, BATCH)), (64, HEADS, 1, BATCH))
                kg = g.tensor("k", (row_width, case_tokens, BATCH))
                vg = g.tensor("v", (row_width, case_tokens, BATCH))
                if append:
                    ids = g.tensor("ids", (1, BATCH), "i64")
                    kg = g.set_rows(g.tensor("new_k", (row_width, 1, BATCH)), ids, kg)
                    vg = g.set_rows(g.tensor("new_v", (row_width, 1, BATCH)), ids, vg)
                kg = g.view(kg, (64, HEADS, case_tokens, BATCH),
                            (4, 256, 4 * row_width, 4 * row_width * case_tokens))
                vg = g.view(vg, (64, HEADS, case_tokens, BATCH),
                            (4, 256, 4 * row_width, 4 * row_width * case_tokens))
                mg = g.view(g.tensor("mask", (case_tokens, BATCH)), (case_tokens, 1, 1, BATCH),
                            (4, 4 * case_tokens, 4 * case_tokens, 4 * case_tokens))
                g.output("y", g.reshape(g.sdpa_decode_f32(qg, kg, vg, mg, SCALE), (width, BATCH)))
                return g

            q = torch.randn((BATCH, width), generator=generator) / 8
            k = torch.full((BATCH, case_tokens, row_width), 31.)
            v = torch.full_like(k, 31.)
            k[..., :width] = torch.randn((BATCH, case_tokens, width), generator=generator) / 8
            v[..., :width] = torch.randn((BATCH, case_tokens, width), generator=generator) / 8
            mask = torch.zeros((BATCH, case_tokens))
            payloads = {"q": q, "k": k, "v": v, "mask": mask}
            expected_k, expected_v = k.clone(), v.clone()
            if append:
                ids = torch.tensor([[1], [case_tokens - 1]], dtype=torch.int64)
                new_k = torch.randn((BATCH, 1, row_width), generator=generator) / 4
                new_v = torch.randn((BATCH, 1, row_width), generator=generator) / 4
                payloads.update(ids=ids, new_k=new_k, new_v=new_v)
                for batch in range(BATCH):
                    expected_k[batch, ids[batch, 0]] = new_k[batch, 0]
                    expected_v[batch, ids[batch, 0]] = new_v[batch, 0]
                mask[0].fill_(-float("inf")); mask[0, ids[0, 0]] = 0
            elif special == "all_masked":
                mask.fill_(-float("inf"))
            elif special == "masked_nonfinite":
                mask.fill_(-float("inf")); mask[:, 0] = 0
                k[0, 7, :width] = float("nan")
                v[1, 7, :width] = float("inf")
                expected_k, expected_v = k.clone(), v.clone()
            q64 = q.double().reshape(BATCH, HEADS, 64)
            k64 = expected_k[..., :width].double().reshape(BATCH, case_tokens, HEADS, 64)
            v64 = expected_v[..., :width].double().reshape(BATCH, case_tokens, HEADS, 64)
            scores = torch.einsum("bhd,bthd->bht", q64, k64) * SCALE + mask[:, None, :]
            expected = torch.einsum("bht,bthd->bhd", scores.softmax(-1), v64).float().reshape(BATCH, width)
            with work:
                program = prepare_graph(extra_factory, {}, slots=1)
                uploads = {name: upload(value) for name, value in payloads.items()}
                raw_output = torch.full((BATCH * width + 16,), 47., device="mstar_pi")
                output = raw_output[8:-8].view(BATCH, width)
                lease = program.try_acquire()
                if lease is None: raise RuntimeError("attention integration control slot unavailable")
                lease.submit({name: value[1] for name, value in uploads.items()}, {"y": output})
                host, done = stage(raw_output)
                observations = {name: stage(value[0]) for name, value in uploads.items()}
            done.synchronize()
            for _, event in observations.values(): event.synchronize()
            lease.close(); mstar_pi.collect()
            actual = host[8:-8].view(BATCH, width)
            rejection = None
            if special:
                # Preserve exact NaN classification, then apply unchanged3e-5
                # bounds to any finite elements. No TorchSDPA zero convention.
                classification = torch.equal(torch.isnan(actual), torch.isnan(expected))
                finite = torch.isfinite(expected)
                try: torch.testing.assert_close(actual[finite], expected[finite], rtol=3e-5, atol=3e-5)
                except AssertionError as error: rejection = str(error)
                if not classification: rejection = "explicit equation NaN classification differs"
            else:
                try: torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
                except AssertionError as error: rejection = str(error)
            physical_expected = {name: value[2] for name, value in uploads.items()}
            if append:
                for name, changed in (("k", expected_k), ("v", expected_v)):
                    physical_expected[name] = uploads[name][2].clone()
                    physical_expected[name][8:-8].view(changed.shape).copy_(changed)
            inputs_ok = all(torch.equal(value[0].view(torch.int32), physical_expected[name].view(torch.int32))
                            for name, value in observations.items())
            guards = bool(torch.all(host[:8] == 47) and torch.all(host[-8:] == 47))
            retained = True
            for old, snapshot in zip(held, saved):
                with work: old_host, old_done = stage(old)
                old_done.synchronize()
                retained = retained and torch.equal(old_host.view(torch.int32), snapshot.view(torch.int32))
            independent = all(output.data_ptr() != old.data_ptr() for old in held)
            held.append(output); saved.append(actual.clone())
            info = program.info()
            retired = info["active_slots"] == 0 and info["submissions"] == info["retirements"] == 1
            program.close()
            records.append({"case": label, "repeat": repeat,
                "shape": {"head_dim": 64, "heads": HEADS, "tokens": case_tokens, "batch": BATCH},
                "set_rows_dependency": append, "input_state_expected_bits": inputs_ok,
                "comparison_passed": rejection is None, "comparison_error": rejection,
                "expected_nan_elements": int(torch.isnan(expected).sum()),
                "actual_nan_elements": int(torch.isnan(actual).sum()), "rtol": 3e-5, "atol": 3e-5,
                "output_sentinels": guards, "retained_outputs_exact": retained,
                "independent_output_storage": independent, "retired": retired,
                "passed": rejection is None and inputs_ok and guards and retained and independent and retired,
                "scope": "same explicit FP64 primitive equation; state dependency/full-domain controls; no model oracle"})
        mstar_pi.synchronize(); mstar_pi.collect()
    records[0]["capability_query_control"] = {"supported": supported,
        "invalid_metadata_rejected": invalid_query_rejected,
        "scope": "preparation only; current Runtime DRM identity; no tensor storage or submission"}
    return records


class _Receipt:
    def __init__(self): self.value = None
    def send(self, value): self.value = value
    def close(self): pass


def one_process(config, connection=None):
    receipt = _Receipt()
    try:
        retained_gate.one_process(config, receipt, cases=run_cases)
        result = receipt.value
        final_health = mstar_pi.health()
        result.update(outcome(result["records"], final_health))
        result.update(final_native_health=final_health,
            scope="initial bounded generic Q1 attention mechanism only; not full primitive/model qualification",
            launcher_exit_semantics="zero requires completed execution and closed healthy context; numerical FAIL stays FAIL")
        if connection: connection.send(result)
        else: print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return result
    except BaseException:
        if connection and receipt.value is not None: connection.send(receipt.value)
        raise
    finally:
        if connection: connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shader", required=True)
    parser.add_argument("--render-node", default="/dev/dri/renderD128")
    parser.add_argument("--wait-timeout-ms", type=int, default=30000)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--skip-spawn", action="store_true")
    args = parser.parse_args()
    if not hasattr(Graph, "sdpa_decode_f32"):
        raise RuntimeError("explicit Q1 attention API unavailable; no native context initialized")
    config = {"shader": str(Path(args.shader).resolve(strict=True)), "render_node": args.render_node,
              "wait_timeout_ms": args.wait_timeout_ms, "validation": args.validation}
    parent = one_process(config)
    if parent["execution_status"] != "PASS": raise SystemExit(1)
    if not args.skip_spawn:
        context = mp.get_context("spawn"); reader, writer = context.Pipe(duplex=False)
        child = context.Process(target=one_process, args=(config, writer)); child.start(); writer.close()
        result = reader.recv(); reader.close()
        print(json.dumps({"spawned_worker": result}, sort_keys=True, allow_nan=False), flush=True)
        if result.get("health", {}).get("poisoned") or "drain_error" in result:
            retained_gate.hold_uncertain_context({"status": "CHILD_HELD", "child_pid": child.pid, "child": result})
        child.join()
        if child.exitcode or result.get("execution_status") != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
