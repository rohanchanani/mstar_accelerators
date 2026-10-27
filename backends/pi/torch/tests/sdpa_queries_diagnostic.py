"""Small matched native attention-operation diagnostic, outside acceleration qualification.

Graph construction, leases and event observation follow graph_consumer.py.
Managed initialization/drain/release and retained roots use the existing M*
worker and qualification session. This script defines no model reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

OPERATIONS = tuple((f"{mode}_q{queries}_t{tokens}", mode, tokens, queries, False)
                   for queries in (1, 4) for tokens in (448, 1500) for mode in ("old", "fused")) + (
                       ("fused_q4_padded_state_producers", "fused", 17, 4, True),)
HEADS, HEAD_DIM = 6, 64



def outcome(record, failure, released):
    operations = record["operations"]
    complete = tuple(operation["operation"] for operation in operations) == tuple(
        operation[0] for operation in OPERATIONS) and all(
            len(operation["timings"]) == len(operation["comparisons"]) == 4
            for operation in operations)
    comparisons_pass = complete and all(check["passed"] for operation in operations
                                        for check in operation["comparisons"])
    return dict(status="PASS" if comparisons_pass and failure is None and released else "FAIL",
                execution_status="PASS" if complete and released and failure is None else "FAIL",
                comparison_status="PASS" if comparisons_pass else "FAIL" if complete else "INCOMPLETE",
                native_owner_released=released, complete=complete, driver_error=failure)


def run_operation(worker, name, mode, *, tokens, queries, state_case):
    import torch
    import mstar_pi
    from mstar_pi.ggml import Graph, prepare_graph
    from mstar_accelerators.implementations.whisper.pi_v3d.graph import _attention
    from mstar_accelerators.testing.whisper import _observe

    batch = 2 if state_case else 1
    padding = 8 if state_case else 0
    width, row_width, scale = HEADS * HEAD_DIM, HEADS * HEAD_DIM + padding, HEAD_DIM ** -0.5
    supported = Graph.supports_sdpa_queries_f32(HEADS, tokens, queries, batch)
    if not supported:
        raise RuntimeError("new attention primitive unsupported on actual current runtime")
    generator = torch.Generator().manual_seed(20261024)
    # Reset per operation: old/new algorithms receive bit-identical tensors,
    # masks and physical views. Main cases use the exact accepted old builder.
    q_base = torch.randn((batch, queries, width), generator=generator) / 8
    k_base = torch.full((batch, tokens, row_width), 31.)
    v_base = torch.full_like(k_base, 31.)
    k_base[..., :width] = torch.randn((batch, tokens, width), generator=generator) / 8
    v_base[..., :width] = torch.randn((batch, tokens, width), generator=generator) / 8
    masks = torch.zeros((batch, queries, tokens))
    if tokens == 448:
        for query in range(queries): masks[:, query, 197 + query:] = -float("inf")
    payloads = {"k": k_base, "v": v_base, "mask": masks}
    expected_k, expected_v = k_base.clone(), v_base.clone()
    if state_case:
        write_ids = torch.stack((torch.arange(queries), torch.arange(tokens-queries, tokens))).long()
        read_ids = torch.arange(tokens-1, -1, -1).repeat(batch, 1).int()
        new_k = torch.randn((batch, queries, row_width), generator=generator) / 4
        new_v = torch.randn((batch, queries, row_width), generator=generator) / 4
        # Explicit full-domain NaN equation for query3 and masked nonfinite V.
        masks[0, 3].fill_(-float("inf"))
        masks[1, 2].fill_(-float("inf")); masks[1, 2, 0] = 0
        v_base[1, 0, :width] = float("inf")
        expected_v = v_base.clone()
        for b in range(batch):
            expected_k[b, write_ids[b]] = new_k[b]
            expected_v[b, write_ids[b]] = new_v[b]
        payloads.update(ids=write_ids, read_ids=read_ids, new_k=new_k, new_v=new_v)
    query_factors = (1., -1., .5, -.5)
    query_payloads = [q_base * factor for factor in query_factors]
    if not state_case:
        payloads = {key: value[0] for key, value in payloads.items()}
        query_payloads = [value[0] for value in query_payloads]
    kd, vd = expected_k[..., :width].double(), expected_v[..., :width].double()
    if state_case:
        kd = torch.stack([kd[b, read_ids[b].long()] for b in range(batch)])
        vd = torch.stack([vd[b, read_ids[b].long()] for b in range(batch)])
    kd, vd = (x.reshape(batch, tokens, HEADS, HEAD_DIM) for x in (kd, vd))
    expected = []
    for q in query_payloads:
        qd = q.double().reshape(batch, queries, HEADS, HEAD_DIM)
        scores = torch.einsum("bqhd,bthd->bhqt", qd, kd) * scale + masks[:, None]
        result = torch.einsum("bhqt,bthd->bqhd", scores.softmax(-1), vd).float()
        expected.append(result.reshape(batch, queries, width) if state_case else result.reshape(queries, width))

    def factory():
        g = Graph(max_nodes=80)
        q = g.tensor("q", (width, queries, batch) if state_case else (width, queries))
        k = g.tensor("k", (row_width, tokens, batch) if state_case else (width, tokens))
        v = g.tensor("v", (row_width, tokens, batch) if state_case else (width, tokens))
        mask = g.tensor("mask", (tokens, queries, batch) if state_case else (tokens, queries))
        if state_case:
            ids = g.tensor("ids", (queries, batch), "i64")
            reads = g.tensor("read_ids", (tokens, batch), "i32")
            # These actual returned aliases, rather than another input alias of
            # the allocation, are predecessors of GET_ROWS and the new op.
            k = g.get_rows(g.set_rows(g.tensor("new_k", (row_width, queries, batch)), ids, k), reads)
            v = g.get_rows(g.set_rows(g.tensor("new_v", (row_width, queries, batch)), ids, v), reads)
        if mode == "old":
            y = _attention(g, q, k, v, heads=HEADS, width=width, queries=queries, keys=tokens, mask=mask)
        else:
            q = g.reshape(q, (HEAD_DIM, HEADS, queries, batch))
            strides = (4, HEAD_DIM*4, row_width*4, row_width*tokens*4)
            k = g.view(k, (HEAD_DIM, HEADS, tokens, batch), strides)
            v = g.view(v, (HEAD_DIM, HEADS, tokens, batch), strides)
            mask = g.view(mask, (tokens, queries, 1, batch), (4, tokens*4, tokens*queries*4, tokens*queries*4))
            y = g.sdpa_queries_f32(q, k, v, mask, scale)
            y = g.reshape(y, (width, queries, batch) if state_case else (width, queries))
        g.output("y", y)
        return g

    def upload(cpu):
        guarded_cpu = torch.full((cpu.numel()+16,), 31, dtype=cpu.dtype)
        guarded_cpu[8:-8].view(cpu.shape).copy_(cpu)
        raw = guarded_cpu.to("mstar_pi", non_blocking=True)
        return raw, raw[8:-8].view(cpu.shape), guarded_cpu

    preparation_start = time.perf_counter_ns()
    with worker.stream:
        uploaded = {key: upload(value) for key,value in payloads.items()}
        uploaded_queries = [upload(value) for value in query_payloads]
        inputs = {key: item[1] for key,item in uploaded.items()}
        q_values = [item[1] for item in uploaded_queries]
        raw_outputs = [torch.full((batch*queries*width+16,), 47., device="mstar_pi")
                       for _ in range(worker.warmups+worker.repeats)]
        shape = (batch,queries,width) if state_case else (queries,width)
        outputs = [raw[8:-8].view(shape) for raw in raw_outputs]
        program = prepare_graph(factory, {}, slots=1)
    worker.session.prepared.append(program)
    worker.session.retained.extend([uploaded, uploaded_queries, raw_outputs, outputs])
    worker.stream.synchronize(); mstar_pi.collect()
    preparation_seconds = (time.perf_counter_ns()-preparation_start)*1e-9
    plan = program.info()
    if plan["managed_sdpa_queries_nodes_per_slot"] != (1 if mode == "fused" else 0):
        raise RuntimeError(f"actual prepared attention path mismatch: {plan}")
    timings = []
    for index, output in enumerate(outputs):
        if worker.stop_requested: raise InterruptedError("diagnostic stopped between settled native calls")
        mstar_pi.collect()
        lease = program.try_acquire()
        if lease is None: raise RuntimeError("completed diagnostic invocation did not return its slot")
        event = mstar_pi.Event()
        with worker.stream:
            start = time.perf_counter_ns()
            lease.submit(dict(inputs, q=q_values[index]), {"y": output})
            submitted = time.perf_counter_ns()
            event.record(worker.stream)
            recorded = time.perf_counter_ns()
        event.synchronize()
        completed = time.perf_counter_ns()
        lease.close(); mstar_pi.collect()
        timings.append({"index": index, "phase": "warmup" if index<worker.warmups else "warm",
            "submit_seconds": (submitted-start)*1e-9, "event_record_seconds": (recorded-submitted)*1e-9,
            "completion_wait_seconds": (completed-recorded)*1e-9, "total_seconds": (completed-start)*1e-9,
            "boundary": "native submit through real same-stream completion; preparation/acquisition/oracle/readback excluded"})
    with worker.stream:
        observed = _observe({str(i): raw for i,raw in enumerate(raw_outputs)}, worker.stream)
        observed_inputs = _observe({**{key: item[0] for key,item in uploaded.items()},
                                   **{f"q{i}": item[0] for i,item in enumerate(uploaded_queries)}}, worker.stream)
    physical_expected = {key: item[2] for key,item in uploaded.items()}
    physical_expected.update({f"q{i}": item[2] for i,item in enumerate(uploaded_queries)})
    if state_case:
        for key, value in (("k", expected_k), ("v", expected_v)):
            physical_expected[key] = physical_expected[key].clone()
            physical_expected[key][8:-8].view(value.shape).copy_(value)
    input_bits = all(torch.equal(observed_inputs[key].view(torch.int32), value.view(torch.int32))
                     for key,value in physical_expected.items())
    independent = len({output.data_ptr() for output in outputs}) == len(outputs)
    comparisons = []
    for key,raw in observed.items():
        index = int(key); actual = raw[8:-8].view(shape); ref = expected[index]
        error = None
        classification = (torch.equal(torch.isnan(actual),torch.isnan(ref)) and
                          torch.equal(torch.isposinf(actual),torch.isposinf(ref)) and
                          torch.equal(torch.isneginf(actual),torch.isneginf(ref)))
        finite = torch.isfinite(ref)
        try: torch.testing.assert_close(actual[finite], ref[finite], rtol=3e-5, atol=3e-5)
        except AssertionError as rejection: error = str(rejection)
        if not classification: error = "explicit equation nonfinite classification differs"
        guards = bool(torch.all(raw[:8]==47) and torch.all(raw[-8:]==47))
        maximum = (actual[finite]-ref[finite]).abs().max().item() if finite.any() else 0.
        comparisons.append({"index": index, "passed": error is None and guards and input_bits and independent,
            "error": error, "max_finite_abs_error": maximum if math.isfinite(maximum) else None,
            "nonfinite_classification_exact": classification, "expected_nan_elements": int(torch.isnan(ref).sum()),
            "output_guards": guards, "input_and_expected_state_bits_exact": input_bits,
            "independent_retained_storage": independent})
    info = program.info()
    if info["active_slots"] != 0 or not (info["submissions"] == info["retirements"] == 4):
        raise RuntimeError(f"diagnostic work did not retire: {info}")
    program.close()
    return {"operation": name, "algorithm": mode, "shape": {"D":64,"H":HEADS,"T":tokens,"B":batch,"Q":queries},
            "actual_capability_query": supported, "set_rows_get_rows_producer_chain": state_case,
            "token_padding_floats": padding, "binding_offsets_bytes": {**{key:8*value.element_size() for key,value in payloads.items()}, "q":32, "output":32},
            "input_shapes": {**{key:list(value.shape) for key,value in payloads.items()}, "q":list(query_payloads[0].shape)},
            "input_strides_bytes": {**{key:[n*value.element_size() for n in value.stride()] for key,value in payloads.items()},
                                    "q":[n*4 for n in query_payloads[0].stride()]},
            "query_factors": query_factors, "preparation_seconds": preparation_seconds,
            "timings": timings, "comparisons": comparisons, "graph": info,
            "rtol":3e-5,"atol":3e-5,
            "oracle":"same retained primitive explicit FP64 score/softmax/weighted-value equation cast F32; no model oracle"}


class DiagnosticWorker:
    def __init__(self, *, worker_id, device, runtime_extension, record, warmups, repeats):
        import torch
        import mstar_pi
        from mstar_accelerators.testing.whisper import QualificationSession

        del worker_id, device
        self.extension, self.record = runtime_extension, record
        self.warmups, self.repeats = warmups, repeats
        self.stop_requested = False
        self.session = QualificationSession()
        self.stream = mstar_pi.Stream()
        torch.set_num_threads(4)

    def request_stop(self):
        self.stop_requested = True

    def run(self):
        import torch
        with torch.inference_mode():
            for name, mode, tokens, queries, state_case in OPERATIONS:
                self.record["operations"].append(run_operation(self, name, mode, tokens=tokens, queries=queries, state_case=state_case))

    def close(self):
        self.extension.drain()
        self.session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--native-cpp", type=Path, required=True)
    parser.add_argument("--expected-shader-sha256", required=True)
    parser.add_argument("--expected-cpp-sha256", required=True)
    parser.add_argument("--shader", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_tree.resolve(strict=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != args.expected_head:
        raise RuntimeError(f"source identity mismatch: expected {args.expected_head}, found {head}")
    native_cpp = args.native_cpp.resolve(strict=True)
    native_cpp_sha256 = hashlib.sha256(native_cpp.read_bytes()).hexdigest()
    if native_cpp_sha256 != args.expected_cpp_sha256:
        raise RuntimeError(f"native CPP identity mismatch: {native_cpp_sha256}")
    attention_shader = source / "backends/pi/third_party/ggml/src/ggml-vulkan/vulkan-shaders/managed_sdpa_queries_f32.comp"
    attention_shader_sha256 = hashlib.sha256(attention_shader.read_bytes()).hexdigest()
    if attention_shader_sha256 != args.expected_shader_sha256:
        raise RuntimeError(f"attention shader source identity mismatch: {attention_shader_sha256}")
    sys.path[:0] = [str(source / "src"), str(source)]

    import mstar_pi
    from mstar.runtime_extensions import run_managed_worker
    from mstar_accelerators.backends.pi.runtime import PiRuntime

    extension_path = Path(mstar_pi._C.__file__).resolve()
    extension_sha256 = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    if extension_sha256 != args.expected_extension_sha256:
        raise RuntimeError(f"loaded extension identity mismatch: {extension_sha256}")
    record = {"scope": "matched old/fused attention primitive timing and controls only; not acceleration qualification",
              "source_head": head, "source_tree": str(source),
              "native_cpp_path": str(native_cpp), "native_cpp_sha256": native_cpp_sha256,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "attention_shader_source": str(attention_shader), "attention_shader_sha256": attention_shader_sha256,
              "extension_path": str(extension_path),
              "extension_sha256": extension_sha256,
              "warmups": 1, "repeats": 3, "operations": []}
    extension = PiRuntime({"accelerations": str(source / "accelerations"),
                           "native": {"copy_shader": str(args.shader.resolve(strict=True)),
                                      "wait_timeout_ms": 180000, "validation": True}})
    context = extension.admission_context()
    owner = extension.reserve_worker(context, "attention-operation-diagnostic")
    failure = None
    try:
        run_managed_worker(DiagnosticWorker, {
            "worker_id": "attention-operation-diagnostic", "device": "mstar_pi:0",
            "record": record, "warmups": 1, "repeats": 3,
        }, extension, owner)
    except Exception as error:
        if owner.status() != "released":
            raise
        failure = f"{type(error).__name__}: {error}"
    record.update(outcome(record, failure, owner.status() == "released"))
    record["launcher_exit_semantics"] = "zero only for complete execution and released owner; inspect unchanged comparisons for numerical acceptance"
    context.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return 0 if record["execution_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
