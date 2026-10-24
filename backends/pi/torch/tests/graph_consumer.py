"""Real multi-op ggml graph gate under the existing Pi admission/containment.

CPU arithmetic is an independent test oracle, never an inference fallback.
Run the unchanged Stage 1 and Stage 2 gates in addition to this new graph gate.
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
import torch.nn.functional as F
import mstar_pi
from mstar_pi.ggml import Graph, prepare_graph
from consumer_chain import hold_uncertain_context


def run_cases() -> list[dict]:
    if not __debug__:
        raise RuntimeError("graph gate needs assertion oracles")
    records = []
    work = mstar_pi.Stream()
    producer = mstar_pi.Stream()

    def stage(t):
        host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        host.copy_(t, non_blocking=True)
        done = mstar_pi.Event(); done.record(work)
        return host, done

    def factory():
        g = Graph(max_nodes=128)
        k = g.tensor("kernel", (3, 2, 8))
        x = g.tensor("input", (17, 2))
        bias = g.tensor("bias", (8,))
        gain = g.tensor("gain", (8,))
        mask = g.tensor("mask", (8, 17))
        state = g.tensor("state", (8, 24))
        ids64 = g.tensor("ids", (17,), "i64")
        col = g.im2col(k, x, padding=1)
        hidden = g.add(g.mul_mat(g.reshape(k, (6, 8)), col), bias)
        hidden = g.mul(g.norm(hidden, eps=1e-5), gain)
        hidden = g.gelu_erf(hidden)
        probs = g.softmax(hidden, mask, scale=.5)
        changed = g.set_rows(probs, ids64, state)
        gathered = g.get_rows(changed, g.cast(ids64, "i32"))
        # Real transpose/contiguous materialization before a matrix operation.
        matrix = g.cont(g.permute(gathered, (1, 0, 2, 3)))
        gram = g.mul_mat(matrix, matrix)
        g.output("probs", gathered)
        g.output("gram", gram)
        g.output("tokens", g.cast(g.argmax(gathered), "i64"))
        g.effect(changed)
        return g

    with torch.inference_mode():
        gen = torch.Generator().manual_seed(1049)
        k_cpu = torch.randn((8, 2, 3), generator=gen) / 8
        x_cpu = torch.randn((2, 17), generator=gen) / 4
        b_cpu = torch.randn((8,), generator=gen) / 16
        gain_cpu = torch.linspace(.5, 1.5, 8)
        mask_cpu = torch.zeros((17, 8)); mask_cpu[:, -1] = -float("inf")
        ids_cpu = torch.arange(17, dtype=torch.int64) + 3
        with producer:
            constants = {"kernel": k_cpu.to("mstar_pi", non_blocking=True), "bias": b_cpu.to("mstar_pi", non_blocking=True), "gain": gain_cpu.to("mstar_pi", non_blocking=True)}
            mask = mask_cpu.to("mstar_pi", non_blocking=True)
        with work:
            program = prepare_graph(factory, constants, slots=2)
            lease0, lease1 = program.try_acquire(), program.try_acquire()
            assert lease0 is not None and lease1 is not None
            assert not program.available() and program.try_acquire() is None
            lease0.close(); lease0.close(); lease1.close()
            assert program.available()
            raw_x = torch.full((2 * 17 + 64,), -99., device="mstar_pi")
            x = raw_x[32:-32].view(2, 17)
            x.copy_(x_cpu)
            ids = ids_cpu.to("mstar_pi")
            state = torch.full((24, 8), -13., device="mstar_pi")
            raw_out = torch.full((17 * 8 + 64,), -77., device="mstar_pi")
            out = raw_out[32:-32].view(17, 8)
            gram = torch.empty((8, 8), device="mstar_pi")
            tokens = torch.empty((17,), dtype=torch.int64, device="mstar_pi")
            lease = program.try_acquire(); assert lease is not None
            lease.submit({"input": x, "ids": ids, "mask": mask, "state": state},
                         {"probs": out, "gram": gram, "tokens": tokens})
            lease.close()  # No premature slot reuse after ownership transfer.
            retained = out.clone()
            host, done = stage(retained)
            tokens_host, token_done = stage(tokens)
            gram_host, gram_done = stage(gram)
            first_ready = done.query()
        gram_done.synchronize(); mstar_pi.collect()
        expected = F.conv1d(x_cpu.unsqueeze(0), k_cpu, b_cpu, padding=1).squeeze(0).T
        expected = F.layer_norm(expected, (8,), eps=1e-5) * gain_cpu
        expected = F.gelu(expected, approximate="none")
        expected = torch.softmax(expected * .5 + mask_cpu, dim=-1)
        torch.testing.assert_close(host, expected, rtol=5e-5, atol=5e-5)
        torch.testing.assert_close(gram_host, expected.T @ expected, rtol=5e-5, atol=5e-5)
        assert torch.equal(tokens_host, expected.argmax(-1))
        actual_state = state.cpu()
        torch.testing.assert_close(actual_state[3:20], expected, rtol=5e-5, atol=5e-5)
        assert torch.all(actual_state[:3] == -13.) and torch.all(actual_state[20:] == -13.)
        assert program.info()["active_slots"] == 0
        assert program.info()["submissions"] == program.info()["retirements"] == 1
        raw_host = raw_out.cpu()
        assert torch.all(raw_host[:32] == -77.) and torch.all(raw_host[-32:] == -77.)
        records.append({"case": "multi_op_whisper_primitives_state_alias_immediate_clone", "first_event_ready": first_ready,
                        "max_abs_error": (host-expected).abs().max().item(), "guards": True,
                        "graph": program.info(), "host_interstage_wait": False})

        # Reuse the same mutable graph/state with another invocation while
        # preserving the old completed event and independently owned result.
        with work:
            next_x = (-x_cpu).to("mstar_pi")
            next_out = torch.empty_like(out)
            next_gram = torch.empty_like(gram)
            next_tokens = torch.empty_like(tokens)
            lease = program.try_acquire(); assert lease is not None
            lease.submit({"input": next_x, "ids": ids, "mask": mask, "state": state},
                         {"probs": next_out, "gram": next_gram, "tokens": next_tokens})
            next_host, next_done = stage(next_out)
            del next_x, constants, mask
        next_done.synchronize(); mstar_pi.collect()
        assert done.query() and token_done.query()
        torch.testing.assert_close(retained.cpu(), expected, rtol=5e-5, atol=5e-5)
        expected2 = F.conv1d((-x_cpu).unsqueeze(0), k_cpu, b_cpu, padding=1).squeeze(0).T
        expected2 = torch.softmax(F.gelu(F.layer_norm(expected2,(8,),eps=1e-5)*gain_cpu) * .5 + mask_cpu, -1)
        torch.testing.assert_close(next_host, expected2, rtol=5e-5, atol=5e-5)
        records.append({"case": "prepared_slots_retained_output_old_event_released_python_owners", "pass": True})
        assert program.info()["submissions"] == program.info()["retirements"] == 2
        program.close(); program.close()
        assert not program.available()

        # Ties across reduction lanes, all -infinity, signed zero and NaNs must
        # agree with Torch's first-index argmax. Casting preserves I64 token ABI.
        def argmax_graph():
            g = Graph(max_nodes=16); a = g.tensor("x", (1025, 5))
            g.output("ids", g.cast(g.argmax(a), "i64")); return g
        logits = torch.full((5, 1025), -float("inf"))
        logits[0, 999] = logits[0, 2] = 7.
        logits[2, 800] = logits[2, 33] = float("nan")
        logits[3, 1000] = 0.; logits[3, 1] = -0.
        logits[4, 1024] = 11.
        with work:
            ap = prepare_graph(argmax_graph, {}, slots=1)
            a = logits.to("mstar_pi"); result = torch.empty((5,),dtype=torch.int64,device="mstar_pi")
            lease = ap.try_acquire(); lease.submit({"x":a},{"ids":result})
            host_ids, ids_done = stage(result)
        ids_done.synchronize(); assert torch.equal(host_ids,logits.argmax(-1))
        ap.close()
        records.append({"case": "argmax_first_index_nan_inf_zero_i64", "exact": True})

        # A rejected lease cannot have submitted, but prior work still drains.
        def cast_graph():
            g=Graph(max_nodes=16); a=g.tensor("x",(8,),"i64")
            g.output("y",g.cast(g.cast(a,"i32"),"i64")); return g
        values=torch.tensor([0,1,-1,2147483647,-2147483648,42,-117,1024],dtype=torch.int64)
        with work:
            cp=prepare_graph(cast_graph,{},slots=1)
            src=values.to("mstar_pi"); dst=torch.empty_like(src)
            lease=cp.try_acquire()
            try:
                lease.submit({"x":src},{"y":src})
            except mstar_pi.Rejected:
                pass
            else:
                raise AssertionError("destructive output alias accepted")
            assert cp.available() and cp.info()["submissions"] == 0
            lease=cp.try_acquire(); lease.submit({"x":src},{"y":dst})
            result_cpu, copy_done=stage(dst)
        copy_done.synchronize(); assert torch.equal(result_cpu,values)
        cp.close()
        records.append({"case":"integer_word_conversion_rejection_and_prior_work_drain","exact":True})
        mstar_pi.synchronize(); mstar_pi.collect()
    return records


def one_process(config, connection=None, cases=run_cases):
    started=time.monotonic()
    try:
        mstar_pi.initialize(copy_shader=config["shader"],render_node=config["render_node"],
                            wait_timeout_ms=config["wait_timeout_ms"],validation=config["validation"])
        records=cases(); gc.collect(); mstar_pi.collect(); health=mstar_pi.health()
        assert health["live_storages"] == health["retirement_failures"] == health["retained_storage_owners"] == 0, health
        mstar_pi.shutdown()
        result={"status":"PASS","pid":os.getpid(),"records":records,"health_before_shutdown":health,
                "elapsed_seconds":time.monotonic()-started,"scope":"real managed graph primitives; no performance claim"}
        if connection:connection.send(result)
        else:print(json.dumps(result,sort_keys=True),flush=True)
    except BaseException as error:
        result={"status":"FAIL","error":repr(error),"traceback":traceback.format_exc(),"health":mstar_pi.health(),"pid":os.getpid()}
        if result["health"]["poisoned"]:
            if connection:connection.send(result)
            hold_uncertain_context(result)
        if mstar_pi.is_initialized():
            try:mstar_pi.synchronize()
            except BaseException as drain_error:
                result["drain_error"]=repr(drain_error);result["health"]=mstar_pi.health()
                if connection:connection.send(result)
                hold_uncertain_context(result)
        result["accepted_work_settled"]=True
        if connection:connection.send(result)
        else:print(json.dumps(result,sort_keys=True),flush=True)
        raise
    finally:
        if connection:connection.close()


def main(cases=run_cases):
    parser=argparse.ArgumentParser()
    parser.add_argument("--shader",required=True);parser.add_argument("--render-node",default="/dev/dri/renderD128")
    parser.add_argument("--wait-timeout-ms",type=int,default=30000);parser.add_argument("--validation",action="store_true")
    parser.add_argument("--skip-spawn",action="store_true");args=parser.parse_args()
    config={"shader":str(Path(args.shader).resolve(strict=True)),"render_node":args.render_node,
            "wait_timeout_ms":args.wait_timeout_ms,"validation":args.validation}
    one_process(config, cases=cases)
    if not args.skip_spawn:
        ctx=mp.get_context("spawn");reader,writer=ctx.Pipe(duplex=False)
        child=ctx.Process(target=one_process,args=(config,writer,cases));child.start();writer.close()
        result=reader.recv();reader.close();print(json.dumps({"spawned_worker":result},sort_keys=True),flush=True)
        if result.get("health",{}).get("poisoned") or "drain_error" in result:
            hold_uncertain_context({"status":"CHILD_HELD","child_pid":child.pid,"child":result})
        child.join()
        if child.exitcode or result["status"] != "PASS":raise SystemExit(1)


if __name__ == "__main__":main()
