"""Actual ggml metadata, public graph interface and no implicit initialization.

No Pi device is opened. This complements target numerics/lifetime gates; it does
not emulate Vulkan or qualify native execution on this machine.
"""
from __future__ import annotations
import json
import multiprocessing as mp
import torch
import mstar_pi
from mstar_pi.ggml import Graph, PreparedGraph


def preflight() -> dict:
    if not __debug__:
        raise RuntimeError("preflight needs assertion oracles")
    before = mstar_pi.health()
    assert not mstar_pi.is_initialized()
    g = Graph(max_nodes=64)
    kernel = g.tensor("kernel", (3, 2, 4))
    x = g.tensor("input", (9, 2))
    col = g.im2col(kernel, x, padding=1)
    assert col.ne == [6, 9, 1, 1]
    hidden = g.mul_mat(g.reshape(kernel, (6, 4)), col)
    hidden = g.gelu_erf(g.norm(hidden, 1e-5))
    softmax = g.softmax(g.scale(hidden, .25))
    indices = g.tensor("indices", (9,), "i64")
    state = g.tensor("state", (4, 16))
    changed = g.set_rows(softmax, indices, state)
    gathered = g.get_rows(changed, g.cast(indices, "i32"))
    g.output("hidden", g.cont(g.permute(gathered, (1, 0, 2, 3))))
    g.output("tokens", g.cast(g.argmax(gathered), "i64"))
    g.effect(changed)
    assert g.info()["outputs"] == {"hidden": [9, 4, 1, 1], "tokens": [9, 1, 1, 1]}
    sample = Graph(max_nodes=32)
    logits = sample.tensor("logits", (51864, 3))
    seeds = sample.tensor("seeds", (3,), "i64")
    offsets = sample.tensor("offsets", (3,), "i64")
    temperatures = sample.tensor("temperatures", (3,))
    sample.output("tokens", sample.categorical(logits, seeds, offsets, temperatures))
    assert sample.info()["outputs"] == {"tokens": [3, 1, 1, 1]}
    assert sample.allocation_plan(256)["backend_scratch_bytes_per_slot"] == 0
    bad_seeds = sample.tensor("bad_seeds", (3,), "i32")
    bad_temperatures = sample.tensor("bad_temperatures", (2,))
    rejected = []
    for name, call in (
        ("cross_graph", lambda: Graph().scale(x, 2.)),
        ("invalid_reshape", lambda: g.reshape(x, (17,))),
        ("invalid_permutation", lambda: g.permute(x, (0, 0, 2, 3))),
        ("invalid_view", lambda: g.view(x, (9, 3), (4, 36))),
        ("duplicate_name", lambda: g.tensor("input", (9, 2))),
        ("categorical_seed_dtype", lambda: sample.categorical(logits, bad_seeds, offsets, temperatures)),
        ("categorical_temperature_rows", lambda: sample.categorical(logits, seeds, offsets, bad_temperatures)),
        ("categorical_cross_graph", lambda: sample.categorical(logits, seeds, indices, temperatures)),
        ("categorical_noncontiguous", lambda: sample.categorical(sample.permute(logits, (1, 0, 2, 3)), seeds, offsets, temperatures)),
        ("prepare_without_initialization", lambda: PreparedGraph(lambda: g, {}, 1)),
    ):
        try:
            call()
        except (RuntimeError, mstar_pi.Rejected):
            rejected.append(name)
        else:
            raise AssertionError(f"{name} accepted")
    assert mstar_pi.health() == before
    return {"status": "PASS", "scope": "actual ggml metadata, no target execution", "graph": g.info(),
            "categorical": sample.info(), "rejected": rejected}


def child(out):
    try:
        out.send(preflight())
    finally:
        out.close()


def main():
    print(json.dumps(preflight(), sort_keys=True), flush=True)
    ctx = mp.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=child, args=(writer,))
    proc.start(); writer.close()
    result = reader.recv(); reader.close(); proc.join()
    assert proc.exitcode == 0
    print(json.dumps({"spawned_preflight": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
