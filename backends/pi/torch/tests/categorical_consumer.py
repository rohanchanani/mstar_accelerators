"""Real optional categorical primitive gate, using the existing graph runner.

Run through tools/run_hardware.py. This supplies backend conformance evidence;
it is not an alternative registered-acceleration qualification consumer.
"""
from __future__ import annotations

import math
import torch
import mstar_pi
from mstar_pi.ggml import Graph, prepare_graph
from categorical_reference import POLICY, categorical_row, known_answers, scores, uniform
from graph_consumer import main


def run_cases() -> list[dict]:
    if not __debug__:
        raise RuntimeError("categorical gate needs assertion oracles")
    known_answers()
    work, producer = mstar_pi.Stream(), mstar_pi.Stream()
    records = []

    def prepare(rows, vocabulary):
        def factory():
            graph = Graph(max_nodes=16)
            logits = graph.tensor("logits", (vocabulary, rows))
            seeds = graph.tensor("seeds", (rows,), "i64")
            offsets = graph.tensor("offsets", (rows,), "i64")
            temperatures = graph.tensor("temperatures", (rows,))
            graph.output("tokens", graph.categorical(logits, seeds, offsets, temperatures))
            return graph
        return prepare_graph(factory, {}, slots=1)

    def invoke(program, logits, seeds, offsets, temperatures):
        cpu = {"logits": logits, "seeds": seeds, "offsets": offsets, "temperatures": temperatures}
        with producer:
            bound = {name: value.to("mstar_pi", non_blocking=True) for name, value in cpu.items()}
        with work:
            # Guarded destination also exercises a nonzero aligned byte offset.
            # Arbitrary I64 fixture words use the retained H2D copy path;
            # device fill32 only admits repeated 32-bit patterns.
            storage = torch.full((len(seeds) + 64,), -71, dtype=torch.int64).to("mstar_pi")
            output = storage[32:-32]
            lease = program.try_acquire()
            assert lease is not None
            lease.submit(bound, {"tokens": output})
            lease.close()
            retained = output.clone()
            host = torch.empty(retained.shape, dtype=torch.int64, pin_memory=True)
            host.copy_(retained, non_blocking=True)
            done = mstar_pi.Event(); done.record(work)
        # Observation waits here; no host logit or RNG continuation exists
        # between graph submission and the real retained Torch consumer.
        done.synchronize(); mstar_pi.collect()
        for name in ("seeds", "offsets", "temperatures"):
            assert torch.equal(bound[name].cpu(), cpu[name])
        guards = storage.cpu()
        assert torch.all(guards[:32] == -71) and torch.all(guards[-32:] == -71)
        assert program.info()["active_slots"] == 0
        return host.clone(), (output, retained), done

    def expected(logits, seeds, offsets, temperatures):
        return torch.tensor([categorical_row(row, seed, offset, temperature)
                             for row, seed, offset, temperature in zip(
                                 logits.tolist(), seeds.tolist(), offsets.tolist(), temperatures.tolist())],
                            dtype=torch.int64)

    with torch.inference_mode():
        # Last partial tile, masking, all seed/offset words, and row reordering.
        logits = (((torch.arange(1025) * 17) % 41).float() / 7).repeat(32, 1)
        logits[:, ::11] = -math.inf
        seeds = torch.tensor([0, 1, -1, -(1 << 63), (1 << 63) - 1, 1 << 32, 17 << 32, 17] * 4, dtype=torch.int64)
        offsets = torch.tensor([i + ((i % 4) << 32) for i in range(32)], dtype=torch.int64)
        temperatures = torch.tensor([.5, 1., 1.5, 2.] * 8)
        program = prepare(32, 1025)
        target = expected(logits, seeds, offsets, temperatures)
        margins = []
        for delta in (0, 1):
            for row, seed, offset, temperature in zip(logits.tolist(), seeds.tolist(), offsets.tolist(), temperatures.tolist()):
                ranked = sorted(scores(row, seed, offset + delta, temperature), reverse=True)
                margins.append(ranked[0] - ranked[1])
        assert min(margins) > 1e-4, margins  # fixed inputs, never selected from GPU results
        first, retained, done = invoke(program, logits, seeds, offsets, temperatures)
        assert torch.equal(first, target)
        repeat, _, _ = invoke(program, logits, seeds, offsets, temperatures)
        assert torch.equal(first, repeat)
        order = torch.arange(31, -1, -1)
        permuted, _, _ = invoke(program, logits[order], seeds[order], offsets[order], temperatures[order])
        assert torch.equal(permuted, first[order])
        advanced, _, _ = invoke(program, logits, seeds, offsets + 1, temperatures)
        assert torch.equal(advanced, expected(logits, seeds, offsets + 1, temperatures))
        assert not torch.equal(advanced, first)
        assert done.query() and all(torch.equal(value.cpu(), first) for value in retained)
        assert program.info()["submissions"] == program.info()["retirements"] == 4
        records.append({"case": "categorical_exact_mapping_repeat_offset_row_order_retained_results", "policy": POLICY,
                        "tokens": first.tolist(), "minimum_score_margin": min(margins),
                        "seed_offset_immutable": True, "output_guards": True, "graph": program.info()})
        program.close()

        # Two temperatures give different categorical distributions. The
        # independent softmax/binomial check does not demand a particular RNG.
        n = 2048
        weights = torch.tensor([.05, .15, .25, .55], dtype=torch.float32)
        logits = torch.cat((weights.log(), torch.full((2,), -math.inf))).repeat(2 * n, 1)
        seeds = torch.full((2 * n,), 0x1234567887654321, dtype=torch.int64)
        offsets = torch.arange(2 * n, dtype=torch.int64)
        temperatures = torch.cat((torch.full((n,), .5), torch.full((n,), 1.5)))
        program = prepare(2 * n, 6)
        sampled, _, _ = invoke(program, logits, seeds, offsets, temperatures)
        distributions = []
        assert torch.all((sampled >= 0) & (sampled < 4))
        for start, temperature in ((0, .5), (n, 1.5)):
            probabilities = torch.softmax(weights.double().log() / temperature, dim=0)
            counts = torch.bincount(sampled[start:start+n], minlength=4)
            bounds = 6 * torch.sqrt(n * probabilities * (1 - probabilities)) + 2
            assert torch.all((counts - n * probabilities).abs() <= bounds), (counts, probabilities, bounds)
            distributions.append({"temperature": temperature, "samples": n, "counts": counts.tolist(),
                                  "probabilities": probabilities.tolist(), "count_error_bounds": bounds.tolist()})
        records.append({"case": "categorical_distribution_temperature_and_masks", "distributions": distributions})
        program.close()

        # Exceptional rows are explicit substrate behavior outside the ordinary
        # model support domain; they never silently turn into token zero.
        logits = torch.full((9, 1025), -math.inf)
        logits[0, 1024] = -17.
        logits[2, 2] = logits[2, 900] = math.inf
        logits[3, 20] = math.nan
        logits[4:7, 31] = 1.
        logits[7, 605] = logits[7, 679] = 0.
        logits[8, 8] = 0.; logits[8, 900] = -0.
        seeds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 4, 8], dtype=torch.int64)
        offsets = torch.zeros(9, dtype=torch.int64)
        temperatures = torch.tensor([1., 1., 1., 1., 0., -1., math.inf, 1., 1.])
        # This fixed original-index tie crosses reduction lanes.
        assert uniform(4, 0, 605) == uniform(4, 0, 679)
        program = prepare(9, 1025)
        sampled, _, _ = invoke(program, logits, seeds, offsets, temperatures)
        assert torch.equal(sampled, expected(logits, seeds, offsets, temperatures))
        assert sampled[0] == 1024 and sampled[7] == 605
        assert sampled[[1, 3, 4, 5, 6]].tolist() == [-1] * 5
        records.append({"case": "categorical_inf_nan_invalid_temperature_all_masked_ties", "tokens": sampled.tolist(),
                        "tie_tokens": [605, 679], "tie_seed": 4, "invalid_token": -1})
        program.close()
        mstar_pi.synchronize(); mstar_pi.collect()
    return records


if __name__ == "__main__":
    main(cases=run_cases)
