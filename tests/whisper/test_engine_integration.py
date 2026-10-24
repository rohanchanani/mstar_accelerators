"""Interleaved requests through the same filled files and actual M* engine."""
import os
import json
from pathlib import Path

import pytest
import torch

from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE, POS, SAMPLER
from mstar.model.whisper.whisper_model import WhisperModel
from mstar_accelerators.testing.whisper import audio_inputs

from .engine import EngineSession


@pytest.mark.skipif(not os.environ.get("MSTAR_WHISPER_CHECKPOINT"), reason="full checkpoint is an explicit asset")
@pytest.mark.parametrize("request_ids", [(7,), (7, 11)])
@torch.no_grad()
def test_whisper_graph_runs_through_actual_engine_and_registry(request_ids):
    torch.set_num_threads(4)
    model = WhisperModel(os.environ["MSTAR_WHISPER_CHECKPOINT"])
    fixture_root = Path(__file__).parents[1] / "fixtures/whisper"
    fixture_names = ("jfk16k.wav", "jfk_tail16k.wav")
    oracle = json.loads((fixture_root / "reference.json").read_text())["cases"]
    inputs = {rid: audio_inputs(model, fixture_root / fixture_names[index])[0]
              for index, rid in enumerate(request_ids)}
    engine = EngineSession(model, requests=request_ids)
    firsts, tokens, retained, trajectories = {}, {}, {}, {rid: [] for rid in request_ids}
    try:
        for rid in request_ids:
            states = engine.run(rid, "audio_encoder", "prefill", {
                "audio_features": inputs[rid]["audio_features"]}).output["encoder_states"]
            first = engine.run(rid, "decoder", "prefill", {
                "encoder_states": states, "text_inputs": inputs[rid]["text_inputs"]}).output
            assert first["text_inputs"] is first["new_token"]
            firsts[rid] = first
            tokens[rid] = first["new_token"][0]
            retained[rid] = tokens[rid].clone()
            trajectories[rid].append(int(tokens[rid]))
        cross_before = engine.resources[CROSS_KV_CACHE].kv_cache.tensor.clone()
        active = set(request_ids)
        for iteration in range(1, model.get_max_output_tokens()):
            if not active:
                break
            for rid in request_ids:
                if rid not in active:
                    continue
                tokens[rid] = engine.run(rid, "decoder", "decode", {"text_inputs": [tokens[rid]]},
                                          iteration=iteration).output["new_token"][0]
                trajectories[rid].append(int(tokens[rid]))
                if int(tokens[rid]) == model.config.eos_token_id:
                    active.remove(rid)
        assert not active
        assert torch.equal(cross_before, engine.resources[CROSS_KV_CACHE].kv_cache.tensor)
        for index, rid in enumerate(request_ids):
            assert torch.equal(firsts[rid]["new_token"][0], retained[rid])
            assert trajectories[rid] == oracle[fixture_names[index]]["tokens"]
            assert engine.resources[POS].position(rid, "main") == len(trajectories[rid]) + 3
            assert engine.resources[SAMPLER]._sampler._step_offset[rid] == len(trajectories[rid])
        for key in (KV_CACHE, CROSS_KV_CACHE):
            streams = engine.resources[key]._streams
            pages = [set(streams[rid]["main"].page_indices) for rid in request_ids]
            assert len(set.union(*pages)) == sum(map(len, pages))
        assert engine.provider.prepared
        assert {(node, walk) for _, node, walk in engine.executions} == {
            ("audio_encoder", "prefill"), ("decoder", "prefill"), ("decoder", "decode")}
    finally:
        engine.close()
