"""Filesystem copies become independent execution choices in one logical node."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest
import torch

from mstar.engine.execution import AmbiguousExecution, InvocationFacts
from mstar.engine.execution import Support
from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE, POS, SAMPLER
from mstar.model.whisper.whisper_model import WhisperModel
from mstar.model.whisper.submodules import WhisperDecoderSubmodule, WhisperEncoderSubmodule
from mstar_accelerators.sources import load_accelerations
from mstar_accelerators.testing.whisper import audio_inputs

from .cpu_runtime import DEFAULT_SOURCES
from .engine import EngineSession


def copied_files(root, *, specialized):
    root.mkdir()
    declarations = load_accelerations(DEFAULT_SOURCES)
    encoder = next(name for name, value in declarations.items()
                   if value.reference is WhisperEncoderSubmodule)
    shutil.copyfile(DEFAULT_SOURCES / (encoder + ".py"), root / "audio_encoder.py")
    sources = {walk: next(name for name, value in declarations.items()
                         if value.reference is WhisperDecoderSubmodule and
                         value.when.specialize({"graph_walk": walk}).evaluate({}) is not Support.NO)
               for walk in ("prefill", "decode")}
    copies = [("decoder_prefill", "prefill"), ("decoder_decode", "decode")]
    if not specialized:
        copies.append(("decoder_prefill_copy", "prefill"))
    for name, walk in copies:
        destination = root / (name + ".py")
        shutil.copyfile(DEFAULT_SOURCES / (sources[walk] + ".py"), destination)
        with destination.open("a") as output:
            output.write("\nfrom dataclasses import replace\nfrom mstar_accelerators.conditions import When, Eq\n")
            output.write(f"acceleration = replace(acceleration, when=When(acceleration.when, Eq('graph_walk', {walk!r})))\n")
    return root


@pytest.fixture
def scenario(tmp_path):
    checkpoint = os.environ.get("MSTAR_WHISPER_CHECKPOINT")
    if not checkpoint:
        pytest.skip("full checkpoint is an explicit asset")
    torch.set_num_threads(4)
    model = WhisperModel(checkpoint)
    fixtures = Path(__file__).parents[1] / "fixtures/whisper"
    inputs, _ = audio_inputs(model, fixtures / "jfk16k.wav")
    oracle = json.loads((fixtures / "reference.json").read_text())["cases"]["jfk16k.wav"]
    persistent = os.environ.get("MSTAR_WHISPER_VARIANTS_ROOT")
    root = Path(persistent) if persistent else tmp_path
    root.mkdir(exist_ok=True)
    return model, inputs, oracle, root


@torch.no_grad()
def test_copied_files_specialize_one_decoder_with_shared_weights_and_state(scenario):
    model, inputs, oracle, root = scenario
    files = copied_files(root / "specialized", specialized=True)
    engine = EngineSession(model, source=files)
    try:
        modules = engine.manager.engine._submodules
        assert set(modules) == {"audio_encoder", "decoder"}
        shell = modules["decoder"].submodule
        capabilities = shell.execution_capabilities
        assert {option.name for option in capabilities.options} == {"decoder_prefill", "decoder_decode"}
        choices = {
            walk: [option.name for option in capabilities.possibilities(InvocationFacts(walk, 1, lengths))]
            for walk, lengths in (("prefill", (4,)), ("decode", (1,)))
        }
        assert choices == {"prefill": ["decoder_prefill"], "decode": ["decoder_decode"]}
        assert all(modules["decoder"].resources[key] is engine.resources[key]
                   for key in (KV_CACHE, CROSS_KV_CACHE, POS, SAMPLER))
        prepared_weights = [weights for key, weights in shell.binding.prepared.items()
                            if getattr(key, "prefix", None) == "decoder."]
        assert len(prepared_weights) == 1
        weight = prepared_weights[0]["decoder.embed_tokens.weight"]
        decoder_programs = [program for program in engine.provider.prepared
                            if "decoder.embed_tokens.weight" in program.constants]
        # Candidate-local preparation omits unreachable programs. Both mode
        # programs share the one bound immutable weight representation.
        assert len(decoder_programs) == 2
        assert all(program.constants["decoder.embed_tokens.weight"] is weight
                   for program in decoder_programs)
        encoded = engine.run(0, "audio_encoder", "prefill", {
            "audio_features": inputs["audio_features"]}).output["encoder_states"]
        ids = inputs["text_inputs"][0]
        tokens, retained = [], []
        for iteration in range(model.get_max_output_tokens()):
            walk = "prefill" if iteration == 0 else "decode"
            staged = {"text_inputs": [ids]}
            if iteration == 0:
                staged["encoder_states"] = encoded
            result = engine.run(0, "decoder", walk, staged, iteration=iteration)
            ids = result.output["new_token"][0]
            tokens.append(int(ids))
            retained.append(ids)
            if int(ids) == model.config.eos_token_id:
                break
        assert tokens == oracle["tokens"]
        assert [int(token) for token in retained] == tokens
        assert engine.resources[POS].position(0, "main") == oracle["position_after_commit"]
        assert engine.resources[SAMPLER]._sampler._step_offset[0] == oracle["sampling_offset_after_commit"]
        report = {"passed": True, "scope": "explicit CPU interpretation, same copied file callbacks",
                  "source_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in files.glob("*.py")},
                  "logical_nodes": sorted(modules), "decoder_implementation_choices": choices,
                  "decoder_weight_representations": len(prepared_weights),
                  "decoder_programs_share_weight_storage": True,
                  "decoder_programs_prepared": len(decoder_programs),
                  "tokens": tokens, "resource_set_shared": True}
    finally:
        engine.close()
    (root / "specialized.result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def test_unchanged_copies_report_overlap_before_resource_effects(scenario):
    model, inputs, _, root = scenario
    files = copied_files(root / "overlap", specialized=False)
    engine = EngineSession(model, source=files)
    try:
        encoded = engine.run(0, "audio_encoder", "prefill", {
            "audio_features": inputs["audio_features"]}).output["encoder_states"]
        before = {key: engine.resources[key].kv_cache.tensor.clone() for key in (KV_CACHE, CROSS_KV_CACHE)}
        free_pages = {key: engine.resources[key]._arena.num_free for key in before}
        submitted = engine.provider.submissions
        with pytest.raises(AmbiguousExecution, match="overlapping executions") as error:
            engine.run(0, "decoder", "prefill", {"text_inputs": inputs["text_inputs"], "encoder_states": encoded})
        assert engine.provider.submissions == submitted
        assert engine.resources[POS].position(0, "main") == 0
        assert engine.resources[SAMPLER]._sampler._step_offset[0] == 0
        for key in before:
            assert torch.equal(engine.resources[key].kv_cache.tensor, before[key])
            assert engine.resources[key]._arena.num_free == free_pages[key]
        assert all(program.active == 0 for program in engine.provider.prepared)
        report = {"passed": True, "scope": "explicit CPU interpretation",
                  "diagnostic": str(error.value), "decoder_submissions": 0,
                  "position": 0, "sampling_steps": 0, "cache_storage_unchanged": True,
                  "page_reservations_unchanged": True, "active_workspaces": 0}
    finally:
        engine.close()
    (root / "overlap.result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
