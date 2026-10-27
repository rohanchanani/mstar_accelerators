"""Same filled filesystem declarations through the actual M* engine on CPU.

The provider injection is explicit test machinery. The original FP32 graph
supplies encoder/decoder values; FP64 algebra and CPU FP32 encoder diagnostics
retain their distinct claims. No native hardware behavior is simulated.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE, POS, SAMPLER
from mstar.model.whisper.reference import build_decoder, build_encoder
from mstar.model.whisper.whisper_model import WhisperModel
from mstar_accelerators.implementations.whisper.pi_v3d.graph import encoder_graph
from mstar_accelerators.testing.reference_resources.snapshot import ReferencePlanConsumer
from mstar_accelerators.testing.whisper import _compare, audio_inputs
from mstar_accelerators.testing.numerics import compare

from .engine import EngineSession
from .evaluator import ReadOnlyAlgebraGraph


@pytest.mark.skipif(not os.environ.get("MSTAR_WHISPER_CHECKPOINT"), reason="full checkpoint is an explicit asset")
@torch.no_grad()
def test_full_checkpoint_builder_against_original():
    torch.set_num_threads(4)
    model = WhisperModel(os.environ["MSTAR_WHISPER_CHECKPOINT"])
    audio = os.environ.get("MSTAR_WHISPER_AUDIO", str(Path(__file__).parents[1] / "fixtures/whisper/jfk16k.wav"))
    inputs, identity = audio_inputs(model, audio)
    features, ids = inputs["audio_features"][0], inputs["text_inputs"][0]
    weights = model.logical_weights().load()

    precise_audio = build_encoder(model.config, weights, dtype=torch.float64).run(features)
    algebra = ReadOnlyAlgebraGraph({**weights, "audio_features": features})
    encoder_graph(model.config, graph=algebra)
    algebra_encoded = algebra.outputs["encoder_states"].reshape_as(precise_audio)
    checks = [_compare("encoder.algebra_fp64", algebra_encoded, precise_audio)]
    del algebra, algebra_encoded
    reference_audio = build_encoder(model.config, weights).run(features)

    engine = EngineSession(model)
    resources = engine.resources
    try:
        encoder_run = engine.run(0, "audio_encoder", "prefill", {"audio_features": [features]})
        encoded = encoder_run.output["encoder_states"][0]
        cpu_fp32_encoder = compare("encoder.cpu_fp32", encoded, reference_audio, dtype=torch.float32, promoted=precise_audio)
        del precise_audio

        snapshot = ReferencePlanConsumer(resources)
        whole_snapshot = ReferencePlanConsumer(resources)
        reference = build_decoder(model.config, weights)
        whole_reference = build_decoder(model.config, weights)
        ref_shell = model.create_submodule("decoder")
        ref_shell.reference_program = reference
        ref_shell.bind_node_resources(snapshot.resources)
        whole_shell = model.create_submodule("decoder")
        whole_shell.reference_program = whole_reference
        whole_shell.bind_node_resources(whole_snapshot.resources)
        tokens, retained = [], []
        composed_cache_diagnostic = None

        for iteration in range(model.get_max_output_tokens()):
            first = iteration == 0
            walk = "prefill" if first else "decode"
            before_position = resources[POS].position(0, "main")
            before_sampling = resources[SAMPLER]._sampler._step_offset[0]
            cross_before = resources[CROSS_KV_CACHE].kv_cache.tensor.clone()

            def before_commit(_):
                assert resources[POS].position(0, "main") == before_position
                assert resources[SAMPLER]._sampler._step_offset[0] == before_sampling

            engine.provider.observe = before_commit
            staged = {"text_inputs": [ids]}
            if first:
                staged["encoder_states"] = [encoded]
            execution = engine.run(0, "decoder", walk, staged, iteration=iteration)
            engine.provider.observe = None
            plans = execution.batch.step.ctx.plan_results
            snapshot.use(plans)
            whole_snapshot.use(plans)
            expected = reference.run(walk, ModelInputsFromEngine([0], {}, snapshot.resources), ids,
                                     encoded if first else None, return_logits=True)
            whole_expected = whole_reference.run(walk, ModelInputsFromEngine([0], {}, whole_snapshot.resources),
                                                  ids, reference_audio if first else None, return_logits=True)
            actual_logits = execution.observation.argmax_inputs[-1].reshape_as(expected)
            checks.append(_compare(f"{walk}.{iteration}.logits", actual_logits, expected))
            actual = execution.output["new_token"][0]
            assert execution.output["new_token"] is execution.output["text_inputs"]
            assert torch.equal(actual, expected.argmax(dim=-1))
            assert torch.equal(actual, whole_expected.argmax(dim=-1))
            for key in (KV_CACHE, CROSS_KV_CACHE):
                checks.append(_compare(f"{walk}.{iteration}.{key}", resources[key].kv_cache.tensor,
                                       snapshot.resources[key].kv_cache.tensor))
                assert resources[key].kv_cache.tensor[:, 0].count_nonzero() == 0
            if first:
                composed_cache_diagnostic = compare("different_boundary.cross_cache",
                                                     resources[CROSS_KV_CACHE].kv_cache.tensor,
                                                     whole_snapshot.resources[CROSS_KV_CACHE].kv_cache.tensor)
                composed_cache_diagnostic["comparison"] = "different independently encoded inputs; diagnostic only"
            else:
                assert torch.equal(cross_before, resources[CROSS_KV_CACHE].kv_cache.tensor)
                assert plans[CROSS_KV_CACHE]["main"].write_locations.total_tokens == 0
            assert resources[POS].position(0, "main") == iteration + 4
            assert resources[SAMPLER]._sampler._step_offset[0] == iteration + 1
            tokens.append(int(actual.item()))
            retained.append(actual)
            ids = actual
            if tokens[-1] == model.config.eos_token_id:
                break
        assert tokens[-1] == model.config.eos_token_id
        assert [int(value.item()) for value in retained] == tokens
        report = {"gate": "generated_files_engine_algebra_v1", "passed": True, "audio": identity,
                  "tokens": tokens, "generated_tokens": len(tokens), "observations": checks,
                  "text": model.postprocess(torch.tensor(tokens), "text").decode("utf-8"),
                  "declarations": list(engine.registry.accelerations),
                  "programs_prepared": len(engine.provider.prepared),
                  "node_comparison": "identical encoder_states boundary input for decoder numerical/state checks",
                  "composition_comparison": "independent original HF encoder + M* decoder complete tokens through EOS",
                  "composed_internal_cache_diagnostic": composed_cache_diagnostic,
                  "cpu_fp32_encoder_diagnostic": cpu_fp32_encoder,
                  "encoder_reference": "CPU HF FP64 for algebra; original FP32 composed tokens retained",
                  "encoder_trajectory": "unchanged CPU FP32 graph output; no FP64/reference substitution",
                  "claim": "same filled acceleration files, actual registry/engine/commit, FP64 encoder algebra "
                           "and CPU FP32 decoder/state trajectory; no CPU FP32 encoder accuracy or native qualification"}
    finally:
        engine.close()
    report["programs_closed"] = sum(program.closed for program in engine.provider.prepared)
    report["resource_pages_released"] = True
    report_path = os.environ.get("MSTAR_WHISPER_FULL_REPORT")
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
