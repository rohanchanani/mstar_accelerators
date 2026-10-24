import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.binding import resolve_bound_specs
from mstar.engine.resources.sequence import PagedSequenceView
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.sampler.plan import SamplingResult
from mstar.engine.resources.step import StepContext
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine
from mstar.model.whisper.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, POS, SAMPLER, WhisperModelConfig
from mstar.model.whisper.reference import build_decoder, build_encoder
from mstar.model.whisper.submodules import WhisperDecoderSubmodule, WhisperEncoderSubmodule
from mstar.model.whisper.weights import weight_specs
from mstar.model.whisper.whisper_model import WhisperModel
from mstar_accelerators.implementations.whisper.pi_v3d.graph import decoder_graph, encoder_graph
from mstar_accelerators.backends.graph import PagedLayout
from .binding import materialized_decoder_inputs
from mstar_accelerators.testing import reference_resources
from mstar_accelerators.testing.reference_resources.snapshot import ReferencePlanConsumer
from mstar_accelerators.testing.numerics import assert_close

from .evaluator import EagerGraph


@pytest.fixture
def config():
    return WhisperModelConfig(d_model=12, encoder_layers=2, decoder_layers=2,
                              encoder_attention_heads=3, decoder_attention_heads=2,
                              encoder_ffn_dim=17, decoder_ffn_dim=19, num_mel_bins=4,
                              max_source_positions=8, max_target_positions=12, vocab_size=23,
                              decoder_start_token_id=1, eos_token_id=2, no_timestamps_token_id=3,
                              suppress_tokens=[0, 22], begin_suppress_tokens=[7], scale_embedding=True)


def random_weights(config):
    torch.manual_seed(713)
    weights = {}
    for spec in weight_specs(config):
        if spec.alias_of:
            weights[spec.name] = weights[spec.alias_of]
        elif spec.default:
            weights[spec.name] = spec.default()
        else:
            weights[spec.name] = torch.randn(spec.shape) * 0.08
            if "layer_norm.weight" in spec.name:
                weights[spec.name] += 1
    return weights


def resource_set(config):
    model = WhisperModel.__new__(WhisperModel)
    model.config = config
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, {"resources": {
        key: {"max_bytes": 2 * 4 * 5 * config.decoder_layers * config.d_model * 4,
              "prefix_cache": False}
        for key in (KV_CACHE, CROSS_KV_CACHE)
    }})
    specs = resolve_bound_specs(specs, {key: (PagedSequenceView(4),)
                                      for key in (KV_CACHE, CROSS_KV_CACHE)}, dtype=torch.float32)
    by_key = {s.resource_key: s for s in specs}
    providers = {
        KV_CACHE: reference_resources.paged_kv, CROSS_KV_CACHE: reference_resources.paged_kv,
        ATTN: reference_resources.attention, CROSS_ATTN: reference_resources.cross_attention,
        POS: reference_resources.positions, SAMPLER: reference_resources.greedy_sampler,
    }
    resources = {spec.resource_key: providers[spec.resource_key](
        spec, EngineResourceInfo(torch.device("cpu"), kv_dtype=torch.float32, needs_remote_transfer=False,
                                 dependencies={key: by_key[key] for key in spec.depends_on()}))
        for spec in specs}
    return resources, StepRunner(resources), PagedLayout({key: resources[key].config for key in (KV_CACHE, CROSS_KV_CACHE)})


def plan(shell, runner, walk, ids, encoder=None):
    logical = ARNodeInputs(input_ids=ids, input_seq_len=ids.numel(),
                           tensor_inputs={} if encoder is None else {"encoder_states": encoder})
    step = shell.declare_step(walk, [0], [logical])
    step.set_ctx(StepContext(request_ids=[0], graph_walk=walk, slot=0, capture=False))
    assert runner.admit(step).ready
    return step, runner.plan(step)


def mask(config, first):
    result = torch.zeros(1, config.vocab_size)
    result[:, config.suppress_tokens + (config.begin_suppress_tokens if first else [])] = float("-inf")
    return result


def test_encoder_graph_against_original_hf(config):
    weights = random_weights(config)
    features = torch.randn(config.num_mel_bins, config.max_source_positions * 2)
    graph = EagerGraph({**weights, "audio_features": features})
    with torch.no_grad():
        encoder_graph(config, graph=graph)
        reference = build_encoder(config, weights, dtype=torch.float64).run(features)
    assert_close("encoder", graph.outputs["encoder_states"].squeeze(0).squeeze(0), reference,
                 dtype=torch.float32)


def test_prefill_decode_state_and_outputs_against_original_mstar(config):
    weights = random_weights(config)
    native_resources, native_runner, layout = resource_set(config)
    reference_resources_, reference_runner, _ = resource_set(config)
    reference = build_decoder(config, weights)
    reference_shell = WhisperDecoderSubmodule(config, reference_program=reference)
    reference_shell.bind_node_resources(reference_resources_)
    shell = WhisperDecoderSubmodule(config)
    for runner in (native_runner, reference_runner):
        runner.ingest_request(0, {SAMPLER: SamplingReqConfig(temperature=0)})
    for resources in (native_resources, reference_resources_):
        for key in (KV_CACHE, CROSS_KV_CACHE):
            resources[key].kv_cache.tensor.fill_(987.0)
            resources[key].kv_cache.tensor[:, 0].zero_()
    encoder = torch.randn(config.max_source_positions, config.d_model)
    ids = torch.tensor([1, 3, 4, 5])
    held_outputs = []
    cross_after_prefill = None
    with torch.no_grad():
        for iteration in range(9):
            first = iteration == 0
            walk = "prefill" if first else "decode"
            step, plans = plan(shell, native_runner, walk, ids, encoder if first else None)
            ref_step, ref_plans = plan(reference_shell, reference_runner, walk, ids, encoder if first else None)
            inputs = materialized_decoder_inputs(config, native_resources, plans, input_ids=ids,
                                       encoder_states=encoder if first else None, prefill=first)
            graph = EagerGraph({**weights, **inputs, f"{SAMPLER}.logit_mask": mask(config, first)})
            decoder_graph(config, layout, prefill=first, graph=graph, diagnostics=True)
            engine = ModelInputsFromEngine([0], {}, reference_resources_, step=ref_step)
            logits = reference.run(walk, engine, ids, encoder if first else None, return_logits=True)
            actual_logits = graph.outputs["logits"].reshape(1, -1)
            assert_close("logits", actual_logits, logits)
            tokens = graph.outputs["tokens"].reshape(1)
            assert torch.equal(tokens, logits.argmax(dim=-1))
            held_outputs.append((tokens, tokens.clone()))
            # Logical effects do not happen in either physical body.
            assert native_resources[POS].position(0, "main") == (0 if first else iteration + 3)
            assert native_resources[SAMPLER]._sampler._step_offset[0] == iteration
            native_resources[SAMPLER].accept_result(SamplingResult(tokens), step.ctx)
            reference_resources_[SAMPLER].sample([0], logits)
            native_runner.commit(step)
            reference_runner.commit(ref_step)
            assert native_resources[POS].position(0, "main") == iteration + 4
            assert native_resources[SAMPLER]._sampler._step_offset[0] == iteration + 1
            assert native_resources[CROSS_KV_CACHE].kv_cache.tensor[:, 0].count_nonzero() == 0
            for key in (KV_CACHE, CROSS_KV_CACHE):
                assert_close(key, native_resources[key].kv_cache.tensor,
                             reference_resources_[key].kv_cache.tensor)
            if first:
                cross_after_prefill = native_resources[CROSS_KV_CACHE].kv_cache.tensor.clone()
            else:
                assert torch.equal(cross_after_prefill, native_resources[CROSS_KV_CACHE].kv_cache.tensor)
                assert plans[CROSS_KV_CACHE]["main"].write_locations.total_tokens == 0
            ids = tokens
    for held, value in held_outputs:
        assert torch.equal(held, value)
    for runner in (native_runner, reference_runner):
        runner.remove_request(0)
    for resources in (native_resources, reference_resources_):
        assert resources[KV_CACHE]._arena.num_free == resources[KV_CACHE].config.max_num_pages - 1
        assert resources[CROSS_KV_CACHE]._arena.num_free == resources[CROSS_KV_CACHE].config.max_num_pages - 1


def test_fresh_shells_are_weightless_and_preserve_output_limit(config):
    model = WhisperModel.__new__(WhisperModel)
    model.config = config
    first, second = model.create_submodule("decoder"), model.create_submodule("decoder")
    assert first is not second
    assert list(first.parameters()) == []
    assert list(model.create_submodule("audio_encoder").parameters()) == []
    assert model.get_max_output_tokens() == config.max_target_positions - 4


def test_reference_consumes_the_exact_native_plan_objects(config):
    weights = random_weights(config)
    resources, runner, layout = resource_set(config)
    shell = WhisperDecoderSubmodule(config)
    snapshot = ReferencePlanConsumer(resources)
    reference = build_decoder(config, weights)
    reference_shell = WhisperDecoderSubmodule(config, reference_program=reference)
    reference_shell.bind_node_resources(snapshot.resources)
    runner.ingest_request(0, {SAMPLER: SamplingReqConfig(temperature=0)})
    encoder = torch.randn(config.max_source_positions, config.d_model)
    ids = torch.tensor([1, 3, 4, 5])
    with torch.no_grad():
        for first in (True, False):
            walk = "prefill" if first else "decode"
            step, plans = plan(shell, runner, walk, ids, encoder if first else None)
            snapshot.use(plans)
            assert snapshot.resources[KV_CACHE].plans is plans[KV_CACHE]
            assert snapshot.resources[ATTN].plans is plans[ATTN]
            bindings = materialized_decoder_inputs(config, resources, plans, input_ids=ids,
                                         encoder_states=encoder if first else None, prefill=first)
            graph = EagerGraph({**weights, **bindings, f"{SAMPLER}.logit_mask": mask(config, first)})
            decoder_graph(config, layout, prefill=first, graph=graph, diagnostics=True)
            expected = reference.run(walk, ModelInputsFromEngine([0], {}, snapshot.resources), ids,
                                     encoder if first else None, return_logits=True)
            assert_close("snapshot.logits", graph.outputs["logits"].reshape(1, -1), expected)
            tokens = graph.outputs["tokens"].reshape(1)
            resources[SAMPLER].accept_result(SamplingResult(tokens), step.ctx)
            runner.commit(step)
            ids = tokens
