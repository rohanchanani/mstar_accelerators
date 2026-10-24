"""Host checks of explicit FP64 Torch math at the original FP32 state boundary."""
from types import SimpleNamespace

import torch

from mstar.engine.execution import Execution, InvocationFacts, LogicalInvocation
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.sampler.plan import SamplingResult
from mstar.engine.resources.sequence import DenseSequenceView
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import StepContext
from mstar.model.binding import NodeBinding
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine
from mstar.model.weights import LogicalWeightSource, WeightSpec
from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE, POS, SAMPLER
from mstar.model.whisper.reference import build_decoder, build_encoder
from mstar.model.whisper.submodules import WhisperDecoderSubmodule
from mstar.model.whisper.whisper_model import WhisperModel
from mstar_accelerators.backends.torch import TorchProgram, prepare_torch_programs
from mstar_accelerators.backends.weights import prepare_weights
from mstar_accelerators.implementations.whisper.torch import WhisperTorch
from mstar_accelerators.testing.numerics import assert_close
from mstar_accelerators.testing.reference_resources.snapshot import ReferencePlanConsumer
from mstar_accelerators.testing.whisper import _runtime_resources

from .test_complete_programs import config, random_weights


def test_fp64_encoder_preserves_fp32_output_boundary(config):
    weights = random_weights(config)
    precise = {name: value.double() for name, value in weights.items()}
    features = torch.randn(config.num_mel_bins, config.max_source_positions * 2)
    program = WhisperTorch(config, precise, {})
    call = SimpleNamespace(inputs=SimpleNamespace(audio_features=(features,)))
    actual = program.encoder(call)["encoder_states"][0]
    expected = build_encoder(config, weights, dtype=torch.float64).run(features)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


def test_torch_decoder_uses_owned_plans_and_fp32_cache(config):
    weights = random_weights(config)
    model = WhisperModel.__new__(WhisperModel)
    model.config = config
    source = LogicalWeightSource([WeightSpec(name, tuple(value.shape)) for name, value in weights.items()],
                                lambda: weights.items())
    model.logical_weights = lambda: source
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, {"resources": {
        key: {"max_bytes": 3 * 2 * config.decoder_layers * capacity * config.d_model * 4,
              "prefix_cache": False}
        for key, capacity in ((KV_CACHE, config.max_target_positions),
                              (CROSS_KV_CACHE, config.max_source_positions))
    }})
    requests = {key: (DenseSequenceView(),) for key in (KV_CACHE, CROSS_KV_CACHE)}
    specs, resources = _runtime_resources(model, torch.device("cpu"), requests, specs=specs)
    shell = WhisperDecoderSubmodule(config)
    group = SimpleNamespace(rank=0, world_size=1)
    binding = NodeBinding(model, shell, "decoder", torch.device("cpu"), torch.float32,
                          group, group, {}, specs)
    precise = prepare_weights(binding, "decoder.", dtype=torch.float64)
    program = WhisperTorch(config, precise, resources)
    prepared = prepare_torch_programs(binding, resources,
        {walk: TorchProgram(program.decoder) for walk in ("prefill", "decode")})
    runner = StepRunner(resources)
    runner.ingest_request(0, {SAMPLER: SamplingReqConfig(temperature=0)})
    snapshot = ReferencePlanConsumer(resources)
    reference = build_decoder(config, weights)
    reference_shell = WhisperDecoderSubmodule(config, reference_program=reference)
    reference_shell.bind_node_resources(snapshot.resources)
    encoder = torch.randn(config.max_source_positions, config.d_model)
    ids = torch.tensor([1, 3, 4, 5])
    held = []
    with torch.no_grad():
        for index in range(5):
            walk = "prefill" if index == 0 else "decode"
            logical = ARNodeInputs(input_ids=ids, input_seq_len=ids.numel(),
                tensor_inputs={"encoder_states": encoder} if index == 0 else {})
            step = shell.declare_step(walk, [0], [logical])
            step.set_ctx(StepContext(request_ids=[0], graph_walk=walk, slot=0, capture=False))
            assert runner.admit(step).ready
            plans = runner.plan(step)
            snapshot.use(plans)
            lease = prepared.acquire(InvocationFacts(walk, 1, (ids.numel(),)))
            execution = Execution(None, lease)
            call = prepared.materialize(LogicalInvocation(walk, [0], [logical], step, plans), execution)
            actual = prepared.primitive(call, execution)
            expected = reference.run(walk, ModelInputsFromEngine([0], {}, snapshot.resources),
                                     ids, encoder if index == 0 else None, return_logits=True)
            assert torch.equal(actual.tokens, expected.argmax(-1))
            held.append((actual.tokens, actual.tokens.clone()))
            assert resources[POS].position(0, "main") == (0 if index == 0 else index + 3)
            assert resources[SAMPLER]._sampler._step_offset[0] == index
            for key in (KV_CACHE, CROSS_KV_CACHE):
                assert resources[key].kv_cache.tensor.dtype == torch.float32
                assert_close(key, resources[key].kv_cache.tensor, snapshot.resources[key].kv_cache.tensor)
            resources[SAMPLER].accept_result(SamplingResult(actual.tokens), step.ctx)
            runner.commit(step)
            lease.close()
            ids = actual.tokens
    assert all(torch.equal(value, expected) for value, expected in held)
    runner.remove_request(0)
    prepared.close()
