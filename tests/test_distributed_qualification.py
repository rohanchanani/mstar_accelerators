"""Actual CPU role processes with small synthetic Whisper values.

This exercises the central consumer and Gloo, not full-checkpoint or Pi target
qualification. Test-only factories replace assets; production binding, reference
math, resource lifecycle, agreement and both collectives execute unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources.kv.config import KVSpec
from mstar.engine.resources.sequence import DenseSequenceView, SequenceBudget
from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE, SAMPLER, WhisperModelConfig
from mstar.model.whisper.submodules import WhisperDecoderSubmodule, WhisperEncoderSubmodule
from mstar.model.whisper.whisper_model import WhisperModel
from mstar.model.whisper.weights import weight_specs
from mstar.model.weights import LogicalWeightSource
from mstar_accelerators import Acceleration, Eq, Implementation, When
from mstar_accelerators.backends.torch import TorchProgram, prepare_torch_programs
from mstar_accelerators.backends.weights import TensorShard, prepare_weights
from mstar_accelerators.conditions import AllRows, Or
from mstar_accelerators.implementations.whisper.torch import WhisperTorch
from mstar_accelerators.testing import hosting as distributed
from mstar_accelerators.testing.planning import Job
from mstar_accelerators.backends.discovery import Environment
from mstar_accelerators.testing.environments import combine
from mstar_accelerators.testing.harness import Result, Status


_REAL_ROLE = distributed._managed
_FFN_STATS = {}


class _SyntheticModel(WhisperModel):
    def __init__(self, _checkpoint):
        self.config = WhisperModelConfig(
            d_model=12, encoder_layers=2, decoder_layers=2,
            encoder_attention_heads=3, decoder_attention_heads=2,
            encoder_ffn_dim=16, decoder_ffn_dim=20, num_mel_bins=4,
            max_source_positions=8, max_target_positions=12, vocab_size=23,
            decoder_start_token_id=1, eos_token_id=2, no_timestamps_token_id=3,
            lang_to_id={"<|en|>": 4}, task_to_id={"transcribe": 5},
            suppress_tokens=[0, 22], begin_suppress_tokens=[7], scale_embedding=True)
        generator = torch.Generator().manual_seed(713)
        self._weights = {}
        for spec in weight_specs(self.config):
            if spec.alias_of:
                continue
            value = (spec.default() if spec.default else
                     torch.randn(spec.shape, generator=generator) * 0.08)
            if "layer_norm.weight" in spec.name:
                value += 1
            self._weights[spec.source] = value

    def logical_weights(self):
        return LogicalWeightSource(weight_specs(self.config), lambda: self._weights.items())


def _small_specs(model):
    specs = model.get_node_resources()
    for spec in specs:
        if isinstance(spec, KVSpec):
            config = spec.config
            spec.budget = SequenceBudget(3 * 2 * config.num_layers * config.max_seq_len
                                         * config.num_kv_heads * config.head_dim * 4)
    return specs


def _synthetic_audio(model, path, **_kwargs):
    generator = torch.Generator().manual_seed(117)
    return {"audio_features": [torch.randn(model.config.num_mel_bins,
                model.config.max_source_positions * 2, generator=generator)]}, {
        "scope": "synthetic test features", "path": str(path)}


class _SyntheticCase:
    key = 'synthetic-whisper'

    def describe(self, checkpoint):
        return _SyntheticModel(checkpoint)


def _synthetic_role(*args):
    from mstar_accelerators.testing import whisper
    from mstar_accelerators.testing.models import whisper as cases

    whisper.audio_inputs = _synthetic_audio
    cases._resource_specs = _small_specs
    _REAL_ROLE(*args)
    path = Path(args[-1])
    report = json.loads(path.read_text())
    report['test_ffn'] = _FFN_STATS
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _declaration(node, protocol="all_reduce", device="cpu"):
    """Test callbacks only; not an optimizer deliverable or generated book."""
    encoder = node == "audio_encoder"

    def prepare(binding, resources):
        prefix = "encoder." if encoder else "decoder."
        shards = {}
        if not encoder:
            for layer in range(binding.submodule.config.decoder_layers):
                layer_prefix = f"decoder.layers.{layer}"
                shards.update({layer_prefix + ".fc1.weight": TensorShard(0),
                               layer_prefix + ".fc1.bias": TensorShard(0),
                               layer_prefix + ".fc2.weight": TensorShard(1)})
        weights = prepare_weights(binding, prefix, dtype=torch.float64, shards=shards)

        def ffn(prefix, x):
            hidden = F.gelu(F.linear(x, weights[prefix + ".fc1.weight"],
                                     weights[prefix + ".fc1.bias"]), approximate="none")
            local = F.linear(hidden, weights[prefix + ".fc2.weight"])
            stats = _FFN_STATS.setdefault(protocol, {"calls": 0, "nonzero_contributions": 0,
                "rank": binding.tp_group.rank, "local_ffn_width": hidden.shape[-1]})
            stats["calls"] += 1
            stats["nonzero_contributions"] += bool(local.count_nonzero())
            if protocol == "all_reduce":
                combined = binding.tp_group.all_reduce(local)
            else:
                combined = binding.tp_group.all_gather(local.unsqueeze(0), dim=0).sum(0)
            return combined + weights[prefix + ".fc2.bias"]

        math = WhisperTorch(binding.submodule.config, weights, resources,
                            ffn=None if encoder else ffn)
        programs = ({"prefill": TorchProgram(math.encoder)} if encoder else
                    {walk: TorchProgram(math.decoder) for walk in ("prefill", "decode")})
        return prepare_torch_programs(binding, resources, programs)

    condition = When(Eq("device_type", device), Eq("dtype", torch.float32),
                     Eq("batch_size", 1), Eq("tp_size", 1 if encoder else 2), Eq("sp_size", 1))
    if encoder:
        condition = When(condition, Eq("graph_walk", "prefill"))
    else:
        condition = When(condition, AllRows(SAMPLER, Eq("temperature", 0.0)),
            Or(When(Eq("graph_walk", "prefill"), Eq("token_lengths", (4,))),
               When(Eq("graph_walk", "decode"), Eq("token_lengths", (1,)))))
    return Acceleration(WhisperEncoderSubmodule if encoder else WhisperDecoderSubmodule,
        condition, Implementation(prepare,
            resources={} if encoder else {key: DenseSequenceView() for key in (KV_CACHE, CROSS_KV_CACHE)},
            cost=lambda facts: 1.0 if protocol == "all_reduce" else 2.0))


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="CPU Gloo is unavailable")
def test_central_consumer_runs_both_real_roles_and_collective_protocols(tmp_path, monkeypatch):
    source = tmp_path / "corpus"
    source.mkdir()
    for name, node, protocol, device in (
        ("audio_encoder", "audio_encoder", "all_reduce", "cpu"),
        ("decoder_all_reduce", "decoder", "all_reduce", "cpu"),
        ("decoder_gather", "decoder", "gather", "cpu"),
        ("other_target", "decoder", "all_reduce", "mstar_pi"),
    ):
        (source / (name + ".py")).write_text(
            "from tests.test_distributed_qualification import _declaration\n"
            f"acceleration = _declaration({node!r}, {protocol!r}, {device!r})\n")
    checkpoint = tmp_path / "synthetic-checkpoint"
    checkpoint.mkdir()
    audio = tmp_path / "synthetic-input"
    audio.write_bytes(b"test-only synthetic input identity\n")
    monkeypatch.setattr(distributed, "_managed", _synthetic_role)
    env = Environment('cpu', 'cpu', 'cpu', 'cpu', 2, 'gloo',
        'mstar_accelerators.backends.torch.runtime:TorchRuntime')
    # The automatic planner creates separate supported TP1 and TP2 placements;
    # neither backend nor host contains encoder/decoder-specific placement code.
    job = Job(_SyntheticCase(), env, 2, ('decoder',),
              ('decoder_all_reduce', 'decoder_gather'), ())
    report = distributed.execute(str(source), job, str(checkpoint), {'audio': str(audio)})
    encoder = distributed.execute(str(source), Job(_SyntheticCase(), env, 1,
        ('audio_encoder',), ('audio_encoder',), ()), str(checkpoint), {'audio': str(audio)})
    assert encoder['counts'] == {'PASS': 1, 'FAIL': 0, 'SKIP': 0}
    output = Path(os.environ.get("MSTAR_P3_TEST_REPORT_ROOT", tmp_path))
    output.mkdir(parents=True, exist_ok=True)
    (output / "cpu-consumer.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    assert report["counts"] == {"PASS": 2, "FAIL": 0, "SKIP": 0}, report
    assert len({role["pid"] for role in report["roles"].values()}) == 2
    for role in report["roles"].values():
        assert role["runtime_released"]
        for protocol in ("all_reduce", "gather"):
            stats = role["test_ffn"][protocol]
            assert stats["calls"] == stats["nonzero_contributions"] > 0
            assert stats["local_ffn_width"] == 10
    rows = {row["name"]: row for row in report["accelerations"]}
    for name in ("decoder_all_reduce", "decoder_gather"):
        row = rows[name]
        assert not row["details"]["nonparticipating"]
        names = {check["name"] for check in row["checks"]}
        assert names >= {"replicated_state_and_outputs", "trajectory.third_request_backpressure",
            "trajectory.backpressure_is_retryable", "trajectory.reuses_released_blocks",
            "trajectory.other_request_preserved.kv_cache",
            "trajectory.other_request_preserved.cross_kv_cache",
            "trajectory.retained_output_after_reassignment"}
        assert all(check["passed"] for check in row["checks"])


def test_role_aggregation_preserves_failure_and_missing_participation():
    passed = Result("decoder", Status.PASS, checks=({"name": "value", "passed": True},)).to_dict()
    failed = Result("decoder", Status.FAIL, checks=({"name": "value", "passed": False},)).to_dict()
    missing = Result("decoder", Status.SKIP, details={"code": "reference_assets_unavailable"}).to_dict()
    absent = Result("decoder", Status.SKIP, details={"code": "role_not_participating"}).to_dict()
    assert combine("decoder", {"0": passed, "1": failed}, scope="test").status is Status.FAIL
    assert combine("decoder", {"0": passed, "1": missing}, scope="test").status is Status.SKIP
    combined = combine("decoder", {"0": passed, "1": absent}, scope="test")
    assert combined.status is Status.PASS and combined.details["nonparticipating"] == ["1"]
