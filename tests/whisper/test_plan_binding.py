import pytest
import torch

from mstar.engine.resources.attn.planned import AttentionPlan
from mstar.engine.resources.kv.plan import KVWritePlan, SequenceView
from mstar.model.weights import LogicalWeightSource
from mstar.model.whisper.config import WhisperModelConfig
from mstar.model.whisper.weights import weight_specs
from mstar_accelerators.backends.paged import read_rows, write_rows


def test_write_maps_cannot_mutate_the_padding_sink_or_alias_writes():
    with pytest.raises(ValueError, match="non-sink"):
        write_rows(KVWritePlan(torch.tensor([0]), torch.tensor([0]), 1), page_size=4, num_pages=5)
    with pytest.raises(ValueError, match="distinct"):
        write_rows(KVWritePlan(torch.tensor([1, 1]), torch.tensor([0, 0]), 2), page_size=4, num_pages=5)
    rows = write_rows(KVWritePlan(torch.tensor([1, 2]), torch.tensor([3, 0]), 2), page_size=4, num_pages=5)
    assert rows.keys.tolist() == [11, 16]
    assert rows.values.tolist() == [15, 20]


def test_causal_mask_and_padding_use_only_owned_context_and_zero_sink():
    view = SequenceView(7, "main", [2, 1], 6, 2, generation=19)
    plan = AttentionPlan((7,), torch.tensor([0, 2]), (view,), True, 4)
    rows = read_rows(plan, capacity=8, num_pages=4)
    assert rows.keys.tolist() == [16, 17, 18, 19, 8, 9, 0, 0]
    assert rows.values.tolist() == [20, 21, 22, 23, 12, 13, 4, 4]
    assert torch.isfinite(rows.mask).tolist() == [[True]*5 + [False]*3, [True]*6 + [False]*2]
    assert view.generation == 19


def test_model_owned_bias_defaults_and_tied_weight_identity():
    config = WhisperModelConfig(d_model=12, encoder_layers=1, decoder_layers=1,
                                encoder_attention_heads=3, decoder_attention_heads=2,
                                encoder_ffn_dim=17, decoder_ffn_dim=19,
                                max_source_positions=8, max_target_positions=12, vocab_size=23, num_mel_bins=4)
    specs = weight_specs(config)
    checkpoint = [(spec.source, torch.ones(spec.shape)) for spec in specs
                  if spec.alias_of is None and spec.default is None]
    source = LogicalWeightSource(specs, lambda: iter(checkpoint))
    values = source.load()
    assert values["decoder.proj_out.weight"] is values["decoder.embed_tokens.weight"]
    assert values["encoder.layers.0.self_attn.k_proj.bias"].count_nonzero() == 0
    assert values["decoder.layers.0.self_attn.k_proj.bias"].count_nonzero() == 0
    assert values["decoder.layers.0.encoder_attn.k_proj.bias"].count_nonzero() == 0
    assert values["encoder.layers.0.fc1.weight"].shape == (17, 12)
    assert values["decoder.layers.0.fc1.weight"].shape == (19, 12)


def test_typed_results_preserve_storage_and_model_owned_stopping():
    from types import SimpleNamespace

    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.engine.resources.sampler.config import SamplingReqConfig
    from mstar.engine.resources.sampler.plan import SamplingResult
    from mstar.model.submodule_base import BatchedModelOutput
    from mstar.model.whisper.config import SAMPLER
    from mstar.model.whisper.results import DecoderResult, EncoderResult
    from mstar.model.whisper.submodules import WhisperDecoderSubmodule, WhisperEncoderSubmodule

    config = WhisperModelConfig(eos_token_id=2)
    states = torch.randn(8, 12)
    encoder_result = WhisperEncoderSubmodule(config).execution_result(
        EncoderResult((states,)), SimpleNamespace(request_ids=[7]))
    assert encoder_result.outputs.per_rid_outputs[7]["encoder_states"][0] is states
    assert encoder_result.outputs.check_stop_buffers == {}
    assert encoder_result.outputs.get_check_stop_input() == {}
    shell = WhisperDecoderSubmodule(config)
    tokens = torch.tensor([2, 5])
    result = shell.execution_result(DecoderResult(tokens), SimpleNamespace(request_ids=[7, 8]))
    assert isinstance(result.outputs, BatchedModelOutput)
    assert isinstance(result.resource_results[SAMPLER], SamplingResult)
    assert result.resource_results[SAMPLER].tokens is tokens
    for row, rid in enumerate((7, 8)):
        output = result.outputs.per_rid_outputs[rid]
        token_view = output["new_token"][0]
        assert token_view.untyped_storage().data_ptr() == tokens.untyped_storage().data_ptr()
        assert token_view.storage_offset() == row
        info = CurrentForwardPassInfo("request", "decode", 0, 0, 3, rid_handle=rid,
                                      resource_configs={SAMPLER: SamplingReqConfig(temperature=0)})
        shell.postprocess(rid, info, output)
        assert output["text_inputs"] is output["new_token"]
        assert shell.check_stop(rid, info, output) == ({"decode_loop"} if row == 0 else set())
        info.resource_configs[SAMPLER].ignore_eos = True
        assert shell.check_stop(rid, info, output) == set()
        info.dynamic_loop_iter_counts["decode_loop"] = 2
        assert shell.check_stop(rid, info, output) == {"decode_loop"}
