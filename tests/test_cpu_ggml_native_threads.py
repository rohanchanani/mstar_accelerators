"""Physical provider pools are independent of the existing Torch thread policy."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from mstar_accelerators.implementations.whisper import cpu_ggml as cpu
from tests.native_decoder_controls import cpu_ggml_decoder as decoder


def binding(node, **extension):
    return NS(node_name=node, model_config={'runtime_extension':extension})


def test_two_native_pools_can_differ_without_changing_torch_or_artifact_identity():
    mapping={'audio_encoder':4,'decoder':2}
    artifact={'library':'/physical/library.so','sha256':'0'*64}
    with patch.object(cpu.torch,'get_num_threads',side_effect=AssertionError('explicit pool need not query Torch')), \
            patch.object(cpu.torch,'set_num_threads',side_effect=AssertionError('provider must not change Torch')):
        for node, count in mapping.items():
            b=binding(node,cpu_ggml_threads=mapping,cpu_ggml=artifact)
            assert cpu.native_threads(b)==count
            assert cpu.setting(b)==artifact
    assert mapping=={'audio_encoder':4,'decoder':2}


def test_absent_or_unmapped_node_keeps_current_torch_default():
    with patch.object(cpu.torch,'get_num_threads',return_value=3) as get:
        assert cpu.native_threads(binding('decoder'))==3
        assert cpu.native_threads(binding('decoder',cpu_ggml_threads={}))==3
        assert cpu.native_threads(binding('decoder',cpu_ggml_threads={'audio_encoder':4}))==3
    assert get.call_count==3


@pytest.mark.parametrize('value',[None,2,[],{'decoder':True},{'decoder':0},{'decoder':-1},
                                  {'decoder':65},{'decoder':2.0},{'decoder':'2'},{'':2},{2:2},
                                  {'audio_encoder':4,'another_node':False}])
def test_bad_mapping_rejected_before_encoder_or_decoder_library_load(value):
    b=binding('decoder',cpu_ggml_threads=value)
    with patch.object(cpu,'load_library',side_effect=AssertionError('loaded encoder before validating pools')), \
            patch.object(decoder,'library',side_effect=AssertionError('loaded decoder before validating pools')):
        with pytest.raises(ValueError,match='cpu_ggml_threads'):cpu.Encoder(b,{})
        with pytest.raises(ValueError,match='cpu_ggml_threads'):decoder.Body(b,{}, {})


def test_native_default_range_is_checked_before_preparation():
    with patch.object(cpu.torch,'get_num_threads',return_value=0):
        with pytest.raises(ValueError,match='thread count'):cpu.native_threads(binding('decoder'))
