"""Optional native attention selection/producer controls, no numerical oracle."""
from types import SimpleNamespace
import sys

import pytest

from mstar.model.whisper.config import WhisperModelConfig
from mstar_accelerators.implementations.whisper.pi_v3d.graph import _attention
from mstar_accelerators.implementations.whisper.pi_v3d.dense import dense_decoder_graph
from mstar_accelerators.implementations.whisper.pi_v3d.sdpa import decode_attention, q1_attention


@pytest.mark.parametrize('available,result', [(False, None), (True, q1_attention)])
def test_native_caps_choose_physical_path_from_both_contexts(monkeypatch, available, result):
    calls = []
    def supported(*shape):
        calls.append(shape)
        return available
    graph = SimpleNamespace(supports_sdpa_decode_f32=supported, sdpa_decode_f32=object())
    monkeypatch.setitem(sys.modules, 'mstar_pi', SimpleNamespace(ggml=SimpleNamespace(Graph=graph)))
    config = SimpleNamespace(decoder_attention_heads=6,d_model=384,max_target_positions=448,max_source_positions=1500)
    assert decode_attention(config) is result
    assert calls == ([(6,448,1),(6,1500,1)] if available else [(6,448,1)])


def test_missing_physical_api_falls_back_and_runtime_errors_propagate(monkeypatch):
    config = SimpleNamespace(decoder_attention_heads=6,d_model=384,max_target_positions=448,max_source_positions=1500)
    monkeypatch.setitem(sys.modules, 'mstar_pi', SimpleNamespace(ggml=SimpleNamespace(Graph=object())))
    assert decode_attention(config) is None
    def broken(*args): raise RuntimeError('native operational error')
    graph=SimpleNamespace(supports_sdpa_decode_f32=broken,sdpa_decode_f32=object())
    monkeypatch.setitem(sys.modules,'mstar_pi',SimpleNamespace(ggml=SimpleNamespace(Graph=graph)))
    with pytest.raises(RuntimeError,match='native operational error'):
        decode_attention(config)


class Trace:
    def __init__(self): self.calls=[]
    def __getattr__(self,name):
        def call(*args,**kwargs):
            index=len(self.calls)
            self.calls.append((name,args,kwargs))
            return index
        return call


def test_dense_optin_keeps_actual_set_rows_producers_and_default_graph():
    config=WhisperModelConfig(d_model=128,decoder_layers=2,encoder_layers=2,
        encoder_attention_heads=2,decoder_attention_heads=2,max_source_positions=19,max_target_positions=17)
    ordinary,equivalent,fused=Trace(),Trace(),Trace()
    dense_decoder_graph(config,graph=ordinary)
    dense_decoder_graph(config,graph=equivalent,attention=_attention)
    assert ordinary.calls == equivalent.calls
    dense_decoder_graph(config,graph=fused,attention=q1_attention)
    operations=[args for name,args,_ in fused.calls if name=='sdpa_decode_f32']
    assert len(operations)==2*config.decoder_layers
    for i,args in enumerate(operations):
        for view in args[1:3]:
            operation,view_args,_=fused.calls[view]
            assert operation=='reshape'
            source_operation,source_args,_=fused.calls[view_args[0]]
            if i%2==0:
                assert source_operation=='set_rows'
            else:
                assert source_operation=='tensor' and source_args[0].startswith('cross_attn.main.read.')
    assert sum(name=='set_rows' for name,_,_ in ordinary.calls)==sum(
        name=='set_rows' for name,_,_ in fused.calls)==2*config.decoder_layers
    assert sum(name=='argmax' for name,_,_ in fused.calls)==1


def test_q4_callback_uses_original_attention_graph():
    ordinary,fallback=Trace(),Trace()
    args=dict(heads=2,width=128,queries=4,keys=19,mask='mask')
    _attention(ordinary,'q','k','v',**args)
    q1_attention(fallback,'q','k','v',**args)
    assert ordinary.calls==fallback.calls
