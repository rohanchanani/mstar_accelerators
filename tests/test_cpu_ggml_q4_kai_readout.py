"""Explicit KAI physical projection; original policy is the qualification gate."""
import ctypes as C
import os
from types import SimpleNamespace as NS
from unittest.mock import patch
import weakref

import pytest
import torch

from mstar.engine.execution import Support
from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE
from mstar_accelerators.implementations.whisper import cpu_ggml_decoder as tokens
from tests.native_decoder_controls import cpu_ggml_graph_readout as readout
from mstar_accelerators.implementations.whisper import cpu_ggml_readout_weights as weights
from tests.test_cpu_ggml_tokens import Owner, call, meaningful_weights
from tests.test_cpu_ggml_graph_readout import binding


def test_explicit_kai_books_preserve_reference_resources_and_defaults():
    assert tokens.CPUTokenDecoderAvailable().readout_format=='q8_0'
    assert tokens.CPUTokenDecoderAvailable('q4_0_kai').evaluate({}) is Support.UNKNOWN


def test_kai_requires_additive_native_capability_and_plain_still_works():
    # Existing artifacts keep their default capabilities; only KAI rejects.
    def function(): pass
    api=NS(**{name:lambda:None for name in ('cpu_decoder_readout_abi','cpu_decoder_set_readout_q8',
        'cpu_decoder_storage_readout_run','cpu_decoder_set_readout_q4','cpu_decoder_pack_readout_q4')})
    api.cpu_decoder_readout_abi=lambda:1
    with patch.object(readout,'storage_library',return_value=api):
        assert readout.library(NS(),readout_format='q4_0') is api
        assert readout.library(NS(),readout_format='q4_0_kai') is None
        api.cpu_decoder_set_readout_q4_kai=function
        assert readout.library(NS(),readout_format='q4_0_kai') is api


def test_canonical_pack_reused_and_native_lease_format_separate():
    from mstar.engine.resources.sequence import PlainKVEncoding
    b=NS(prepared={},device=torch.device('cpu'),submodule=NS(config=NS(suppressed_tokens=lambda walk:())))
    source={'decoder.embed_tokens.weight':torch.zeros(7,384),'decoder.proj_out.weight':torch.ones(7,384)}
    canonical=torch.zeros(378,dtype=torch.int32)
    with patch.object(weights,'prepare_weights',return_value=source),patch.object(weights,'column_major_weights',side_effect=lambda b,w:w),patch.object(weights,'pack_q4',return_value=canonical) as pack:
        actual=weights.prepared_weights(b,'decoder.',readout_format='q4_0_kai')
        assert actual is weights.prepared_weights(b,'decoder.',readout_format='q4_0_kai')
        assert actual[2] is canonical
    assert pack.call_count==1
    resources={name:NS(config=NS(encoding=PlainKVEncoding(torch.float16))) for name in (KV_CACHE,CROSS_KV_CACHE)}
    created=[]
    def create(*args,**kwargs):
        owner=NS(users=0,run=lambda call:call,close=lambda:None)
        created.append((kwargs['readout_format'],owner));return owner
    with patch.object(tokens,'Decoder',side_effect=create):
        _,plain=tokens.prepare_body(b,{},resources,None,None,(7,384),readout_format='q4_0')
        _,first=tokens.prepare_body(b,{},resources,None,None,(7,384),readout_format='q4_0_kai')
        _,second=tokens.prepare_body(b,{},resources,None,None,(7,384),readout_format='q4_0_kai')
    assert [x[0] for x in created]==['q4_0','q4_0_kai'];owner=created[1][1]
    first();first();assert owner.users==1
    owner.close=lambda:(_ for _ in ()).throw(RuntimeError('busy'))
    with pytest.raises(RuntimeError,match='busy'):second()
    assert owner.users==1 and owner in b.prepared.values()
    owner.close=lambda:None;second();second();plain()
    assert all(x[1] not in b.prepared.values() for x in created)


native=pytest.mark.skipif(not os.environ.get('MSTAR_CPU_DECODER_LIBRARY'),reason='optional actual KAI native artifact absent')


@native
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_native_kai_token_storage_state_lifetime_and_tail(dtype):
    rows=7;b=binding(rows);source=meaningful_weights(rows);logical=torch.sin(torch.arange(rows*384).reshape(rows,384)*.01)*.03
    packed=weights.pack_q4(b,logical);saved=packed.clone();retained=weakref.ref(packed)
    resources={k:Owner(n,dtype) for k,n in ((KV_CACHE,448),(CROSS_KV_CACHE,1500))}
    controls={k:Owner(n,dtype) for k,n in ((KV_CACHE,448),(CROSS_KV_CACHE,1500))}
    masks={'prefill':torch.tensor([2]),'decode':torch.empty(0,dtype=torch.int64)}
    owner=tokens.Decoder(b,source,resources,packed,masks,(rows,384),readout_format='q4_0_kai')
    control=readout.GraphReadoutBody(binding(rows),source,controls,packed,masks,(rows,384),readout_format='q4_0_kai')
    del packed
    try:
        for walk,length,page in [('prefill',4,1),('decode',5,1),('decode',17,1),('prefill',4,2),('decode',5,2)]:
            c=call(walk,length,page,rows);before={k:r.kv_cache.tensor.clone() for k,r in resources.items()}
            actual,token=owner.run(c);expected,other=control.run(c)
            torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-5)
            assert torch.equal(token,other) and int(torch.argmax(actual))==int(token)
            assert torch.isfinite(actual[-1]) # logical final row not an exposed packed duplicate
            for name,r in resources.items():
                assert torch.equal(r.kv_cache.tensor,controls[name].kv_cache.tensor)
                assert torch.equal(r.kv_cache.tensor[:,[0,3-page]],before[name][:,[0,3-page]])
                if name==KV_CACHE:assert torch.equal(r.kv_cache.tensor[:,page,:,length:],before[name][:,page,:,length:])
                elif walk=='decode':assert torch.equal(r.kv_cache.tensor,before[name])
        assert retained() is not None and torch.equal(retained(),saved)
        error=C.create_string_buffer(1024)
        assert owner.library.cpu_decoder_set_readout_q4_kai(owner.handle,owner.packed.data_ptr(),weights.size(owner.packed),rows,384,error,len(error))
        assert b'configured once' in error.value
        before={k:r.kv_cache.tensor.clone() for k,r in resources.items()};owner.suppressed['decode']=torch.tensor([rows])
        with pytest.raises(RuntimeError,match='suppression'):owner.run(call('decode',18,2,rows))
        assert all(torch.equal(r.kv_cache.tensor,before[k]) for k,r in resources.items())
    finally:
        owner.close();control.close()
    assert owner.handle is control.handle is None
