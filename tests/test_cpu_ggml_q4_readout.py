"""Explicit vocabulary format/owner controls; central qualification is unchanged."""
import ctypes as C
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import weakref

import pytest
import torch

from mstar.engine.execution import Support
from mstar.engine.resources.sequence import PlainKVEncoding
from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE
from mstar_accelerators.implementations.whisper import cpu_ggml_decoder as tokens
from mstar_accelerators.implementations.whisper import cpu_ggml_readout_weights as weights
from tests.test_cpu_ggml_tokens import Owner, call, meaningful_weights
from tests.test_cpu_ggml_graph_readout import binding


def test_explicit_books_preserve_semantics_and_q8_default():
    assert tokens.CPUTokenDecoderAvailable().readout_format == 'q8_0'
    assert tokens.CPUTokenDecoderAvailable('q4_0').evaluate({}) is Support.UNKNOWN
    sentinel = object()
    with patch.object(weights, 'q8_weights', return_value=sentinel) as original:
        assert weights.prepared_weights(sentinel, 'decoder.') is sentinel
    original.assert_called_once_with(sentinel, 'decoder.')
    with pytest.raises(ValueError, match='format'):
        weights.prepared_weights(NS(), '', readout_format='q3')


def test_q4_weights_and_leases_are_shared_and_format_separate():
    b = NS(prepared={}, device=torch.device('cpu'),
           submodule=NS(config=NS(suppressed_tokens=lambda walk: (1,) if walk == 'prefill' else ())))
    source = {'decoder.embed_tokens.weight': torch.zeros(7,384),
              'decoder.proj_out.weight': torch.ones(7,384), 'decoder.fc.weight': torch.ones(384,384)}
    with patch.object(weights, 'prepare_weights', return_value=source), \
         patch.object(weights, 'column_major_weights', side_effect=lambda b,w: w), \
         patch.object(weights, 'pack_q4', return_value=torch.zeros(378,dtype=torch.int32)) as pack:
        a = weights.prepared_weights(b, 'decoder.', readout_format='q4_0')
        assert weights.prepared_weights(b, 'decoder.', readout_format='q4_0') is a
    assert pack.call_count == 1 and a[1]['decoder.embed_tokens.weight'] is source['decoder.embed_tokens.weight']
    resources = {name: NS(config=NS(encoding=PlainKVEncoding(torch.float16))) for name in (KV_CACHE,CROSS_KV_CACHE)}
    created=[]
    def create(*args, **kwargs):
        owner=NS(users=0, run=lambda call: call, close=lambda: None)
        created.append((kwargs['readout_format'],owner))
        return owner
    with patch.object(tokens, 'Decoder', side_effect=create):
        _, q8close = tokens.prepare_body(b,{},resources,None,None,(7,384))
        _, first = tokens.prepare_body(b,{},resources,None,None,(7,384),readout_format='q4_0')
        _, second = tokens.prepare_body(b,{},resources,None,None,(7,384),readout_format='q4_0')
    assert [v[0] for v in created] == ['q8_0','q4_0']
    q4=created[1][1]
    first();first();assert q4.users==1
    q4.close=lambda: (_ for _ in ()).throw(RuntimeError('busy'))
    with pytest.raises(RuntimeError,match='busy'):second()
    assert q4.users==1 and q4 in b.prepared.values()
    q4.close=lambda:None;second();second();q8close()
    assert all(v[1] not in b.prepared.values() for v in created)


native = pytest.mark.skipif(not os.environ.get('MSTAR_CPU_DECODER_LIBRARY'),reason='optional native artifact absent')


@native
def test_canonical_packer_extents_overlap_nonfinite_and_source_survival():
    b=binding(7)
    logical=torch.linspace(-1.71,1.83,7*384).reshape(7,384)
    saved=logical.clone();packed=weights.pack_q4(b,logical)
    assert torch.equal(logical,saved) and packed.numel()*4==7*12*18
    # The public C quantizer permits compiler contraction across x*id +8.5.
    # Select the established compiled contract explicitly, never by matching
    # whichever byte answer the packer happens to return. Target GCC uses FMLA;
    # the retained host Clang archive uses separate multiply/add rounding.
    rounding = os.environ.get('MSTAR_Q4_REFERENCE_ROUNDING')
    if rounding not in ('fma', 'separate'):
        raise ValueError('set MSTAR_Q4_REFERENCE_ROUNDING from the compiled public quantizer evidence')
    # Independent scalar model: signed first maximum, F16 scale, nibble pairs.
    x=logical.half().float().reshape(-1,32)
    index=x.abs().argmax(1,keepdim=True);maximum=x.gather(1,index).squeeze(1)
    d=maximum/-8;inv=torch.where(d!=0,1/d,torch.zeros_like(d))
    if rounding == 'fma':
        # For this F16 fixture, the <=35-bit product and 8.5 sum are exactly
        # representable in F64; one final cast models the native FMLA rounding.
        transformed=(x.double()*inv[:,None].double()+8.5).float()
    else:
        transformed=x*inv[:,None]+8.5
    q=transformed.to(torch.int32).clamp(max=15).to(torch.uint8)
    expected=torch.cat((d.half().view(torch.uint8).reshape(-1,2),q[:,:16]|(q[:,16:]<<4)),1).flatten()
    assert torch.equal(packed.view(torch.uint8),expected)
    api=weights.library(b,readout_format='q4_0');error=C.create_string_buffer(1024)
    def invoke(source=logical,output=packed,inbytes=None,outbytes=None):
        return api.cpu_decoder_pack_readout_q4(source.data_ptr(),weights.size(source) if inbytes is None else inbytes,
            output.data_ptr(),weights.size(output) if outbytes is None else outbytes,7,384,error,len(error))
    assert invoke(outbytes=weights.size(packed)-4) and b'extent' in error.value
    assert invoke(output=logical,outbytes=weights.size(packed)) and b'overlap' in error.value
    logical[0,0]=float('nan')
    with pytest.raises(RuntimeError,match='nonfinite'):weights.pack_q4(b,logical)
    logical[0,0]=1e10
    with pytest.raises(RuntimeError,match='overflows'):weights.pack_q4(b,logical)


@native
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_q4_native_token_and_storage_paths_match_and_keep_shared_state(dtype):
    rows=7;b=binding(rows);source=meaningful_weights(rows)
    logical=torch.sin(torch.arange(rows*384).reshape(rows,384)*.01)*.03
    packed=weights.pack_q4(b,logical);ref=weakref.ref(packed)
    resources={k:Owner(n,dtype) for k,n in ((KV_CACHE,448),(CROSS_KV_CACHE,1500))}
    control_resources={k:Owner(n,dtype) for k,n in ((KV_CACHE,448),(CROSS_KV_CACHE,1500))}
    masks={'prefill':torch.tensor([2]),'decode':torch.empty(0,dtype=torch.int64)}
    from tests.native_decoder_controls.cpu_ggml_graph_readout import GraphReadoutBody
    owner=tokens.Decoder(b,source,resources,packed,masks,(rows,384),readout_format='q4_0')
    control=GraphReadoutBody(binding(rows),source,control_resources,packed,masks,(rows,384),readout_format='q4_0')
    del packed
    retained=None
    try:
        for walk,length,page in [('prefill',4,1),('decode',5,1),('decode',6,1),('prefill',4,2),('decode',5,2)]:
            c=call(walk,length,page,rows)
            snapshots={k:r.kv_cache.tensor.clone() for k,r in resources.items()}
            a,token=owner.run(c);e,expected=control.run(c)
            torch.testing.assert_close(a,e,rtol=3e-5,atol=3e-5)
            assert torch.equal(token,expected)
            if retained is None:retained=token.clone()
            for name,r in resources.items():
                assert torch.equal(r.kv_cache.tensor,control_resources[name].kv_cache.tensor)
                assert torch.equal(r.kv_cache.tensor[:,[0,3-page]],snapshots[name][:,[0,3-page]])
                if name==KV_CACHE:assert torch.equal(r.kv_cache.tensor[:,page,:,length:],snapshots[name][:,page,:,length:])
                elif walk=='decode':assert torch.equal(r.kv_cache.tensor,snapshots[name])
        assert ref() is not None
        error=C.create_string_buffer(1024)
        assert owner.library.cpu_decoder_set_readout_q4(owner.handle,owner.packed.data_ptr(),
            weights.size(owner.packed),rows,384,error,len(error)) != 0
        assert b'configured once' in error.value
        before={k:r.kv_cache.tensor.clone() for k,r in resources.items()}
        owner.suppressed['decode']=torch.tensor([rows])
        with pytest.raises(RuntimeError,match='suppression index'):owner.run(call('decode',6))
        assert all(torch.equal(r.kv_cache.tensor,before[k]) for k,r in resources.items())
    finally:
        owner.close();control.close()
    assert ref() is None
