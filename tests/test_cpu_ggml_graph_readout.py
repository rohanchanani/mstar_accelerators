"""Physical graph projection controls; central model qualification stays unchanged."""
import ctypes as C
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import weakref

import pytest
import torch

from mstar.engine.execution import Support
from mstar.engine.resources.sequence import PlainKVEncoding
from mstar.model.whisper.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, POS
from tests.native_decoder_controls import cpu_ggml_graph_readout as graph
from mstar_accelerators.implementations.whisper.pi_v3d.q8 import pack_q8_0
from tests.test_cpu_ggml_decoder import body_weights
from tests.test_cpu_ggml_storage import EncodedAccess


def test_optional_abi_is_unknown_before_binding_and_never_falls_back():
    assert graph.CPUGraphReadoutAvailable().evaluate({}) is Support.UNKNOWN
    with patch.object(graph, 'storage_library', return_value=NS()):
        assert graph.CPUGraphReadoutAvailable().evaluate({'binding': NS()}) is Support.NO


def test_books_keep_reference_formats_and_share_one_physical_owner():
    b = NS(prepared={})
    resources = {name: NS(config=NS(encoding=PlainKVEncoding(torch.float16))) for name in (KV_CACHE, CROSS_KV_CACHE)}
    events = []
    owner = NS(users=0, run=lambda call: call, close=lambda: events.append('closed'))
    with patch.object(graph, 'GraphReadoutBody', return_value=owner) as constructor:
        _, first = graph.prepare_body(b, {}, resources, None, None, (7,384))
        _, second = graph.prepare_body(b, {}, resources, None, None, (7,384))
    assert constructor.call_count == 1
    first();first();assert not events
    second();second();assert events == ['closed'] and not b.prepared


def test_close_failure_retains_owner_and_projection_for_retry():
    b=NS(prepared={});resources={name:NS(config=NS(encoding=PlainKVEncoding(torch.float16))) for name in (KV_CACHE,CROSS_KV_CACHE)}
    owner=NS(users=0,run=lambda call:call,close=lambda:(_ for _ in ()).throw(RuntimeError('busy')))
    with patch.object(graph,'GraphReadoutBody',return_value=owner):
        _,close=graph.prepare_body(b,{},resources,None,None,(7,384))
    with pytest.raises(RuntimeError,match='busy'):close()
    assert owner in b.prepared.values() and owner.users==1
    owner.close=lambda:None;close();assert not b.prepared


class PlainGraphBody(graph.GraphReadoutBody):
    quantized=False


def binding(rows):
    path=Path(os.environ['MSTAR_CPU_DECODER_LIBRARY'])
    return NS(device=torch.device('cpu'),prepared={},model_config={'runtime_extension':{'cpu_ggml':{
        'library':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'cpu_ggml_cross_threads':2}},
        submodule=NS(config=NS(d_model=384,decoder_layers=4,decoder_attention_heads=6,decoder_ffn_dim=1536,
            max_source_positions=1500,max_target_positions=448,scale_embedding=False,vocab_size=rows)))


def call(state,walk,length):
    q=4 if walk=='prefill' else 1
    state[KV_CACHE].plan(q,length);state[CROSS_KV_CACHE].plan(q,1500)
    state[CROSS_KV_CACHE].write.positions=torch.arange(1500,dtype=torch.int64)
    return NS(graph_walk=walk,inputs=NS(input_ids=(torch.zeros(q,dtype=torch.int64),),
        encoder_states=(torch.zeros(1500,384),) if q==4 else ()),plan_results=NS(**{
        POS:{'main':torch.arange(length-q,length)},KV_CACHE:{'main':None},ATTN:{'main':None},
        CROSS_KV_CACHE:{'main':None},CROSS_ATTN:{'main':None}}))


def quantized_dot(packed,rows,x):
    weights=packed.view(torch.uint8).reshape(rows,12,34)
    activation=pack_q8_0(x).view(torch.uint8).reshape(1,12,34)
    def split(t):
        return (t[:,:,:2].contiguous().view(torch.float16).reshape(t.shape[0],12).double(),
                t[:,:,2:].contiguous().view(torch.int8).double())
    dw,qw=split(weights);dx,qx=split(activation)
    return ((qw*qx).sum(-1)*dw*dx).sum(-1).float()


@pytest.mark.skipif(not os.environ.get('MSTAR_CPU_DECODER_LIBRARY'),reason='optional native graph artifact absent')
@pytest.mark.parametrize('rows,body_type',[(7,PlainGraphBody),(7,graph.GraphReadoutBody),(51865,graph.GraphReadoutBody)])
def test_native_last_query_projection_matches_quantized_operands_rebinds_and_rejects_before_writes(rows,body_type):
    torch.manual_seed(19)
    weights=body_weights()
    weights['decoder.layer_norm.bias']=torch.linspace(-.31,.37,384)
    weights['decoder.embed_positions.weight']=torch.sin(torch.arange(448*384,dtype=torch.float32).reshape(448,384)*.019)
    logical=torch.randn(rows,384)*.05
    logical[1]=logical[0]  # preserve exact equal rows through packing
    packed=pack_q8_0(logical);ref=weakref.ref(packed)
    state={KV_CACHE:EncodedAccess(448,torch.float16),CROSS_KV_CACHE:EncodedAccess(1500,torch.float16)}
    owner=body_type(binding(rows),weights,state,packed,{'prefill':torch.tensor([2]),'decode':torch.empty(0,dtype=torch.int64)},(rows,384))
    execute=owner.execute_native
    captured=[]
    def capture(actual,hidden,call):
        result=execute(actual,hidden,call)
        captured.append(hidden.clone())
        return result
    owner.execute_native=capture
    standalone=None
    if os.environ.get('MSTAR_CPU_Q8_LIBRARY'):
        standalone=C.CDLL(os.environ['MSTAR_CPU_Q8_LIBRARY']).q8cpu_readout
        standalone.argtypes=[C.c_void_p,C.c_size_t,C.c_int64,C.c_int64,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t]
        standalone.restype=C.c_int
    del packed;assert ref() is not None
    retained=saved=None
    try:
        for walk,length in [('prefill',4),('decode',5),('decode',6)]:
            logits,token=owner.run(call(state,walk,length))
            actual_expected=quantized_dot(owner.packed,rows,captured[-1])
            if standalone is not None:
                control_logits=torch.empty_like(logits);control_token=torch.empty(1,dtype=torch.int64)
                mask=owner.suppressed[walk]
                assert standalone(owner.packed.data_ptr(),graph.size(owner.packed),rows,384,captured[-1].data_ptr(),
                    control_logits.data_ptr(),control_token.data_ptr(),mask.data_ptr(),mask.numel())==0
                torch.testing.assert_close(logits,control_logits,rtol=3e-5,atol=3e-5)
                assert torch.equal(token,control_token)
            if walk=='prefill':actual_expected[2]=-float('inf')
            torch.testing.assert_close(logits,actual_expected,rtol=3e-5,atol=3e-5)
            assert token.item()==actual_expected.argmax().item()
            if retained is None:retained,saved=logits,logits.clone()
        def snapshots():return [t.clone() for r in state.values() for group in (r.keys,r.values) for t in group]
        before=snapshots();native=owner.library.cpu_decoder_storage_readout_run
        def wrong_extent(handle,argument,error,size):
            argument._obj.logits_bytes-=4
            return native(handle,argument,error,size)
        with patch.object(owner.library,'cpu_decoder_storage_readout_run',side_effect=wrong_extent):
            with pytest.raises(RuntimeError,match='extent'):owner.run(call(state,'decode',7))
        assert all(torch.equal(a,b) for a,b in zip(before,snapshots()))
        def output_alias(handle,argument,error,size):
            argument._obj.logits=argument._obj.body.kv[0].self_k
            return native(handle,argument,error,size)
        with patch.object(owner.library,'cpu_decoder_storage_readout_run',side_effect=output_alias):
            with pytest.raises(RuntimeError,match='overlap'):owner.run(call(state,'decode',7))
        assert all(torch.equal(a,b) for a,b in zip(before,snapshots()))
        owner.suppressed['decode']=torch.tensor([rows],dtype=torch.int64)
        with pytest.raises(RuntimeError,match='suppression index'):owner.run(call(state,'decode',7))
        assert all(torch.equal(a,b) for a,b in zip(before,snapshots()))
        owner.suppressed['decode']=torch.arange(2,rows,dtype=torch.int64)
        tied,token=owner.run(call(state,'decode',7))
        assert tied[0]==tied[1] and token.item()==0
        owner.suppressed['decode']=torch.arange(rows,dtype=torch.int64)
        logits,token=owner.run(call(state,'decode',7))
        assert torch.isneginf(logits).all() and token.item()==0
        owner.close();assert ref() is None
        assert torch.equal(retained,saved)
    finally:owner.close()


@pytest.mark.skipif(not os.environ.get('MSTAR_CPU_DECODER_LIBRARY'),reason='optional native graph artifact absent')
def test_native_nonfinite_hidden_rejected_before_quantization():
    rows=7;weights=body_weights();weights['decoder.layer_norm.bias'][0]=float('nan')
    state={KV_CACHE:EncodedAccess(448,torch.float16),CROSS_KV_CACHE:EncodedAccess(1500,torch.float16)}
    owner=PlainGraphBody(binding(rows),weights,state,pack_q8_0(torch.zeros(rows,384)),
        {'prefill':torch.empty(0,dtype=torch.int64),'decode':torch.empty(0,dtype=torch.int64)},(rows,384))
    try:
        with pytest.raises(RuntimeError,match='nonfinite'):owner.run(call(state,'decode',4))
    finally:owner.close()
