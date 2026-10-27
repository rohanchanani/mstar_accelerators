"""Native token entry mechanism controls, not model qualification."""
import ctypes as C
import os
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from mstar.engine.execution import Support
from mstar.engine.resources.attn.planned import AttentionPlan
from mstar.engine.resources.kv.config import ResolvedSequenceConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import KVPlanOutput, PagedIndptrs, SequenceView
from mstar.engine.resources.sequence import DenseSequenceView, PagedSequenceView, PlainKVEncoding
from mstar.engine.resources.sequence_view import AttentionUse, SequenceUse, DenseSequenceAccess
from mstar.model.whisper.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, POS
from mstar.model.whisper.results import DecoderResult
from mstar_accelerators.implementations.whisper import cpu_ggml_decoder as native
from mstar_accelerators.implementations.whisper import cpu_ggml_decoder_abi as abi
from tests.native_decoder_controls import cpu_ggml_graph_readout as old
from mstar_accelerators.implementations.whisper.pi_v3d.q8 import pack_q8_0
from tests.test_cpu_ggml_decoder import body_weights
from tests.test_cpu_ggml_graph_readout import binding


class Owner:
    # Exercise actual owner view dispatch and the actual Dense/Paged access
    # implementations; only arena allocation is a small local fixture.
    view = KVManager.view
    logical_use = KVManager.logical_use

    def __init__(self, capacity, dtype):
        encoding = PlainKVEncoding(dtype)
        self.config = ResolvedSequenceConfig(num_layers=4, num_kv_heads=6, head_dim=64,
            max_seq_len=capacity, page_size=capacity, max_num_pages=3, prefix_cache=False,
            dense=True, encoding=encoding, inherited_encoding=PlainKVEncoding(torch.float32))
        self.kv_cache = NS(tensor=torch.full((4,3,2,capacity,6,64), .019, dtype=dtype), encoding=encoding)


def uses(q,length,page=1,*,capacity=448,causal=True):
    v = SequenceView('row','main',[page],length,q if capacity==448 else capacity)
    indptr = PagedIndptrs(torch.tensor([0,q],dtype=torch.int32),torch.tensor([0,1],dtype=torch.int32),
        torch.tensor([page],dtype=torch.int32),torch.tensor([length],dtype=torch.int32))
    write = KVPlanOutput(indptr,[v])
    read = AttentionPlan(('row',),indptr.qo_indptr,(v,),causal,capacity)
    return SequenceUse.from_plan(write),AttentionUse.from_plan(read)


def call(walk,length,page=1,rows=7):
    from mstar_accelerators.implementations.whisper.decoder_call import Call, Inputs, Plans
    q=4 if walk=='prefill' else 1
    sw,sr=uses(q,length,page);cw,cr=uses(q,1500,page,capacity=1500,causal=False)
    inputs=Inputs((torch.arange(q,dtype=torch.int64)%rows,),
        (torch.sin(torch.arange(1500*384).reshape(1500,384)*.009),) if q==4 else ())
    return Call(walk,inputs,DecoderResult(torch.empty(1,dtype=torch.int64)),Plans(
        { 'main':sw },{ 'main':sr },{ 'main':cw },{ 'main':cr },
        {'main':torch.arange(length-q,length,dtype=torch.int64)},None),(walk,))


def test_same_dense_owner_legally_supplies_paged_planes_and_admitted_plans():
    owner=Owner(448,torch.float16)
    dense=owner.view(DenseSequenceView(encoding=owner.config.encoding))
    paged=owner.view(PagedSequenceView(page_size=448,encoding=owner.config.encoding))
    write,read=uses(1,9,page=2)
    d,=dense.for_attention(read)
    p=paged.for_attention(read)
    assert p is read._plan and paged.for_step(write) is write._plan
    for layer in range(4):
        assert d.keys[layer].data_ptr()==paged.layer(layer)[2,0].data_ptr()
        assert d.values[layer].data_ptr()==paged.layer(layer)[2,1].data_ptr()
    with pytest.raises(ValueError,match='page size'):
        owner.view(PagedSequenceView(page_size=128,encoding=owner.config.encoding))
    with pytest.raises(ValueError,match='encoding'):
        owner.view(PagedSequenceView(encoding=PlainKVEncoding(torch.float32)))


def test_new_books_preserve_reference_formats_and_shared_native_owner():
    assert native.CPUTokenDecoderAvailable().evaluate({}) is Support.UNKNOWN
    b = NS(device=torch.device('cpu'), prepared={}, model_config={
        'runtime_extension': {'cpu_ggml': {'library': '/missing-control.so', 'sha256': '0' * 64}}})
    with patch.object(abi, 'load_library', return_value=NS()):
        assert native.CPUTokenDecoderAvailable().evaluate({'binding': b}) is Support.NO
    resources={k:NS(config=NS(encoding=PlainKVEncoding(torch.float16))) for k in (KV_CACHE,CROSS_KV_CACHE)}
    events=[];owner=NS(users=0,run=lambda x:x,close=lambda:events.append('closed'));b=NS(prepared={})
    with patch.object(native,'Decoder',return_value=owner) as create:
        _,a=native.prepare_body(b,{},resources,None,None,(7,384))
        _,z=native.prepare_body(b,{},resources,None,None,(7,384))
    assert create.call_count==1
    a();a();assert not events
    owner.close=lambda:(_ for _ in ()).throw(RuntimeError('busy'))
    with pytest.raises(RuntimeError,match='busy'):z()
    assert owner in b.prepared.values()
    owner.close=lambda:events.append('closed');z();z()
    assert events==['closed'] and not b.prepared


def meaningful_weights(rows):
    torch.manual_seed(91);weights=body_weights()
    for name,w in weights.items():
        if w.ndim==2 and 'embed_' not in name:w.normal_(0,.006)
        elif name.endswith('.bias'):w.normal_(0,.003)
    weights['decoder.embed_tokens.weight']=torch.randn(rows,384)*.07
    weights['decoder.embed_positions.weight']=torch.sin(torch.arange(448*384).reshape(448,384)*.021)*.03
    return weights


@pytest.mark.skipif(not os.environ.get('MSTAR_CPU_DECODER_LIBRARY'),reason='optional native token artifact absent')
@pytest.mark.parametrize('dtype,rows,scale',[(torch.float16,7,False),(torch.float32,7,False),
    (torch.float16,51865,False),(torch.float16,7,True)])
def test_native_token_graph_matches_old_graph_and_preserves_other_pages_and_rows(dtype,rows,scale):
    weights=meaningful_weights(rows);projection=pack_q8_0(torch.randn(rows,384)*.04)
    masks={'prefill':torch.tensor([2]),'decode':torch.empty(0,dtype=torch.int64)}
    resources={k:Owner(n,dtype) for k,n in ((KV_CACHE,448),(CROSS_KV_CACHE,1500))}
    controls={k:Owner(n,dtype) for k,n in ((KV_CACHE,448),(CROSS_KV_CACHE,1500))}
    b=binding(rows);b.submodule.config.scale_embedding=scale
    cb=binding(rows);cb.submodule.config.scale_embedding=scale
    owner=native.Decoder(b,weights,resources,projection,masks,(rows,384))
    control=old.GraphReadoutBody(cb,weights,controls,projection,masks,(rows,384))
    held=saved=None
    try:
        for walk,length,page in [('prefill',4,1),('decode',5,1),('decode',6,1),('prefill',4,2),('decode',5,2)]:
            c=call(walk,length,page,rows)
            before={k:r.kv_cache.tensor.clone() for k,r in resources.items()}
            expected,token=control.run(c)
            with patch.object(DenseSequenceAccess,'for_attention',side_effect=AssertionError('dense masks built')),\
                    patch('torch.nn.functional.embedding',side_effect=AssertionError('Torch embedding executed')),\
                    patch('torch.arange',side_effect=AssertionError('Torch positions constructed')),\
                    patch('torch.full',side_effect=AssertionError('Torch mask constructed')):
                logits,actual=owner.run(c)
            torch.testing.assert_close(logits,expected,rtol=3e-5,atol=3e-5)
            assert torch.equal(actual,token)
            if held is None:held,saved=actual,actual.clone()
            for key,r in resources.items():
                assert torch.equal(r.kv_cache.tensor,controls[key].kv_cache.tensor)
                others=[x for x in range(3) if x!=page]
                assert torch.equal(r.kv_cache.tensor[:,others],before[key][:,others])
                if key==KV_CACHE:
                    assert torch.equal(r.kv_cache.tensor[:,page,:,length:],before[key][:,page,:,length:])
                elif walk=='decode':assert torch.equal(r.kv_cache.tensor,before[key])
        assert torch.equal(held,saved)
        owner.close();assert torch.equal(held,saved)
    finally:owner.close();control.close()


@pytest.mark.skipif(not os.environ.get('MSTAR_CPU_DECODER_LIBRARY'),reason='optional native token artifact absent')
def test_native_invalid_metadata_and_outputs_reject_before_any_owner_write():
    rows=7;weights=meaningful_weights(rows);projection=pack_q8_0(torch.randn(rows,384)*.04)
    resources={k:Owner(n,torch.float16) for k,n in ((KV_CACHE,448),(CROSS_KV_CACHE,1500))}
    owner=native.Decoder(binding(rows),weights,resources,projection,
        {'prefill':torch.tensor([2]),'decode':torch.empty(0,dtype=torch.int64)},(rows,384))
    raw=owner.library.cpu_decoder_token_run
    try:
        def snapshots():return [r.kv_cache.tensor.clone() for r in resources.values()]
        for field,value,error in [('self_page',3,'page'),('self_begin',0,'geometry'),
                ('logits',resources[KV_CACHE].kv_cache.tensor.data_ptr(),'overlap'),('token_bytes',4,'extent')]:
            before=snapshots()
            def bad(handle,argument,message,size):
                setattr(argument._obj,field,value);return raw(handle,argument,message,size)
            with patch.object(owner.library,'cpu_decoder_token_run',side_effect=bad):
                with pytest.raises(RuntimeError,match=error):owner.run(call('decode',5))
            assert all(torch.equal(a,b.kv_cache.tensor) for a,b in zip(before,resources.values()))
        before=snapshots();c=call('prefill',4);c.inputs.input_ids[0][0]=rows
        with pytest.raises(RuntimeError,match='token ID'):owner.run(c)
        assert all(torch.equal(a,b.kv_cache.tensor) for a,b in zip(before,resources.values()))
        from dataclasses import replace
        c=call('prefill',4);use=c.plan_results.cross_attn['main']
        other=use._plan.contexts[0]._replace(page_idxs=[2])
        c.plan_results.cross_attn['main']=AttentionUse.from_plan(replace(use._plan,contexts=(other,)))
        with pytest.raises(ValueError,match='cross write/read'):owner.run(c)
        assert all(torch.equal(a,b.kv_cache.tensor) for a,b in zip(before,resources.values()))
    finally:owner.close()
