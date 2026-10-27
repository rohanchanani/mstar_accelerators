from dataclasses import replace
import pytest
import torch
from mstar.model.fidelity import resolve_fidelity
from mstar_accelerators.testing.corpus import MODELS
from mstar_accelerators.testing.harness import Comparison
from mstar_accelerators.testing.quality import QualityOutput, resolve_quality, score, quality_comparator


def spec():
    case = MODELS[0]
    return resolve_quality(case.describe(), case.deployment_config, case.identity())


def test_deployment_default_and_closed_language():
    assert resolve_fidelity({}).mode == 'numerical'
    for config in ({'mode':'unsafe'}, {'mode':'quality'}, {'mode':'numerical','rtol':1},
                   {'mode':'quality','at_least':'x','callback':'custom'}):
        with pytest.raises(ValueError): resolve_fidelity({'fidelity':config})


def test_baseline_applicability():
    case = MODELS[0]
    with pytest.raises(ValueError, match='checkpoint'):
        resolve_quality(case.describe(), case.deployment_config, {**case.identity(),'revision':'different'})


def test_exact_integer_quality_budget_and_completeness():
    q = spec()
    rows = [dict(id=s['id'],text=s['reference'],stop='eos') for s in q['samples']]
    rows[0]['text'] += ' unexpected' * 10
    assert score('q',QualityOutput(tuple(rows)),q)['passed']
    rows[0]['text'] += ' unexpected'
    assert not score('q',QualityOutput(tuple(rows)),q)['passed']
    assert not score('q',QualityOutput(tuple(rows[:-1])),q)['passed']
    rows[0]['text'] = q['samples'][0]['reference']
    rows[0]['stop'] = 'length'
    assert not score('q',QualityOutput(tuple(rows)),q)['passed']


def test_quality_cannot_waive_execution_obligations():
    c = Comparison('floating',torch.tensor([2.]),torch.tensor([1.]),promoted=lambda: (_ for _ in ()).throw(AssertionError('not numerical')))
    assert quality_comparator(None,c)['passed']
    assert not quality_comparator(None,replace(c,exact=True))['passed']
    assert not quality_comparator(None,replace(c,actual=torch.tensor([float('nan')])))['passed']
    assert not quality_comparator(None,replace(c,actual=torch.ones(2)))['passed']
    assert not quality_comparator(None,replace(c,actual=torch.ones(1,dtype=torch.bfloat16)))['passed']
    c = Comparison('mask',torch.tensor([float('-inf'),1.]),torch.tensor([float('-inf'),2.]),suppression_mask=torch.tensor([True,False]))
    assert quality_comparator(None,c)['passed']
    assert not quality_comparator(None,replace(c,actual=torch.ones(2)))['passed']


def test_continuation_copies_logical_tokens_between_different_page_layouts():
    from types import SimpleNamespace as NS
    from mstar.engine.resources.kv.cache import KVCache
    from mstar.engine.resources.kv.config import PagedKVConfig
    from mstar.model.whisper.config import KV_CACHE,CROSS_KV_CACHE
    from mstar_accelerators.testing.models.whisper_quality import _copy_committed
    def cache(size, dtype):
        return KVCache(PagedKVConfig(num_layers=1,num_kv_heads=1,head_dim=2,
                                    page_size=size,max_num_pages=4,max_seq_len=8),torch.device('cpu'),dtype)
    source,target,sp,tp={},{},{},{}
    ai=(torch.tensor([2,2,1]),torch.tensor([0,1,0]))
    bi=(torch.tensor([1,1,1]),torch.tensor([1,2,3]))
    values=torch.tensor([[[0.1001,0.2002]],[[0.3003,0.4004]],[[0.5005,0.6006]]],dtype=torch.float16)
    for key in (KV_CACHE,CROSS_KV_CACHE):
        a,b=cache(2,torch.float16),cache(4,torch.float32)
        a.write_tokens(0,values,-values,*ai)
        b.tensor.fill_(17.)
        source[key],target[key]=NS(kv_cache=a),NS(kv_cache=b)
        sp[key]={'main':NS(write_locations=NS(token_to_page=ai[0],token_to_cache=ai[1]))}
        tp[key]={'main':NS(write_locations=NS(token_to_page=bi[0],token_to_cache=bi[1]))}
    adapter=NS(_observe=lambda tensors:{k:v.clone() for k,v in tensors.items()},
               session=NS(retained=[]),stream=None)
    _copy_committed(adapter,source,sp,target,tp)
    for key in source:
        assert torch.equal(target[key].kv_cache.read_tokens(0,*bi),source[key].kv_cache.read_tokens(0,*ai).float())
        assert (target[key].kv_cache.tensor[:,0] == 17).all()
