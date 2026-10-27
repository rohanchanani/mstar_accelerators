"""Prepared support must preserve authored three-valued live semantics."""
from itertools import product
from types import SimpleNamespace as NS

import pytest
from mstar.engine.execution import Support
from mstar_accelerators.conditions import All,AllRows,Constant,Eq,In,Or,Range
from mstar_accelerators.evaluation import prepare_evaluator


@pytest.mark.parametrize('condition',[
    All(),Or(),Eq('x',1),In('x',(1,3)),Range('x',1,3),
    All(Eq('x',1),All(Range('y',0,2),Eq('z',2))),
    Or(Eq('x',1),Or(Range('y',0,2),Eq('z',2))),
    All(Or(Eq('x',1),Eq('y',2)),Or(Eq('z',2),Constant(Support.UNKNOWN))),
    AllRows('sampler',All(Range('temperature',0,2),Eq('top_k',0))),
])
def test_all_partial_fact_combinations_match_original(condition):
    evaluate=prepare_evaluator(condition)
    for values in product((None,0,1,2,3),repeat=3):
        for omit in (False,True):
            facts={k:v for k,v in zip(('x','y','z'),values) if not omit or v is not None}
            for rows in (None,(),({'temperature':1,'top_k':0},),
                         ({'temperature':1,'top_k':0},{'temperature':None,'top_k':0}),
                         ({'temperature':1,'top_k':0},{'temperature':3,'top_k':0})):
                facts['resources']={'sampler':rows}
                assert evaluate(facts) is condition.evaluate(facts)


def test_custom_conditions_subclasses_and_eager_order_retained():
    calls=[]
    class Custom:
        def evaluate(self,facts):
            calls.append(facts['value'])
            return facts['value']
    class CustomEq(Eq):
        def evaluate(self,facts):return Support.UNKNOWN
    for factory in (All,Or):
        condition=factory(Constant(Support.NO),factory(Custom(),CustomEq('x',1)))
        evaluate=prepare_evaluator(condition)
        for value in Support:
            calls.clear();expected=condition.evaluate({'value':value,'x':1});before=list(calls)
            calls.clear();assert evaluate({'value':value,'x':1}) is expected
            assert calls==before
    class Broken:
        def evaluate(self,_facts):raise RuntimeError('live predicate failure')
    with pytest.raises(RuntimeError,match='live predicate failure'):
        prepare_evaluator(All(Constant(Support.NO),Broken()))({})


def test_mutated_rows_and_dynamic_values_are_never_cached():
    condition=All(Eq('batch_size',1),AllRows('sampler',Eq('temperature',0.)))
    evaluate=prepare_evaluator(condition)
    row={'temperature':0.};facts={'batch_size':1,'resources':{'sampler':[row]}}
    assert evaluate(facts) is Support.YES
    row['temperature']=.7;assert evaluate(facts) is Support.NO
    row.clear();assert evaluate(facts) is Support.UNKNOWN
    facts['batch_size']=2;assert evaluate(facts) is Support.NO
