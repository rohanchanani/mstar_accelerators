"""Placement metadata is reusable; live plans/tensors/owned results are not."""
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
import torch

from mstar.engine.execution import Execution,InvocationFacts,LogicalInvocation
from mstar.model.contract import NodeContract,TensorInput,TensorOutput
from mstar_accelerators.compiler import bind_default_call
from mstar_accelerators.backends.torch import TorchProgram,prepare_torch_programs


@dataclass(frozen=True)
class Outputs:
    y:torch.Tensor
    rows:tuple[torch.Tensor,...]


def setup():
    contract=NodeContract((TensorInput('x','tensor_inputs.x',('tokens',2),torch.float32),
                          TensorInput('prefix','tensor_inputs.prefix',(2,),walks=('prefill',))),
                         (TensorOutput('y',('batch',),torch.int64),
                          TensorOutput('rows',(2,),torch.float32,per_request=True)),Outputs,'node.')
    uses=[]
    def logical_use(plan):uses.append(plan);return NS(plan=plan)
    binding=NS(device=torch.device('cpu'),dtype=torch.float32,services=None,
               graph_walks=('prefill','decode'),submodule=NS(logical_contract=lambda:contract))
    return binding,{'state':NS(logical_use=logical_use),'plain':NS()},uses


def invocation(walks,values,plans):
    inputs=[NS(input_seq_len=len(value),tensor_inputs={'x':value,'prefix':torch.tensor([3.,4.])}) for value in values]
    return LogicalInvocation('mixed' if len(set(walks))>1 else walks[0],tuple(range(len(values))),
                             inputs,None,plans,row_walks=walks)


def test_mixed_rows_keep_order_live_conversion_and_fresh_outputs_and_owner_uses():
    binding,resources,uses=setup();default=bind_default_call(binding,resources)
    x=torch.arange(4,dtype=torch.float32).reshape(2,2)
    noncontiguous=torch.arange(8,dtype=torch.float64).reshape(2,4)[:,::2]
    assert not noncontiguous.is_contiguous()
    plan1,plan2=object(),object()
    call=default.materialize(invocation(('decode','prefill'),(x,noncontiguous),{'state':plan1,'plain':plan2}),None)
    assert call.inputs.x[0] is x
    assert call.inputs.x[1].is_contiguous() and call.inputs.x[1].dtype==torch.float32
    torch.testing.assert_close(call.inputs.x[1],noncontiguous.float())
    assert call.inputs.prefix[0] is None and call.inputs.prefix[1].tolist()==[3.,4.]
    assert call.row_walks==('decode','prefill')
    assert call.plan_results.state.plan is plan1 and call.plan_results.plain is plan2
    assert uses==[plan1]
    default.zero(call,None)
    later=default.materialize(invocation(('decode',),(x,),{'state':plan2}),None)
    assert uses==[plan1,plan2] and later.outputs.y.data_ptr()!=call.outputs.y.data_ptr()
    assert call.outputs.y.shape==(2,) and len(call.outputs.rows)==2
    assert later.outputs.y.shape==(1,) and len(later.outputs.rows)==1
    assert later.inputs.prefix==() and call.outputs.y.tolist()==[0,0]
    assert all(row.tolist()==[0.,0.] for row in call.outputs.rows)
    with pytest.raises(ValueError,match='expected logical shape'):
        bad=invocation(('decode',),(torch.zeros(2,3),),{'state':object()})
        default.materialize(bad,None)
    assert uses==[plan1,plan2]
    with pytest.raises(KeyError,match='missing'):
        default.materialize(invocation(('decode',),(x,),{'missing':object()}),None)


@pytest.mark.parametrize('mode',('identical','distinct_alias','fresh','bad_field','bad_shape'))
def test_primitive_result_checks_and_exact_identity_copy_rule(mode):
    binding,resources,_=setup();seen=[]
    def run(call):
        call.outputs.y.fill_(9)
        seen.append(call.outputs.y._version)
        if mode=='bad_field':return {'wrong':call.outputs.y}
        source=(call.outputs.y if mode in ('identical','bad_shape') else
                call.outputs.y.view_as(call.outputs.y) if mode=='distinct_alias' else torch.tensor([7]))
        rows=(torch.zeros(3),) if mode=='bad_shape' else (torch.tensor([2.,4.]),)
        return {'y':source,'rows':rows}
    program=prepare_torch_programs(binding,resources,{'decode':TorchProgram(run,slots=1)})
    facts=InvocationFacts('decode',1,(1,));lease=program.acquire(facts)
    assert program.acquire(facts) is None
    call=program.materialize(invocation(('decode',),(torch.zeros(1,2),),{}),Execution(None,lease))
    if mode.startswith('bad'):
        with pytest.raises(ValueError,match='fields differ' if mode=='bad_field' else 'expected'):
            program.primitive(call,Execution(None,lease))
    else:
        result=program.primitive(call,Execution(None,lease))
        assert result is call.outputs and result.y.tolist()==[7 if mode=='fresh' else 9]
        assert result.y._version==seen[0]+(mode!='identical')
        assert result.rows[0].tolist()==[2.,4.]
    lease.close();program.close()
    with pytest.raises(RuntimeError,match='closed'):
        program.acquire(facts)


def test_tensor_subclass_retains_ordinary_conversion_dispatch():
    class Tagged(torch.Tensor):
        calls=[]
        @classmethod
        def __torch_function__(cls,func,types,args=(),kwargs=None):
            if func.__name__ in ('to','contiguous'):
                cls.calls.append(func.__name__)
            return super().__torch_function__(func,types,args,kwargs or {})
    binding,resources,_=setup();default=bind_default_call(binding,resources)
    value=torch.ones(1,2).as_subclass(Tagged)
    Tagged.calls.clear()
    call=default.materialize(invocation(('decode',),(value,),{}),None)
    assert Tagged.calls==['to','contiguous','to']
    assert isinstance(call.inputs.x[0],Tagged)
