"""Test inputs lowered through the same shared compiler as serving."""
from types import SimpleNamespace

import torch

from mstar.engine.execution import Execution, LogicalInvocation
from mstar.engine.resources.kv.config import KVSpec
from mstar.model.binding import NodeBinding
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine
from mstar.model.whisper.whisper_model import WhisperModel
from mstar_accelerators.backends.graph import graph_bindings, paged_views
from mstar_accelerators.compiler import bind_default_call


def materialized_decoder_inputs(config, resources, plans, *, input_ids, encoder_states, prefill):
    model = WhisperModel.__new__(WhisperModel)
    model.config = config
    shell = model.create_submodule("decoder")
    specs = model.get_node_resources()
    for spec in specs:
        if isinstance(spec, KVSpec):
            spec.config = resources[spec.resource_key].config
    group = SimpleNamespace(world_size=1, rank=0)
    binding = NodeBinding(model, shell, "decoder", torch.device("cpu"), torch.float32,
                          group, group, {}, specs)
    defaults = bind_default_call(binding, resources)
    logical = ARNodeInputs(input_ids=input_ids, input_seq_len=input_ids.numel(),
                           tensor_inputs={} if encoder_states is None else {"encoder_states": encoder_states})
    invocation = LogicalInvocation("prefill" if prefill else "decode", [0], [logical],
                                   ModelInputsFromEngine([0], {}, resources), plans)
    return graph_bindings(defaults, defaults.materialize(invocation, Execution(None, None)),
                          views=paged_views(binding, resources))
