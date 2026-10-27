"""Model-owned Whisper reference executed and committed by M*.

The provider supplies CPU arithmetic; M* owns independent pages, positions,
request registration, plans, sampling effects, and release.
"""
from dataclasses import replace
import torch
from mstar.distributed.communication import CommGroup
from mstar.model.whisper.config import ATTN, CROSS_ATTN, KV_CACHE, CROSS_KV_CACHE, POS, SAMPLER
from mstar.model.whisper.results import EncoderResult, DecoderResult
from mstar.engine.resources.sampler.config import SamplingReqConfig
from ..reference_execution import ReferenceExecution
from .. import reference_resources as providers


class WhisperReference:
    def __init__(self, binding, program, session, *, specs=None):
        self.program, self.logits = program, None
        # Reference floating state uses reference precision, independently of an
        # accelerator's physical storage encoding. Preserve resolved geometry.
        from mstar.engine.resources.kv.config import KVSpec
        resolved = []
        for spec in (binding.resource_specs if specs is None else specs):
            if isinstance(spec, KVSpec) and hasattr(spec.config, 'encoding'):
                from mstar.engine.resources.kv.encoding import PlainKVEncoding
                spec = replace(spec, config=replace(spec.config,
                    encoding=PlainKVEncoding(binding.dtype), inherited_encoding=PlainKVEncoding(binding.dtype)))
            resolved.append(spec)
        specs = tuple(resolved)
        self.active = set()
        shell = binding.model.create_submodule(binding.node_name)
        binding = replace(binding, submodule=shell, device=torch.device('cpu'),
                          tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial(),
                          graph_walks=('prefill',) if binding.node_name == 'audio_encoder' else ('prefill', 'decode'),
                          resource_specs=binding.resource_specs if specs is None else specs,
                          prepared={})
        self.execution = ReferenceExecution(binding, self, session, providers={
            KV_CACHE: providers.paged_kv, CROSS_KV_CACHE: providers.paged_kv,
            ATTN: providers.attention, CROSS_ATTN: providers.cross_attention,
            POS: providers.positions, SAMPLER: providers.greedy_sampler})

    def bind_resources(self, resources):
        self.resources = resources
        for module in self.program.modules():
            bind = getattr(module, 'bind_resources', None)
            if bind is not None:
                bind(resources)

    def run(self, invocation):
        if invocation.graph_walk == 'prefill' and self.execution.binding.node_name == 'audio_encoder':
            return EncoderResult(tuple(self.program.run(x.tensor_inputs['audio_features']) for x in invocation.inputs))
        ids = torch.cat([x.input_ids for x in invocation.inputs])
        states = (torch.cat([x.tensor_inputs['encoder_states'] for x in invocation.inputs])
                  if invocation.graph_walk == 'prefill' else None)
        self.logits = self.program.run(invocation.graph_walk, invocation.engine_inputs, ids, states,
                                       return_logits=True).detach().clone()
        return DecoderResult(self.resources[SAMPLER].sample(invocation.request_ids, logits=self.logits))

    def invoke(self, walk, rid, ids=None, states=None, features=None):
        if rid not in self.active:
            cfg = SamplingReqConfig(temperature=0., repetition_penalty=1.)
            cfg.apply_conductor_config(seed=rid, resource_key=SAMPLER)
            self.execution.add_request(rid, {SAMPLER: cfg} if ids is not None else {})
            self.active.add(rid)
        raw = {'audio_features': [features]} if features is not None else {'text_inputs': [ids]}
        if states is not None:
            raw['encoder_states'] = [states]
        return self.execution.run(walk, rid, raw)

    def remove(self, rid):
        if rid in self.active:
            self.execution.remove_request(rid)
            self.active.remove(rid)
