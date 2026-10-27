"""Independent HF computation inside M*'s callable runner and owned KV state."""
from dataclasses import replace
from types import SimpleNamespace
import torch
from mstar.distributed.communication import CommGroup
from mstar.engine.resources.binding import resolve_bound_specs
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.model.binding import NodeBinding
from mstar.model.qwen3_omni.config import TALKER_KV, TALKER_SAMPLER, CODE_PRED_SAMPLER
from mstar.model.qwen3_omni.results import TalkerResult
from ..reference_execution import ReferenceExecution
from mstar.engine.resources.sampler.resource import SamplerResource
from .qwen3_omni_hf import HuggingFaceTalkerOracle
from .qwen3_omni_oracle import _Request


def read_history(resource, invocation, key, row):
    view = invocation.plan_results[key]['main'].views[row]
    n = view.length - view.to_compute
    device = resource.layer_view(0).device
    positions = torch.arange(view.start, view.start + n, device=device)
    pages = torch.tensor(view.page_idxs, device=device)[positions // resource.config.page_size]
    slots = positions % resource.config.page_size
    return [(resource.layer_view(i)[pages, 0, slots], resource.layer_view(i)[pages, 1, slots])
            if n else None for i in range(resource.config.num_layers)]


def write_history(resource, invocation, key, row, history):
    view = invocation.plan_results[key]['main'].views[row]
    device = resource.layer_view(0).device
    positions = torch.arange(view.start + view.length - view.to_compute,
                             view.start + view.length, device=device)
    pages = torch.tensor(view.page_idxs, device=device)[positions // resource.config.page_size]
    slots = positions % resource.config.page_size
    for i, (k, v) in enumerate(history):
        layer = resource.layer_view(i)
        layer[pages, 0, slots] = k[-view.to_compute:].to(layer)
        layer[pages, 1, slots] = v[-view.to_compute:].to(layer)


def reference_binding(model, node, device, dtype, walks):
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, {'resources': {
        spec.resource_key: {'max_bytes': 3 * 1024**3, 'prefix_cache': False}
        for spec in specs if hasattr(spec, 'config') and hasattr(spec.config, 'num_kv_heads')}})
    specs = resolve_bound_specs(specs, {}, dtype=dtype)
    return NodeBinding(model, model.create_submodule(node), node, torch.device(device), dtype,
                       CommGroup.trivial(), CommGroup.trivial(), {'max_concurrent_requests': 8},
                       specs, graph_walks=walks)


class TalkerReference:
    def __init__(self, model, session, device, dtype, policy):
        self.oracle = HuggingFaceTalkerOracle(model, device=device, dtype=dtype)
        self.policy = policy
        self.teacher = None
        self.last = {}
        self.binding = reference_binding(model, 'Talker', device, dtype,
                            ('talker_prefill', 'talker_last_prefill', 'talker_decode'))
        self.execution = ReferenceExecution(self.binding, self, session,
                  providers={key: self._sampler_provider for key in (TALKER_SAMPLER, CODE_PRED_SAMPLER)})

    def _sampler_provider(self, spec, info):
        groups = self.oracle.config.num_code_groups
        def sample(logits, plan):
            # Teacher-forced history is a case input, shared by all three runs.
            # The runtime still owns applying the returned tokens and RNG step.
            group = 0 if spec.resource_key == TALKER_SAMPLER else self.depth_group
            if group:
                self.depth_group += 1
            return (logits.argmax(-1) if self.teacher is None else
                    self.teacher[group:group+1].to(logits.device, torch.long))
        return SamplerResource(spec.vocab_size, spec.enable_repetion_penalty, info.device,
            info.joint_comm_group, spec.enable_min_p, planned=True, sample=sample,
            offset_stride=groups-1 if spec.resource_key == CODE_PRED_SAMPLER else 1)

    def bind_resources(self, resources): self.resources = resources

    def reset(self, rid):
        self.last.pop(rid, None)

    def add(self, rid): self.execution.add_request(rid, self.policy(rid))
    def remove(self, rid):
        self.execution.remove_request(rid)
        self.last.pop(rid, None)

    def seen(self, rid):
        mask = self.resources[TALKER_SAMPLER]._sampler.get_token_mask(rid)._seen_token_mask
        return set(mask.nonzero().flatten().cpu().tolist())

    def kv(self, rid):
        resource = self.resources[TALKER_KV]
        state = resource._streams[rid]['main']
        view = SimpleNamespace(length=state.stored_len, to_compute=0, start=0, page_idxs=state.page_indices)
        invocation = SimpleNamespace(plan_results={TALKER_KV: {'main': SimpleNamespace(views=[view])}})
        return tuple(read_history(resource, invocation, TALKER_KV, 0))

    def run(self, invocation):
        results = []
        for row, (rid, inputs) in enumerate(zip(invocation.request_ids, invocation.inputs, strict=True)):
            kv = self.resources[TALKER_KV]
            view = invocation.plan_results[TALKER_KV]['main'].views[row]
            state = _Request(read_history(kv, invocation, TALKER_KV, row),
                length=view.length-view.to_compute, seen=self.seen(rid),
                talker_draws=self.resources[TALKER_SAMPLER]._sampler._step_offset[rid],
                predictor_draws=self.resources[CODE_PRED_SAMPLER]._sampler._step_offset[rid])
            self.oracle.requests[rid] = state
            try:
                result = self.oracle.step(invocation.graph_walk, rid, inputs,
                    repetition_penalty=1.1, teacher_codes=self.teacher,
                    sample=lambda logits, group: self.resources[
                        TALKER_SAMPLER if group == 0 else CODE_PRED_SAMPLER].sample([rid], logits=logits[None])[0])
                write_history(kv, invocation, TALKER_KV, row, state.kv)
                self.last[rid] = result
                results.append(result.result)
            finally:
                self.oracle.reset(rid)
        if results[0] is None:
            return None
        return TalkerResult(*(torch.cat([getattr(r, field) for r in results])
                              for field in ('codec_tokens', 'codec_embeds', 'tokens')))

    def step(self, walk, rid, raw, *, teacher_codes=None):
        self.teacher = teacher_codes
        self.depth_group = 1
        try:
            self.execution.run(walk, rid, raw)
        finally:
            self.teacher = None
        return self.last[rid]
