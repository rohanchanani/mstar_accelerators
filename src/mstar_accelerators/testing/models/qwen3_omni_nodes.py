"""Thinker and audio cases through the ordinary engine and centralized policy."""
from collections import defaultdict
from types import SimpleNamespace
import gc
import time

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import ExecutingBatch
from mstar.engine.execution import LogicalInvocation
from mstar.engine.resources.step import StepContext
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.model.binding import NodeBinding
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.qwen3_omni.config import THINKER_KV, THINKER_POS, THINKER_SAMPLER
from mstar.worker.engine_manager import EngineManager
from mstar_accelerators.binding import static_facts
from mstar_accelerators.compiler import bind_tensor_result
from mstar_accelerators.registry import Registry
from mstar_accelerators.testing.harness import CasePlan, Comparison


class NodeCaseAdapter:
    def __init__(self, model, *, session, device, options, node):
        self.model, self.session = model, session
        self.device, self.options, self.node = torch.device(device), options, node
        self.dtype = getattr(torch, options.get('dtype', 'bfloat16'))
        self.walks = (('prefill_text', 'prefill_audio', 'prefill_vision', 'thinker_decode', 'thinker_mixed')
                      if node == 'Thinker' else ('prefill_audio',))

    def __call__(self, name, acceleration):
        group = SimpleNamespace(world_size=1, rank=0)
        binding = NodeBinding(self.model, self.model.create_submodule(self.node), self.node,
                              self.device, self.dtype, group, group, {},
                              self.model.get_node_resources(), graph_walks=self.walks)
        static = static_facts(binding)
        batches = self.options.get('batches', [1, 2, 8]) if self.node == 'Thinker' else [1]
        details = {'finite_domain_only': True, 'timings': [],
                   'reference': 'retained MStar model reference through the ordinary engine',
                   'cases': 'fixed seeded text, media and mixed walks; logical result/state comparison'}
        cache = None

        def execute(witness):
            nonlocal cache
            if cache is None:
                reference = self._run(None, batches, details)
                actual = self._run((name, acceleration), batches, details)
                cache = defaultdict(list)
                for key, expected in reference.items():
                    walk, label = key
                    label = f"{walk}/{label}"
                    observed = actual[key]
                    cache[walk].append(Comparison(label + '/fields', tuple(sorted(observed)),
                                                  tuple(sorted(expected)), exact=True))
                    for field, wanted in expected.items():
                        got = observed.get(field)
                        cache[walk].append(Comparison(label + '/' + field, got, wanted,
                            exact=not isinstance(wanted, torch.Tensor) or not wanted.is_floating_point(),
                            dtype=wanted.dtype if isinstance(wanted, torch.Tensor) else None))
            # A declaration may accept every walk without an explicit Or. Its
            # single solved alternative must still compare every executed case.
            return [comparison for comparisons in cache.values() for comparison in comparisons]

        return CasePlan(static, {'graph_walk': self.walks, 'batch_size': batches}, execute,
                        'retained model reference', 'actual registry execution and resource lifecycle', details)

    @torch.no_grad()
    def _run(self, candidate, batches, details):
        config = {'max_concurrent_requests': max(batches), 'autocast_dtype': 'bfloat16',
                  'resources': {THINKER_KV: {'max_bytes': 3 * 1024**3, 'prefix_cache': False}},
                  'runtime_extension': {'nodes': [self.node] if candidate else []}}
        registry = None if candidate is None else Registry({candidate[0]: candidate[1]})
        manager = EngineManager()
        self.session.managers.append(manager)
        manager.load({self.node}, self.device, config, WorkerParallelGroups(1, 0), None,
                     self.model, None, runtime_extension=None if registry is None else
                     SimpleNamespace(assemble=registry.assemble))
        manager.warmup_all()
        engine = manager.engine
        resources = engine._submodules[self.node].resources
        validate = None
        if candidate is not None:
            option = engine.execution_capabilities(self.node).options[0]
            validate = bind_tensor_result(option.runner.binding)
        saved = {}
        rng = torch.Generator().manual_seed(20261024)
        h = self.model.config.thinker_hidden_size
        for batch_size in batches:
            rids = list(range(10, 10 + batch_size))
            policies = {}
            for rid in rids:
                policy = SamplingReqConfig(temperature=0., repetition_penalty=1.)
                policy.apply_conductor_config(seed=rid, resource_key=THINKER_SAMPLER)
                policies[rid] = {THINKER_SAMPLER: policy} if self.node == 'Thinker' else {}
                manager.add_request(rid, policies[rid])
            for walk in self.walks:
                if walk == 'thinker_mixed' and batch_size == 1:
                    continue  # A combined walk requires different per-row walks.
                print(f'{self.node}: {"reference" if candidate is None else candidate[0]} '
                      f'batch={batch_size} walk={walk}', flush=True)
                infos, raw = {}, {}
                for row, rid in enumerate(rids):
                    row_walk = ('prefill_text' if row % 2 else 'thinker_decode') if walk == 'thinker_mixed' else walk
                    infos[rid] = CurrentForwardPassInfo(str(rid), row_walk, 0, rid, 32, rid_handle=rid,
                        resource_configs=policies[rid],
                        step_metadata={'is_last_prefill': row % 2 == 0, 'audio_output': row % 3 != 1})
                    if self.node == 'audio_encoder':
                        raw[rid] = {'audio_features': [torch.randn(128, 201, generator=rng).to(self.device)],
                                    'audio_seqlens': [torch.tensor([201], device=self.device)]}
                    elif row_walk in ('prefill_text', 'thinker_decode'):
                        ids = [872] if row_walk == 'thinker_decode' else [151644, 872, 198, 9707, 13][:3 + row % 3]
                        raw[rid] = {'text_inputs': [torch.tensor(ids, device=self.device)]}
                    elif row_walk == 'prefill_audio':
                        raw[rid] = {'audio_embeds': [torch.randn(3 + row % 2, h, generator=rng).to(self.device, self.dtype)]}
                    else:
                        raw[rid] = {'vision_embeds': [torch.randn(4, h, generator=rng).to(self.device, self.dtype)],
                            'image_grid_thw': [torch.tensor([[1, 4, 4]], device=self.device)],
                            'deepstack': [torch.randn(4, h, generator=rng).to(self.device, self.dtype)
                                          for _ in self.model.config.vision.deepstack_visual_indexes]}
                context = StepContext(rids, walk, 0, False,
                    request_walks={rid: info.graph_walk for rid, info in infos.items()}
                    if walk == 'thinker_mixed' else {})
                batch = ExecutingBatch(self.node, infos, context,
                                       per_request_input_tensors=raw)
                torch.cuda.synchronize(self.device)
                start = time.perf_counter()
                engine.prepare_inputs(batch)
                if batch.failed_requests or batch.skipped_rids:
                    raise RuntimeError(f'qualification input preparation failed: {batch.failed_requests}; '
                                       f'skipped: {batch.skipped_rids}')
                if candidate is not None:
                    engine.reserve_replay_slot(batch)
                    selected = batch.selected_execution
                    capabilities = engine.execution_capabilities(self.node)
                    option = capabilities.fixed_option if selected is None else selected.option
                    if option is None or option.name != candidate[0]:
                        raise RuntimeError(f'qualification did not select {candidate[0]} '
                            f'for {walk}/batch{batch_size}: {batch.admit_error}')
                outputs = engine.exec(batch)
                torch.cuda.synchronize(self.device)
                if batch.admit_error is not None:
                    raise RuntimeError(str(batch.admit_error))
                if batch.failed_requests:
                    raise RuntimeError(f'qualification execution failed: {batch.failed_requests}')
                details['timings'].append({'implementation': 'reference' if candidate is None else candidate[0],
                    'walk': walk, 'batch': batch_size, 'seconds': time.perf_counter() - start,
                    'captured': batch.step_context.slot_lease is not None,
                    'scope': 'single diagnostic sample including preparation; not a benchmark claim'})
                if validate is not None:
                    # Inspect owned production outputs before model postprocessing.
                    # The validator's reconstruction never replaces the serving result.
                    validate(outputs, LogicalInvocation(walk, rids, batch.inputs,
                        ModelInputsFromEngine(request_ids=rids, per_request_info=infos,
                                              resources=resources),
                        row_walks=tuple(infos[rid].graph_walk for rid in rids)))
                for rid in rids:
                    values = outputs.per_rid_outputs.get(rid, {})
                    # Apply the same model semantics before comparing graph-visible fields.
                    shell = engine._submodules[self.node].submodule
                    shell.postprocess(rid, infos[rid], values)
                    result = {field: value[0].detach().cpu().clone() for field, value in values.items() if value}
                    if self.node == 'Thinker':
                        result['kv_length'] = resources[THINKER_KV]._streams[rid]['main'].stored_len
                        result['position'] = resources[THINKER_POS]._counters[rid]['main']
                    saved[walk, f'b{batch_size}/r{rid}'] = result
            for rid in rids:
                manager.remove_request(rid)
        manager.shutdown()
        self.session.managers.remove(manager)
        self.model._submodule_cache.clear()
        del manager, engine, resources
        gc.collect()
        torch.cuda.empty_cache()
        return saved
