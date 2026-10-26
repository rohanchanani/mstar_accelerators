"""Bounded Qwen Talker trajectories through the actual registered engine runner.

This optional correctness consumer owns its short synthetic inputs and CPU
oracle. It neither changes serving support nor replaces submitted computation.
The engine owns every actual request, KV allocation, sampler and graph replay.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import ExecutingBatch
from mstar.engine.execution import InvocationFacts, Support
from mstar.engine.resources.sampler.config import SamplingReqConfig, sampling_policy_facts
from mstar.engine.resources.step import StepContext
from mstar.model.binding import NodeBinding
from mstar.model.qwen3_omni.config import CODE_PRED_SAMPLER, TALKER_KV, TALKER_POS, TALKER_SAMPLER
from mstar.model.qwen3_omni.logical import TalkerSubmodule
from mstar.worker.engine_manager import EngineManager

from mstar_accelerators.binding import invocation_facts, static_facts
from mstar_accelerators.registry import Registry
from mstar_accelerators.testing.harness import CasePlan, Comparison, Unavailable
from .qwen3_omni_oracle import TalkerOracle

WALKS = ('talker_prefill', 'talker_last_prefill', 'talker_decode')


def _configuration(options):
    config = {}
    if options.get('deployment'):
        import yaml
        config = yaml.safe_load(Path(options['deployment']).read_text())
    config = dict(config)
    config['max_concurrent_requests'] = max(options.get('batches', (1, 2, 8)))
    config['runtime_extension'] = {'nodes': ['Talker']}
    config.setdefault('autocast_dtype', options.get('dtype', 'bfloat16'))
    # This finite test uses the deployment budget when supplied; the default
    # holds all short requests and donor prefill capture buckets comfortably.
    config['resources'] = {**config.get('resources', {})}
    config['resources'].setdefault(TALKER_KV, {'max_bytes': 640 * 1024**2, 'prefix_cache': False})
    return config


def _policy(rid):
    values = {
        TALKER_SAMPLER: SamplingReqConfig(temperature=0., repetition_penalty=1.1),
        CODE_PRED_SAMPLER: SamplingReqConfig(temperature=0.),
    }
    for key, value in values.items():
        value.apply_conductor_config(seed=1729 + rid, resource_key=key)
    return values


def _offset(resource, rid):
    # Captured draws advance the authoritative device master. Reading the
    # eager host counter here would falsely report no RNG progress.
    buffers = resource._cg_buffers
    if buffers is not None and buffers.has_slot(rid):
        slot = buffers._rid_to_slot[rid]
        # A newly registered/recycled slot is logically zero until its next
        # draw installs the deferred reset. Prefill does not plan a sampler.
        if slot in buffers._pending_init or slot in buffers._pending_offset_reset:
            return 0
        return int(buffers.offset.master[slot].item())
    return int(resource._sampler._step_offset[rid])


class QualificationSession:
    """Retain live engine roots until the driver's ordinary device drain."""
    def __init__(self):
        self.managers = []

    def close(self):
        for manager in reversed(self.managers):
            manager.shutdown()
        self.managers.clear()


class TalkerCaseAdapter:
    def __init__(self, model, *, session, device, options=None, prepare_engine=None):
        self.model, self.session = model, session
        self.device = torch.device(device)
        self.options = dict(options or {})
        self.dtype = getattr(torch, self.options.get('dtype', 'bfloat16'))
        self.config = _configuration(self.options)
        self.batches = tuple(self.options.get('batches', (1, 2, 8)))
        self.decode_steps = self.options.get('decode_steps', 3)
        # Testing may install an explicitly labeled host replay backend before
        # warmup. The production driver never supplies this hook.
        self.prepare_engine = prepare_engine
        self.oracle = None

    def __call__(self, name, acceleration):
        group = SimpleNamespace(world_size=1, rank=0)
        shell = self.model.create_submodule('Talker')
        binding = NodeBinding(self.model, shell, 'Talker', self.device, self.dtype,
                              group, group, self.config, self.model.get_node_resources(),
                              graph_walks=WALKS)
        static = static_facts(binding)
        traces = []
        details = {
            'preparation': 'exact registered implementation.prepare, ordinary EngineManager and selected.run',
            'case_policy': 'short-greedy-trajectories-v1',
            'batch_domain': list(self.batches), 'decode_steps': self.decode_steps,
            'sampling': 'greedy; layer-0 repetition penalty 1.1; exact codes and draw accounting',
            'finite_domain_only': True, 'trajectories': traces,
            'capture_policy': 'exercise requested decode recipes within their declared batch cap',
            'reference_storage': 'named logical weights on CPU FP32; no serving model implementation',
        }
        domains = {'graph_walk': WALKS, 'batch_size': self.batches}
        # Static nonmatches remain a solver SKIP without reading weights.
        if acceleration.when.evaluate(static) is Support.NO:
            return CasePlan(static, domains, lambda witness: (),
                            'independent eager CPU logical Talker', 'not executed', details)
        if acceleration.reference is not TalkerSubmodule:
            raise Unavailable('reference_adapter', 'the Talker adapter handles the logical Talker only')
        policies = {key: (sampling_policy_facts(value),) for key, value in _policy(10).items()}
        singleton = static | invocation_facts(InvocationFacts('talker_prefill', 1, (2,), policies,
                                                             row_walks=('talker_prefill',)))
        if acceleration.when.evaluate(singleton) is not Support.YES:
            raise Unavailable('trajectory_reference',
                              'this bounded adapter needs the declaration to construct its own prefill history')
        cache = None

        def execute(witness):
            nonlocal cache
            if cache is None:
                if self.oracle is None:
                    self.oracle = TalkerOracle(self.model)
                cache = self._execute(name, acceleration, static, traces)
            return cache[witness['graph_walk']]

        return CasePlan(static, domains, execute,
                        'independent eager CPU FP32 Talker and dense per-request KV',
                        f'actual registered program through EngineManager on {self.device}', details)

    @torch.no_grad()
    def _execute(self, name, acceleration, static, traces):
        registry = Registry({name: acceleration})
        manager = EngineManager()
        self.session.managers.append(manager)  # retain partial construction on error
        manager.load({'Talker'}, self.device, self.config, WorkerParallelGroups(1, 0),
                     None, self.model, None,
                     runtime_extension=SimpleNamespace(assemble=registry.assemble))
        engine = manager.engine
        if self.prepare_engine is not None:
            self.prepare_engine(engine)
        manager.warmup_all()
        management = engine._submodules['Talker']
        resources, shell = management.resources, management.submodule
        checks = defaultdict(list)
        rng = torch.Generator().manual_seed(20261024)
        initial_free = resources[TALKER_KV]._arena.allocator.num_free

        for batch_size in self.batches:
            rids = list(range(10, 10 + batch_size))
            facts = {**static, **invocation_facts(InvocationFacts(
                'talker_decode', batch_size, (1,) * batch_size,
                {key: tuple(sampling_policy_facts(_policy(rid)[key]) for rid in rids)
                 for key in (TALKER_SAMPLER, CODE_PRED_SAMPLER)},
                row_walks=('talker_decode',) * batch_size))}
            if acceleration.when.evaluate(facts) is not Support.YES:
                traces.append({'batch_size': batch_size, 'executed': False,
                               'reason': 'additional consumer batch is outside authored support'})
                continue
            record = {'batch_size': batch_size, 'executed': True, 'steps': []}
            traces.append(record)
            for rid in rids:
                self.oracle.reset(rid)
                manager.add_request(rid, _policy(rid))
            previous, retained = {}, []
            active = list(rids)

            def run(walk, raw, label):
                infos = {rid: CurrentForwardPassInfo(f'qualification-{rid}', walk, 0, 0, 64,
                                                    rid_handle=rid) for rid in active}
                batch = ExecutingBatch('Talker', infos, StepContext(list(active), walk, 0, False),
                                       per_request_input_tensors=raw)
                engine.prepare_inputs(batch)
                actual_facts = {**static, **invocation_facts(InvocationFacts(
                    walk, len(active), tuple(row.input_seq_len for row in batch.inputs),
                    engine.resource_facts('Talker', active), row_walks=(walk,) * len(active)))}
                if acceleration.when.evaluate(actual_facts) is not Support.YES:
                    raise ValueError(f'{name}: qualification trajectory {label} is outside authored when')
                expected = [self.oracle.step(walk, rid, row, repetition_penalty=1.1)
                            for rid, row in zip(active, batch.inputs, strict=True)]
                # Pin through the same public selection/geometry entry point
                # used by serving; retain the actual identity before exec releases
                # the selection. Never infer execution identity from the test input.
                engine.reserve_replay_slot(batch)
                selected = batch.selected_execution
                if selected is None:
                    raise RuntimeError(f'{name}: engine selected no physical implementation')
                identity = selected.option.name
                physical_runner = selected.option.runner
                graph_runner = None if physical_runner is None else physical_runner.graph_runner
                capture_cap = None if graph_runner is None else graph_runner.max_batch_size_for(walk)
                capture_expected = (self.device.type == 'cuda' and walk == 'talker_decode'
                                    and graph_runner is not None and graph_runner.captures_walk(walk)
                                    and capture_cap is not None and len(active) <= capture_cap)
                output = engine.exec(batch)
                if batch.admit_error is not None:
                    raise RuntimeError(f'{name}: actual engine admission failed: {batch.admit_error}')
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                lease = batch.step_context.slot_lease
                captured = lease is not None
                record['steps'].append({'walk': walk, 'label': label, 'rows': len(active),
                                        'path': 'captured' if captured else 'eager',
                                        'implementation': identity, 'capture_expected': capture_expected,
                                        'padded_rows': len(active) if lease is None else lease.bucket.bs})
                prefix = f'batch{batch_size}/{label}'
                target = checks[walk]
                target.append(Comparison(prefix + '/selected_program', identity, name, exact=True))
                if captured:
                    target.append(Comparison(prefix + '/captured_program', lease.bucket.execution_name,
                                             identity, exact=True))
                if capture_expected:
                    target.append(Comparison(prefix + '/captured_decode', captured, True, exact=True))
                for row, rid in enumerate(active):
                    ref = expected[row]
                    stem = prefix + f'/request{rid}'
                    kv = resources[TALKER_KV]
                    state = kv._streams[rid]['main']
                    target.extend((
                        Comparison(stem + '/kv_length', state.stored_len, ref.kv_length, exact=True),
                        Comparison(stem + '/position', resources[TALKER_POS]._counters[rid]['main'],
                                   ref.position, exact=True),
                        Comparison(stem + '/sampler_draws',
                                   tuple(_offset(resources[key], rid) for key in
                                         (TALKER_SAMPLER, CODE_PRED_SAMPLER)),
                                   ref.sampler_offsets, exact=True),
                    ))
                    positions = torch.arange(ref.kv_length, device=self.device)
                    pages = torch.tensor(state.page_indices, device=self.device)[positions // kv.config.page_size]
                    for layer, (keys, values) in enumerate(self.oracle.kv(rid)):
                        cache = kv.layer_view(layer)
                        for component, index, wanted in (('k', 0, keys), ('v', 1, values)):
                            observed = cache[pages, index, positions % kv.config.page_size].detach().cpu().clone()
                            target.append(Comparison(stem + f'/layer{layer}/{component}', observed,
                                                     wanted.clone(), dtype=self.dtype))
                    if ref.result is None:
                        continue
                    values = output.per_rid_outputs[rid]
                    codec = values['codec_tokens'][0].detach().cpu().clone()
                    embeds = values['talker_input_embeds'][0].detach().cpu().clone()
                    token = values['new_token'][0].detach().cpu().clone()
                    target.extend((
                        Comparison(stem + '/codec_tokens', codec, ref.result.codec_tokens[0], exact=True),
                        Comparison(stem + '/token', token, ref.result.tokens[0], exact=True),
                        Comparison(stem + '/codec_sum', embeds, ref.result.codec_embeds, dtype=torch.float32),
                    ))
                    mask = resources[TALKER_SAMPLER]._sampler.get_token_mask(rid)._seen_token_mask
                    seen = tuple(mask.nonzero().flatten().detach().cpu().tolist())
                    target.append(Comparison(stem + '/seen_tokens', seen, tuple(sorted(ref.seen_tokens)), exact=True))
                    previous[rid] = values['talker_input_embeds'][0]
                    retained.append((values['codec_tokens'][0], codec, walk, stem))

            raw = {}
            for rid in active:
                length = 2 + rid % 2
                raw[rid] = {'thinker_states': [torch.randn(length, self.model.config.thinker_hidden_size,
                                                           generator=rng).to(self.device, self.dtype)],
                            'thinker_mask': [torch.tensor([False, True] + ([False] if length == 3 else []),
                                                        device=self.device)]}
            run('talker_prefill', raw, 'prefill')
            run('talker_last_prefill', {rid: {'thinker_states': [torch.randn(
                1, self.model.config.thinker_hidden_size, generator=rng).to(self.device, self.dtype)]}
                                      for rid in active}, 'last-prefill')
            for iteration in range(self.decode_steps):
                active = rids[::-1] if iteration % 2 else list(rids)
                raw = {rid: {'talker_input_embeds': [previous[rid]]} for rid in active}
                if iteration == 0:
                    for rid in active:
                        raw[rid]['thinker_states'] = [torch.randn(
                            1, self.model.config.thinker_hidden_size, generator=rng).to(self.device, self.dtype)]
                run('talker_decode', raw, f'decode{iteration}')
            # Remove one request, then replay the surviving state in
            # another row order. No second allocator or sampler is introduced.
            # Seven survivors from the eight-row case replay an eight-row
            # bucket, exercising real Talker padding, 15 draws and remapping.
            removed = {rids[0]} if len(rids) > 1 else set()
            for rid in removed:
                manager.remove_request(rid)
                self.oracle.reset(rid)
            if removed:
                active = [rid for rid in reversed(rids) if rid not in removed]
                run('talker_decode', {rid: {'talker_input_embeds': [previous[rid]]} for rid in active},
                    'survivors-after-removal')
            for tensor, snapshot, walk, stem in retained:
                checks[walk].append(Comparison(stem + '/retained_output', tensor.detach().cpu().clone(),
                                               snapshot, exact=True))
            for rid in rids:
                if rid not in removed:
                    manager.remove_request(rid)
                    self.oracle.reset(rid)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            resources[TALKER_KV].assert_pages_conserved()
            checks['talker_decode'].extend((
                Comparison(f'batch{batch_size}/all_pages_released',
                           resources[TALKER_KV]._arena.allocator.num_free, initial_free, exact=True),
                Comparison(f'batch{batch_size}/logical_conditioning_released',
                           bool(set(rids) & shell._eos_embed_sent), False, exact=True),
            ))
        return checks


def adapters(model, **kwargs):
    from .qwen3_omni_nodes import NodeCaseAdapter
    from mstar.model.qwen3_omni.thinker_logical import ThinkerLogical
    from mstar.model.qwen3_omni.audio_logical import AudioEncoderLogical
    return {TalkerSubmodule: TalkerCaseAdapter(model, **kwargs),
            ThinkerLogical: NodeCaseAdapter(model, node="Thinker", **kwargs),
            AudioEncoderLogical: NodeCaseAdapter(model, node="audio_encoder", **kwargs)}
