"""Bounded Qwen Talker trajectories through the actual registered engine runner.

This central consumer owns short state trajectories, speech fixtures, and the
same-precision HF reference. It neither changes serving support nor replaces submitted computation.
The engine owns every actual request, KV allocation, sampler and graph replay.
"""
from __future__ import annotations

from collections import defaultdict
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
from .qwen_runtime_reference import TalkerReference
from ..generation import GreedyOutput, GreedySequence
from ..observations import SamplerObservation
from ..numerics import compare
from ..policy import QWEN_BATCHES, QWEN_DECODE_STEPS

WALKS = ('talker_prefill', 'talker_last_prefill', 'talker_decode')


def _configuration(options):
    config = {}
    config['max_concurrent_requests'] = max(QWEN_BATCHES)
    config['runtime_extension'] = {'nodes': ['Talker']}
    config.setdefault('autocast_dtype', options.get('dtype', 'bfloat16'))
    # The central finite-domain budget holds all short requests and prefill
    # capture buckets. M* resolves and allocates the requested resources.
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


from ..whisper import QualificationSession as EngineSession


class QualificationSession(EngineSession):
    """Retain live engine roots until the driver's ordinary device drain."""
    def __init__(self):
        super().__init__()
        self.managers = []

    def close(self):
        for manager in reversed(self.managers):
            manager.shutdown()
        self.managers.clear()
        super().close()


class TalkerCaseAdapter:
    def __init__(self, model, *, session, device, options=None, prepare_engine=None):
        self.model, self.session = model, session
        self.device = torch.device(device)
        self.options = dict(options or {})
        self.dtype = getattr(torch, self.options.get('dtype', 'bfloat16'))
        self.config = _configuration(self.options)
        self.batches = QWEN_BATCHES
        self.decode_steps = QWEN_DECODE_STEPS
        # Testing may install an explicitly labeled host replay backend before
        # warmup. The production driver never supplies this hook.
        self.prepare_engine = prepare_engine
        self.oracle = self.promoted = None
        self.history, self.promoted_results = [], None

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
            'case_policy': 'shared-history-generation-v2',
            'batch_domain': list(self.batches), 'decode_steps': self.decode_steps,
            'sampling': 'codec-sequence first-divergence reciprocal top-five; exact own argmax and draw accounting for all groups',
            'finite_domain_only': True, 'trajectories': traces,
            'capture_policy': 'exercise requested decode recipes within their declared batch cap',
            'reference_storage': f'stock HF on {self.device} at {self.dtype}; shared candidate history',
            'intermediate_kv_policy': 'numerical differences are diagnostics; shape, dtype, finiteness, retained prefix and lifecycle are gates',
        }
        domains = {'graph_walk': WALKS, 'batch_size': self.batches}
        # Static nonmatches remain a solver SKIP without reading weights.
        if acceleration.when.evaluate(static) is Support.NO:
            return CasePlan(static, domains, lambda witness: (),
                            'stock HF same-precision Talker', 'not executed', details)
        if acceleration.reference is not TalkerSubmodule:
            raise Unavailable('reference_adapter', 'the Talker adapter handles the logical Talker only')
        policies = {key: (sampling_policy_facts(value),) for key, value in _policy(10).items()}
        singleton = static | invocation_facts(InvocationFacts('talker_prefill', 1, (2,), policies,
                                                             row_walks=('talker_prefill',)))
        if acceleration.when.evaluate(singleton) is not Support.YES:
            raise Unavailable('trajectory_reference',
                              'this bounded adapter needs the declaration to construct its own prefill history')
        from .qwen_speech import fixtures
        self.speech_fixtures = fixtures(self.model, self.dtype)
        cache = None

        def execute(witness):
            nonlocal cache
            if cache is None:
                if self.oracle is None:
                    self.oracle = TalkerReference(self.model, self.session, self.device, self.dtype, _policy)
                cache = self._execute(name, acceleration, static, traces, details)
            return cache[witness['graph_walk']]

        return CasePlan(static, domains, execute,
                        'stock HF same-precision Talker and dense per-request KV',
                        f'actual registered program through EngineManager on {self.device}', details)

    def _higher(self, index, group):
        if self.promoted_results is None:
            self.promoted = TalkerReference(self.model, self.session, self.device, torch.float64, _policy)
            self.promoted_results = {}
            for i, event in enumerate(self.history):
                if event[0] == 'add':
                    self.promoted.add(event[1])
                elif event[0] == 'remove':
                    self.promoted.remove(event[1])
                else:
                    _, walk, rid, raw, codec = event
                    value = self.promoted.step(walk, rid, raw, teacher_codes=codec)
                    if value.result is not None:
                        self.promoted_results[i] = tuple(x.cpu() for x in
                            (value.talker_logits,) + value.predictor_logits)
        return self.promoted_results[index][group]

    @staticmethod
    def _suppression_mask(shell, walk, group, logits):
        mask = torch.zeros(logits.shape, dtype=torch.bool)
        if group == 0:
            ids = shell.logical_contract().suppressed_tokens[TALKER_SAMPLER][walk]
            mask[list(ids)] = True
        return mask

    @torch.no_grad()
    def _execute(self, name, acceleration, static, traces, details):
        registry = Registry({name: acceleration})
        manager = EngineManager()
        self.session.managers.append(manager)  # retain partial construction on error
        manager.load({'Talker'}, self.device, self.config, WorkerParallelGroups(1, 0),
                     None, self.model, None,
                     runtime_extension=SimpleNamespace(assemble=registry.assemble))
        engine = manager.engine
        if self.prepare_engine is not None:
            self.prepare_engine(engine)
        management = engine._submodules['Talker']
        resources, shell = management.resources, management.submodule
        observations = {key: SamplerObservation(resources[key])
                        for key in (TALKER_SAMPLER, CODE_PRED_SAMPLER)}
        manager.warmup_all()
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
                self.oracle.add(rid)
                self.history.append(('add', rid))
                manager.add_request(rid, _policy(rid))
            previous, retained, previous_kv = {}, [], {}
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
                samples = {key: observer.read(lease) for key, observer in observations.items()}
                sampling = walk != 'talker_prefill'
                if sampling:
                    target.extend((
                        Comparison(prefix + '/talker_sample_calls', len(samples[TALKER_SAMPLER]), 1, exact=True),
                        Comparison(prefix + '/predictor_sample_calls', len(samples[CODE_PRED_SAMPLER]),
                                   self.model.config.num_code_groups - 1, exact=True),
                    ))
                    if len(samples[TALKER_SAMPLER]) != 1 or len(samples[CODE_PRED_SAMPLER]) != self.model.config.num_code_groups - 1:
                        raise RuntimeError('sampler observations missing from the executed path')
                for row, rid in enumerate(active):
                    values = output.per_rid_outputs.get(rid, {})
                    codec = values['codec_tokens'][0].detach().cpu().clone() if sampling else None
                    prior_seen = self.oracle.seen(rid)
                    ref = self.oracle.step(walk, rid, raw[rid], teacher_codes=codec)
                    reference_index = len(self.history)
                    self.history.append(('step', walk, rid,
                        {key: [x.detach().cpu().clone() for x in tensors] for key, tensors in raw[rid].items()}, codec))
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
                    for layer, (keys, cache_values) in enumerate(self.oracle.kv(rid)):
                        cache = kv.layer_view(layer)
                        for component, index, wanted in (('k', 0, keys), ('v', 1, cache_values)):
                            observed = cache[pages, index, positions % kv.config.page_size].detach().cpu().clone()
                            key = (rid, layer, component)
                            label = stem + f'/layer{layer}/{component}'
                            target.append(Comparison(label, (tuple(observed.shape), observed.dtype,
                                                               bool(observed.isfinite().all())),
                                                     (tuple(wanted.shape), self.dtype, True), exact=True))
                            if key in previous_kv:
                                old = previous_kv[key]
                                target.append(Comparison(label + '/retained_prefix',
                                                         observed[:old.shape[0]], old, exact=True))
                            previous_kv[key] = observed
                            # A MoE routing near-tie changes later internal KV values.
                            # Keep this evidence without treating CPU/GPU trajectory
                            # identity as the semantic contract of the whole node.
                            record.setdefault('kv_numerical_diagnostics', []).append(
                                compare(label, observed, wanted.cpu(), dtype=self.dtype))
                    if ref.result is None:
                        continue
                    embeds = values['talker_input_embeds'][0].detach().cpu().clone()
                    token = values['new_token'][0].detach().cpu().clone()
                    observed_samples = samples[TALKER_SAMPLER] + samples[CODE_PRED_SAMPLER]
                    sampled_codes = torch.tensor([int(tokens[row]) for _, tokens in observed_samples], dtype=torch.long)
                    target.extend((
                        Comparison(stem + '/codec_tokens', codec, sampled_codes, exact=True),
                        Comparison(stem + '/token', token, sampled_codes[0], exact=True),
                        Comparison(stem + '/codec_sum', embeds, ref.result.codec_embeds.cpu(), dtype=torch.float32),
                    ))
                    generated, reference_generated, suppression_masks = [], [], []
                    for group, ((logits, tokens), wanted) in enumerate(zip(
                            observed_samples, (ref.talker_logits,) + ref.predictor_logits, strict=True)):
                        actual_logits = logits[row].clone()
                        if group == 0:
                            # The actual M* sampler loads BF16 readout values into
                            # FP32 before penalty arithmetic (sampler/kernels.py).
                            # Match that observed semantic boundary, preserving a
                            # separate exact check on the incoming representation.
                            target.append(Comparison(stem + f"/group{group}/readout_dtype",
                                str(actual_logits.dtype), str(self.dtype), exact=True))
                            actual_logits = actual_logits.float()
                        if group == 0 and prior_seen:
                            ids = torch.tensor(sorted(prior_seen), dtype=torch.long)
                            v = actual_logits[ids]
                            actual_logits[ids] = torch.where(v > 0, v / 1.1, v * 1.1)
                        generated.append(GreedyOutput(actual_logits, int(tokens[row])))
                        reference_generated.append(GreedyOutput(wanted.cpu(), int(wanted.argmax())))
                        suppression_masks.append(self._suppression_mask(shell, walk, group, wanted))
                    # The depth predictor emits one autoregressive codec frame.
                    # Teacher forcing agrees with both own histories through its
                    # first divergence. Later comparisons are conditional
                    # diagnostics; own-history continuations are tested below.
                    target.append(Comparison(stem + '/generation',
                        GreedySequence(tuple(generated)), GreedySequence(tuple(reference_generated)),
                        suppression_mask=tuple(suppression_masks)))
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
                self.oracle.remove(rid)
                self.history.append(('remove', rid))
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
                    self.oracle.remove(rid)
                    self.history.append(('remove', rid))
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            resources[TALKER_KV].assert_pages_conserved()
            checks['talker_decode'].extend((
                Comparison(f'batch{batch_size}/all_pages_released',
                           resources[TALKER_KV]._arena.allocator.num_free, initial_free, exact=True),
                Comparison(f'batch{batch_size}/logical_conditioning_released',
                           bool(set(rids) & shell._eos_embed_sent), False, exact=True),
            ))
        for observer in observations.values():
            observer.close()
        from .qwen_speech import continuations
        checks['talker_decode'].extend(continuations(self, name, acceleration, manager, static, details))
        return checks


def adapters(model, **kwargs):
    from .qwen3_omni_nodes import NodeCaseAdapter
    from mstar.model.qwen3_omni.thinker_logical import ThinkerLogical
    from mstar.model.qwen3_omni.audio_logical import AudioEncoderLogical
    return {TalkerSubmodule: TalkerCaseAdapter(model, **kwargs),
            ThinkerLogical: NodeCaseAdapter(model, node="Thinker", **kwargs),
            AudioEncoderLogical: NodeCaseAdapter(model, node="audio_encoder", **kwargs)}
