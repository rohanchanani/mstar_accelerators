"""Real-declaration cases for Whisper's logical encoder/decoder references.

This adapter interprets logical inputs and state for the differential consumer.
It owns no serving policy and is not selected by a deployment. Each registration
gets a fresh independent case: decode starts from a CPU-reference prefill. The
same adapter also samples a compatible registered prefill before decode, using
one runtime state owner. A declaration covering both walks can supply its own
prefill when the corpus has no distinct compatible producer.
"""
from __future__ import annotations

import gc
import importlib.util
import hashlib
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
from mstar.engine.execution import InvocationFacts, LogicalInvocation, Support
from mstar.engine.resources.binding import merge_resource_requests, resolve_bound_specs
from mstar.engine.resources.sampler.plan import SamplingResult
from mstar.engine.resources.step import StepContext
from mstar.engine.resources.spec import ResourceViewConflict
from mstar.model.binding import NodeBinding
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, NodeInputs
from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE, POS, SAMPLER
from mstar.model.whisper.submodules import (
    WhisperDecoderSubmodule,
    WhisperEncoderSubmodule,
)

from mstar_accelerators.binding import invocation_facts, static_facts
from mstar_accelerators.conditions import Constant
from mstar_accelerators.registry import Registry
from mstar_accelerators.testing.harness import CasePlan, Comparison, Unavailable
from .whisper_reference import WhisperReference
from ..graph_observations import GraphSamplingObservation
from ..generation import GreedyOutput
from mstar_accelerators.testing.runtime import load_engine, join_placements, execute as engine_execute
from mstar_accelerators.testing.reference_resources.snapshot import (
    ReferencePlanConsumer,
)
from mstar_accelerators.testing.sampling import (
    ALPHA, DRAW_SEEDS, CategoricalDistribution, CategoricalSupport,
    policy_domains, request_config,
)
from mstar_accelerators.testing.whisper import (
    _observe,
    _resource_specs,
)


@dataclass(frozen=True)
class _InvocationShape:
    """Realizable cases, separate from the acceleration's serving conditions."""

    prefill_tokens: int

    def evaluate(self, facts):
        if any(facts.get(key) is None for key in ("graph_walk", "batch_size", "token_lengths")):
            return Support.UNKNOWN
        expected = {"prefill": (self.prefill_tokens,), "decode": (1,)}.get(facts["graph_walk"])
        valid = facts["batch_size"] == 1 and facts["token_lengths"] == expected
        if facts.get("total_tokens") is not None:
            valid = valid and facts["total_tokens"] == sum(facts["token_lengths"])
        return Support.YES if valid else Support.NO

    def specialize(self, facts):
        result = self.evaluate(facts)
        return self if result is Support.UNKNOWN else Constant(result)


class WhisperCaseAdapter:
    """One model/input environment shared by independent per-registration cases.

    ``session`` belongs to the existing managed worker. A normally observed case
    settles and closes its roots before the next consumer-selected witness.
    Exceptions retain those roots and propagate to the worker's drain/fatal
    boundary; a Python exception is not completion.
    ``scope`` must describe actual execution, e.g. native Pi or host algebra.
    """

    def __init__(self, model, features, *, session, device, stream=None, corpus=None,
                 language="en", task="transcribe", scope=None, parallel_groups=None, model_config=None, quality=None):
        self.model, self.features, self.session = model, features, session
        self.model_config = dict(model_config or {})
        self.quality = quality
        self.device, self.stream = torch.device(device), stream
        self.language, self.task = language, task
        self.corpus = dict(corpus or {})
        self.scope = scope or f"registered implementation execution on {self.device.type}"
        self.parallel_groups = parallel_groups
        self.agreement = None

    @torch.no_grad()
    def __call__(self, name, acceleration):
        encoder = acceleration.reference is WhisperEncoderSubmodule
        if not encoder and acceleration.reference is not WhisperDecoderSubmodule:
            raise Unavailable("reference_adapter", "this adapter handles Whisper's two logical references")
        node_name = "audio_encoder" if encoder else "decoder"
        if (self.parallel_groups is not None
                and node_name not in self.parallel_groups.node_to_tp_group):
            raise Unavailable("role_not_participating", "this role does not host the logical node")
        shell = self.model.create_submodule(node_name)
        specs = _resource_specs(self.model)
        group = SimpleNamespace(world_size=1, rank=0)
        tp_group = sp_group = group
        self.agreement = None
        if self.parallel_groups is not None:
            from mstar.distributed.execution import ExecutionAgreement
            tp_group = self.parallel_groups.get_tp_config_for_node(node_name)
            sp_group = self.parallel_groups.get_sp_config_for_node(node_name)
            self.agreement = ExecutionAgreement(
                self.parallel_groups.get_joint_group_for_node(node_name), node_name)
        from ..policy import dtype_for
        binding = NodeBinding(self.model, shell, node_name, self.device, dtype_for(self.model, self.model_config),
                              tp_group, sp_group, self.model_config, specs)
        prompt = tuple(self.model.config.decoder_prompt_ids(self.language, self.task))
        domains = {"batch_size": (1,), "graph_walk": ("prefill",) if encoder else ("prefill", "decode")}
        if not encoder:
            domains["token_lengths"] = ((len(prompt),), (1,))
            domains["resources"] = policy_domains(acceleration.when, SAMPLER)
        facts = static_facts(binding)
        if self.quality is not None:
            from .whisper_quality import supported_walks, run_quality
            quality_walks = supported_walks(acceleration, binding, self.quality)
        reference, states = None, None
        # Resolve and run the CPU reference before allocating native state or
        # preparing a physical program. A statically rejected binding needs no
        # reference construction: the common solver will diagnose that skip.
        if acceleration.when.evaluate(facts) is not Support.NO:
            if encoder:
                states = self._encode(binding, torch.float32)

            else:
                reference = self._reference(binding)
                encoder_shell = self.model.create_submodule("audio_encoder")
                encoder_binding = replace(binding, submodule=encoder_shell,
                                          node_name="audio_encoder", prepared={})
                states = self._encode(encoder_binding, torch.float32)
            # Initial CPU reference execution has finished; states own a clone.
            # Do not carry its encoder/engine allocation into candidate setup.
            self.session.close()
            gc.collect()
        trajectories = {"policy": "one-compatible-prefill-per-decode-witness-v2",
                        "selection": "first distinct compatible registration by sorted name, then the decode registration's own prefill",
                        "scope": "bounded registered trajectories, not all pairs",
                        "cases": []}

        def settle():
            if self.stream is not None:
                self.stream.synchronize()
            self.graph_observation.values.clear()
            self.session.close()

        quality_done = False

        def execute(witness):
            nonlocal quality_done
            self.graph_observation = GraphSamplingObservation()
            with self.graph_observation.install():
                comparisons = self._execute(acceleration, binding, witness, prompt, reference, states)
            # All observations now own CPU values. This is test-time settlement,
            # never a serving wait or an exception-path storage release.
            settle()
            if not encoder and witness["graph_walk"] == "decode":
                record = {"decode": name, "considered": [], "executed": False}
                trajectories["cases"].append(record)
                companion = self._prefill_companion(name, acceleration, binding, witness, prompt, record)
                if companion is not None:
                    record["prefill"] = companion[0]
                    with self.graph_observation.install():
                        comparisons.extend(self._execute(
                        acceleration, binding, witness, prompt, reference, states,
                        companion=companion, trajectory=record))
                    settle()
                    record["executed"] = True
            if self.quality is not None and not quality_done:
                with self.graph_observation.install():
                    comparisons.append(run_quality(self, acceleration, binding, self.quality, quality_walks))
                settle()
                quality_done = True
            return comparisons

        return CasePlan(
            facts=facts, domains=domains,
            execute=execute,
            reference=("model-owned HF encoder through M*, CPU FP32; lazy FP64 on numerical failure" if encoder else
                       "model-owned M* decoder, CPU FP32, exact invocation plans and independent physical state"),
            scope=self.scope,
            details={"case": "independent logical state per consumer-selected witness", "state_seed": "none" if encoder else "CPU reference",
                     "preparation": "exact registered implementation.prepare; no diagnostic replacement",
                     "internal_logits": "actual sampler operands; pinned PyTorch comparator with lazy FP64 reference",
                     "sampling": {"policy": "categorical-cdf-dkw-v1",
                                  "draws": len(DRAW_SEEDS), "alpha": ALPHA,
                                  "domain": "canonical greedy or positive temperature; top_k=0, top_p=1, min_p=0, repetition_penalty=1",
                                  "inputs": "bounded policy witnesses from when; exact seeded replay; identical logical state for distribution draws",
                                  "state": "full independent and paired numerical state checks; draw trials retain tokens and exact position/sampler effects",
                                  "independent_decode_history": "CPU-reference prefill with argmax teacher-forcing token; this is case construction, not a stochastic draw oracle",
                                  "trajectory": "actual implementation samples teacher-force the independent reference",
                                  "scope": "bounded seeded empirical CDF; no cross-implementation identical-draw promise or arbitrary-input guarantee"},
                     "trajectories": trajectories, "quality": self.quality},
            constraints=None if encoder else _InvocationShape(len(prompt)),
        )

    def _observe(self, values):
        if self.device.type == "cpu":
            return {key: tensor.detach().clone() for key, tensor in values.items()}
        return _observe(values, self.stream)

    def _admit(self, runner, step):
        if self.agreement is None:
            return runner.admit(step)
        return self.agreement.admit(
            runner, step, request_ids=tuple(f"qualification:{rid}" for rid in step.ctx.request_ids))

    def _executing(self):
        return nullcontext() if self.agreement is None else self.agreement.executing()

    def _complete(self):
        if self.agreement is not None:
            self.agreement.complete()

    def _replica_checks(self, observed):
        if self.agreement is None or self.agreement.group.world_size == 1:
            return []
        identities = {}
        for name, tensor in observed.items():
            value = tensor.detach().cpu().contiguous()
            identities[name] = (str(value.dtype), tuple(value.shape),
                                hashlib.sha256(value.numpy().tobytes()).hexdigest())
        peers = self.agreement.group.all_gather_object(identities)
        return [Comparison("replicated_state_and_outputs", peers,
                           [peers[0]] * len(peers), exact=True)]

    def _encode(self, binding, dtype):
        reference = self._reference(binding, dtype=dtype)
        execution = WhisperReference(replace(binding, dtype=dtype), reference, self.session)
        _, result = execution.invoke('prefill', 0, features=self.features)
        execution.remove(0)
        return result.encoder_states[0].detach().cpu().clone()

    def _reference(self, binding, *, dtype=torch.float32):
        shell = self.model.create_submodule(binding.node_name)
        factory = getattr(shell, "make_reference_program", None)
        if not callable(factory):
            raise Unavailable("missing_reference", f"{type(shell).__name__} has no executable reference factory")
        if importlib.util.find_spec("transformers") is None:
            raise Unavailable("reference_dependency", "the Whisper CPU reference requires transformers")
        from mstar.distributed.communication import CommGroup
        reference_binding = replace(binding, submodule=shell, device=torch.device("cpu"),
                                    dtype=dtype, prepared={}, tp_group=CommGroup.trivial(),
                                    sp_group=CommGroup.trivial())
        try:
            return factory(reference_binding)
        except NotImplementedError as exc:
            raise Unavailable("unavailable_reference", str(exc) or "reference factory is not implemented") from exc

    def _check_when(self, acceleration, binding, witness, walk, token_lengths=None, *, policies=None):
        policies = policies if policies is not None else witness.get("resources", {})
        actual = {**static_facts(binding),
                  **invocation_facts(InvocationFacts(walk, 1, token_lengths, policies,
                                                    row_walks=(walk,)))}
        for key in ("graph_walk", "batch_size", "token_lengths", "total_tokens", "resources"):
            if key in witness and witness[key] != actual[key]:
                raise RuntimeError(f"the materialized case differs from its solved {key} witness")
        if acceleration.when.evaluate(actual) is not Support.YES:
            raise RuntimeError("the materialized logical case does not satisfy the registered when")

    def _prefill_companion(self, name, acceleration, binding, witness, prompt, record):
        """Sample an actual same-reference transition without allocating state.

        The model adapter knows which logical history constructs decoder state.
        The corpus supplies possible physical producers. Production assembly and
        resource resolution alone determine their representation compatibility.
        A malformed configuration or arbitrary preparation error is not a skip.
        """
        policies = witness["resources"]
        facts = {**static_facts(binding), **invocation_facts(InvocationFacts(
            "prefill", 1, (len(prompt),), policies, row_walks=("prefill",)))}
        decoder = Registry({name: acceleration}).assemble(
            replace(binding, graph_walks=("decode",)), require_coverage=False)
        candidates = [(other_name, other) for other_name, other in sorted(self.corpus.items())
                      if other_name != name]
        candidates.append((name, acceleration))
        for other_name, other in candidates:
            if other.reference is not acceleration.reference:
                continue
            support = other.when.evaluate(facts)
            candidate = {"registration": other_name, "support": support.name}
            record["considered"].append(candidate)
            if support is not Support.YES:
                continue
            prefill = Registry({other_name: other}).assemble(
                replace(binding, graph_walks=("prefill",)), require_coverage=False)
            requests = merge_resource_requests(prefill.resource_requests, decoder.resource_requests)
            try:
                resolve_bound_specs(binding.resource_specs, requests, dtype=binding.dtype,
                                    group_sizes={spec.resource_key: binding.tp_group.world_size
                                                 * binding.sp_group.world_size
                                                 for spec in binding.resource_specs})
            except ResourceViewConflict as exc:
                candidate["compatible"] = False
                candidate["reason"] = str(exc)
                continue
            candidate["compatible"] = True
            record["selection_kind"] = ("same_registration" if other_name == name else
                                        "distinct_registration")
            record["resource_requests"] = {
                key: [repr(view) for view in views] for key, views in requests.items()
            }
            return other_name, other
        record["reason"] = "no supported, representation-compatible prefill in this corpus or declaration"
        return None

    @torch.no_grad()
    def _execute(self, acceleration, binding, witness, prompt, reference, states, *,
                 companion=None, trajectory=None):
        walk = witness["graph_walk"]
        encoder = isinstance(binding.submodule, WhisperEncoderSubmodule)
        config = None if encoder else request_config(witness, SAMPLER)
        lengths = None if encoder else ((len(prompt),) if walk == "prefill" else (1,))
        self._check_when(acceleration, binding, witness, walk, lengths)
        # Qualification specializes the actual registration to this witness,
        # using the very same assembly stages as a worker placement.
        shared_program = companion is not None and companion[1] is acceleration
        binding = replace(binding, graph_walks=("prefill", "decode") if shared_program else (walk,), prepared={})
        self.session.bindings.append(binding)
        assembly = Registry({"candidate": acceleration}).assemble(binding, require_coverage=False)
        if encoder:
            engine = load_engine(binding, assembly, self.session, self.parallel_groups)
            engine.add_request(0)
            batch, observations = engine_execute(engine, binding.node_name, walk, 0,
                                                   {"audio_features": [self.features]})
            if batch.admit_error is not None:
                raise RuntimeError(f"encoder admission failed: {batch.admit_error}")
            result, _, _ = observations[0]
            self.session.retained.extend(result.encoder_states)
            actual = self._observe({"encoder": result.encoder_states[0]})["encoder"]
            engine.remove_request(0)
            return [Comparison("encoder.rows", len(result.encoder_states), 1, exact=True),
                    Comparison("encoder", actual, states, dtype=torch.float32, promoted=lambda: self._encode(binding, torch.float64))]

        placements = [assembly]
        if companion is not None and not shared_program:
            placements.insert(0, Registry({companion[0]: companion[1]}).assemble(
                replace(binding, graph_walks=("prefill",)), require_coverage=False))
        engine = load_engine(binding, join_placements(*placements), self.session, self.parallel_groups)
        resources = engine._resources
        runner = engine._runner
        reference_state = ReferencePlanConsumer(resources)
        specs = tuple(replace(spec, config=resources[spec.resource_key].config)
                      if spec.resource_key in (KV_CACHE, CROSS_KV_CACHE) else spec
                      for spec in binding.resource_specs)
        ordinary = WhisperReference(binding, reference, self.session, specs=specs)
        history, higher_values = [], None
        def remove(rid):
            engine.remove_request(rid)
            if rid in ordinary.active:
                history.append(('remove', rid))
            ordinary.remove(rid)

        def higher(index, key):
            nonlocal higher_values
            if higher_values is None:
                precise = WhisperReference(replace(binding, dtype=torch.float64),
                    self._reference(binding, dtype=torch.float64), self.session, specs=specs)
                higher_values = {}
                for i, event in enumerate(history):
                    if event[0] == 'remove':
                        precise.remove(event[1])
                        continue
                    _, rid, selected_walk, ids, encoder_states = event
                    batch, _ = precise.invoke(selected_walk, rid, ids, encoder_states)
                    plans = batch.step_context.plan_results
                    higher_values[i] = {'logits': precise.logits.detach().cpu().clone(),
                        **{key: precise.resources[key].kv_cache.tensor[:, list(live_pages(plans, key, rid))].detach().clone()
                           for key in (KV_CACHE, CROSS_KV_CACHE)}}
                # Comparison runs after candidate settlement. The promoted
                # replay now owns copied observations, not live runtime state.
                self.session.close()
            return higher_values[index][key]
        if companion is not None:
            trajectory["resolved_resources"] = {
                key: {"layout": "contiguous" if getattr(resources[key].config, "dense", False) else "paged",
                      "block_tokens": resources[key].config.page_size,
                      "blocks": resources[key].config.max_num_pages,
                      "partition": getattr(resources[key].config, "partition", "legacy")}
                for key in (KV_CACHE, CROSS_KV_CACHE)
            }

        def declare(rid, selected_walk, ids, encoder_states=None):
            logical = ARNodeInputs(input_ids=ids, input_seq_len=ids.numel(),
                                   tensor_inputs={} if encoder_states is None else {"encoder_states": encoder_states})
            step = binding.submodule.declare_step(selected_walk, [rid], [logical])
            step.set_ctx(StepContext([rid], selected_walk, slot=0, capture=False))
            return step, logical

        def plan(rid, selected_walk, ids, encoder_states=None):
            # Only initial CPU-oracle fixture construction uses this helper.
            # Every candidate invocation is executed by Engine.exec below.
            step, logical = declare(rid, selected_walk, ids, encoder_states)
            if not self._admit(runner, step).ready:
                raise RuntimeError("the generated single-request case failed resource admission")
            with self._executing():
                plans = runner.plan(step)
                self._complete()
            native_engine = ModelInputsFromEngine([rid], {}, resources, step=step)
            reference_engine = ModelInputsFromEngine([rid], {}, reference_state.resources, step=step)
            invocation = LogicalInvocation(selected_walk, [rid], [logical], native_engine, plans,
                                           row_walks=(selected_walk,))
            return step, invocation, reference_engine

        def live_pages(plans, key, rid):
            return tuple(dict.fromkeys(page for use in plans[key].values()
                                       for view in use.views if view.request_id == rid
                                       for page in view.page_idxs))

        reference_cases = {}
        seed_holders = []
        self.session.retained.append(seed_holders)

        def reference_logits(rid, selected_walk, unused_engine, plans, ids, encoder_states, *, reuse=False):
            if reuse and selected_walk in reference_cases:
                for key in (KV_CACHE, CROSS_KV_CACHE):
                    tensor = reference_state.resources[key].kv_cache.tensor
                    tensor.zero_()
                    tensor[:, list(live_pages(plans, key, rid))] = reference_cases[selected_walk][1][key].to(tensor.dtype)
                return reference_cases[selected_walk][0]
            ids = self._observe({'ids': ids})['ids'] if ids.device.type != 'cpu' else ids.detach().clone()
            index = len(history)
            history.append(('step', rid, selected_walk, ids, encoder_states))
            batch, _ = ordinary.invoke(selected_walk, rid, ids, encoder_states)
            ref_plans = batch.step_context.plan_results
            saved = {}
            for key in (KV_CACHE, CROSS_KV_CACHE):
                source_pages = list(live_pages(ref_plans, key, rid))
                target_pages = list(live_pages(plans, key, rid))
                saved[key] = ordinary.resources[key].kv_cache.tensor[:, source_pages].detach().clone()
                # This snapshot represents the candidate's declared physical
                # storage encoding; independent reference computation stays FP32.
                snapshot = reference_state.resources[key].kv_cache.tensor
                snapshot[:, target_pages] = saved[key].to(snapshot.dtype)
            reference_cases[selected_walk] = (ordinary.logits.detach().cpu().clone(), saved, index)
            return reference_cases[selected_walk][0]

        def run_step(rid, selected_walk, ids, declaration, *, prefix="",
                     reuse_reference=False, sample_only=False):
            nonlocal actual_distribution_logits
            encoder_states = states if selected_walk == "prefill" else None
            self._check_when(declaration, binding, {}, selected_walk, (ids.numel(),),
                             policies={SAMPLER: resources[SAMPLER].policy_facts([rid])})
            before_position = resources[POS].position(rid, "main")
            before_sampling = resources[SAMPLER]._sampler._step_offset[rid]
            # Immutability compares implementation state with its own snapshot;
            # accuracy against independent reference state remains numerical.
            cross_before = None
            self_before = None
            if selected_walk == "decode" and not sample_only:
                prior = self._observe({
                    CROSS_KV_CACHE: resources[CROSS_KV_CACHE].kv_cache.tensor,
                    KV_CACHE: resources[KV_CACHE].kv_cache.tensor,
                })
                cross_before, self_before = prior[CROSS_KV_CACHE], prior[KV_CACHE]
            def observe(result, invocation):
                return (resources[POS].position(rid, "main"),
                        resources[SAMPLER]._sampler._step_offset[rid])

            raw = {"text_inputs": [ids]}
            if encoder_states is not None:
                raw["encoder_states"] = [encoder_states]
            with self.graph_observation.capture_torch(self.model.config.vocab_size):
                batch, observations = engine_execute(engine, binding.node_name, selected_walk, rid,
                                                       raw, observe=observe)
            if batch.admit_error is not None:
                raise RuntimeError(f"qualification admission failed: {batch.admit_error}")
            result, invocation, primitive_state = observations[0]
            reference_engine = ModelInputsFromEngine([rid], {}, reference_state.resources,
                                                    step=invocation.engine_inputs.step)
            cloned = result.tokens.clone()
            self.session.retained.append(result.tokens)
            expected_logits = (reference_cases[selected_walk][0] if sample_only else reference_logits(
                rid, selected_walk, reference_engine, invocation.plan_results, ids, encoder_states,
                reuse=reuse_reference))
            actual_logits = None
            if self.graph_observation.values:
                actual_logits = self._observe({'logits': self.graph_observation.values[-1]})['logits'].reshape_as(expected_logits)
            # A returned token alone cannot qualify the floating computation.
            if actual_logits is None:
                raise RuntimeError("no observation of the executed primitive's sampling input")
            actual_distribution_logits = actual_logits
            expected_tokens = (actual_logits.argmax(dim=-1) if config.temperature == 0 else
                               CategoricalSupport(actual_logits))
            observed = self._observe({"tokens": result.tokens, "clone": cloned, **({} if sample_only else {
                KV_CACHE: resources[KV_CACHE].kv_cache.tensor,
                CROSS_KV_CACHE: resources[CROSS_KV_CACHE].kv_cache.tensor})})
            # The observation boundary has completed the preceding seed copies.
            # Keep their roots on exceptional paths, but do not accumulate a
            # full physical-cache staging copy for every ordinary draw trial.
            seed_holders.clear()
            checks = [Comparison("tokens", observed["tokens"], expected_tokens, exact=True),
                      Comparison("immediate_clone", observed["clone"],
                                 expected_tokens if config.temperature == 0 else observed["tokens"], exact=True),
                      Comparison("seed.position", before_position,
                                 len(prompt) if selected_walk == "decode" else 0, exact=True),
                      Comparison("seed.sampling", before_sampling,
                                 1 if selected_walk == "decode" else 0, exact=True),
                      Comparison("primitive.position_unchanged", primitive_state[0],
                                 before_position, exact=True),
                      Comparison("primitive.sampling_unchanged", primitive_state[1],
                                 before_sampling, exact=True)]
            mask = torch.zeros_like(expected_logits, dtype=torch.bool)
            mask[:, self.model.config.suppressed_tokens(selected_walk)] = True
            checks.append(Comparison("logits", actual_logits, expected_logits,
                                     promoted=lambda index=reference_cases[selected_walk][2]: higher(index, 'logits'), suppression_mask=mask, internal=True))
            checks.extend(self._replica_checks(observed))
            pages = {}
            for key in (KV_CACHE, CROSS_KV_CACHE):
                pages[key] = tuple(sorted(live_pages(invocation.plan_results, key, rid)))
                if sample_only:
                    continue
                # Free blocks have no logical value. Comparing the live blocks
                # (including their full tail) catches stale-NaN contamination
                # without treating deliberately poisoned free storage as state.
                columns = list(pages[key])
                checks.append(Comparison(key, observed[key][:, columns],
                                         reference_state.resources[key].kv_cache.tensor[:, columns].clone(),
                                         promoted=lambda index=reference_cases[selected_walk][2], key=key: higher(index, key)))
                sink = observed[key][:, 0]
                checks.append(Comparison(key + ".sink", sink, torch.zeros_like(sink), exact=True))
            if selected_walk == "decode" and not sample_only:
                columns = list(pages[CROSS_KV_CACHE])
                checks.append(Comparison("decode.cross_state_unchanged",
                                         observed[CROSS_KV_CACHE][:, columns],
                                         cross_before[:, columns], exact=True))
            if selected_walk == "decode" and not sample_only:
                # Appending a token must preserve this request's committed prefix,
                # independently of its deliberately approximate storage precision.
                page_size = resources[KV_CACHE].config.page_size
                page_ids, offsets = [], []
                for use in invocation.plan_results[KV_CACHE].values():
                    for view in use.views:
                        if view.request_id != rid:
                            continue
                        for position in range(view.start, view.start + view.length - view.to_compute):
                            page_ids.append(view.page_idxs[position // page_size])
                            offsets.append(position % page_size)
                checks.append(Comparison("decode.self_prefix_unchanged",
                                         observed[KV_CACHE][:, page_ids, :, offsets],
                                         self_before[:, page_ids, :, offsets], exact=True))
            # Engine has already adapted the result and committed the declared effects.
            checks.extend((Comparison("commit.position", resources[POS].position(rid, "main"),
                                      before_position + ids.numel(), exact=True),
                           Comparison("commit.sampling", resources[SAMPLER]._sampler._step_offset[rid],
                                      before_sampling + 1, exact=True)))
            return ([replace(check, name=prefix + check.name) for check in checks],
                    result.tokens, expected_tokens if config.temperature == 0 else observed["tokens"], pages)

        def ingest(rid, *, seed=None):
            engine.add_request(rid, {SAMPLER: replace(
                config, _seed=DRAW_SEEDS[rid] if seed is None else seed)})

        def seed_decode(rid):
            seed_ids = torch.tensor(prompt, dtype=torch.long)
            seed_step, seed_invocation, seed_engine = plan(rid, "prefill", seed_ids, states)
            logits = reference_logits(rid, "prefill", seed_engine, seed_invocation.plan_results,
                                      seed_ids, states, reuse=True)
            seed_ids = logits.argmax(dim=-1)
            # Independent reference-state construction: argmax is a fixed,
            # lawful teacher-forcing token, not an expected stochastic draw.
            for key in (KV_CACHE, CROSS_KV_CACHE):
                seed = reference_state.resources[key].kv_cache.tensor.clone()
                seed_holders.append(seed)
                resources[key].kv_cache.tensor.copy_(seed, non_blocking=True)
            runner.accept_results(seed_step, {SAMPLER: SamplingResult(seed_ids.to(self.device))})
            runner.commit(seed_step)
            return seed_ids

        actual_distribution_logits = None
        ids = torch.tensor(prompt, dtype=torch.long)
        ingest(0)
        if companion is None:
            if walk == "decode":
                ids = seed_decode(0)
            checks, _, first_sample, _ = run_step(
                0, walk, ids, acceleration, reuse_reference=True)
            remove(0)
            if config.temperature > 0:
                samples = [first_sample]
                for index, seed in enumerate((*DRAW_SEEDS[1:], DRAW_SEEDS[0]), 1):
                    for key in (KV_CACHE, CROSS_KV_CACHE):
                        reference_state.resources[key].kv_cache.tensor.zero_()
                    ingest(0, seed=seed)
                    if walk == "decode":
                        ids = seed_decode(0)
                    more, _, sampled, _ = run_step(
                        0, walk, ids, acceleration,
                        prefix=f"sampling.draw{index}.", reuse_reference=True, sample_only=True)
                    checks.extend(more)
                    remove(0)
                    if index < len(DRAW_SEEDS):
                        samples.append(sampled)
                    else:
                        checks.append(Comparison("sampling.seeded_replay", sampled, first_sample, exact=True))
                checks.append(Comparison("sampling.distribution", torch.cat(samples),
                                         CategoricalDistribution(actual_distribution_logits, config.temperature)))
        else:
            def transition(rid, label):
                prefill_checks, output, expected, pages = run_step(
                    rid, "prefill", ids, companion[1], prefix=label + "prefill.")
                decode_checks, decoded, expected_decode, pages = run_step(
                    rid, "decode", output, acceleration, prefix=label + "decode.")
                retained = self._observe({"prefill": output})["prefill"]
                return (prefill_checks + decode_checks + [Comparison(
                    label + "retained_prefill_after_decode", retained, expected, exact=True)],
                    decoded, expected_decode, pages)

            checks, retained, expected_retained, first_pages = transition(0, "trajectory.")
            # The supplied bounded environment has two dense slots. Exercise
            # capacity/reassignment through those same programs when resolved
            # storage has that shape; other representations retain the sampled
            # transition and report that this lifecycle case was not applicable.
            two_slots = all(getattr(resources[key].config, "dense", False)
                            and resources[key].config.max_num_pages == 3
                            for key in (KV_CACHE, CROSS_KV_CACHE))
            trajectory["lifecycle"] = {"policy": "two-live-backpressure-reassignment-v1",
                                       "applicable": two_slots}
            if two_slots:
                ingest(1)
                more, _, _, second_pages = transition(1, "trajectory.second.")
                checks.extend(more)
                survivor_reference_index = reference_cases["decode"][2]
                survivor_before = self._observe({
                    key: resources[key].kv_cache.tensor for key in second_pages})
                survivor_position = resources[POS].position(1, "main")
                survivor_sampling = resources[SAMPLER]._sampler._step_offset[1]
                ingest(2)
                blocked, observations = engine_execute(
                    engine, binding.node_name, "prefill", 2,
                    {"text_inputs": [ids], "encoder_states": [states]})
                checks.append(Comparison("trajectory.third_request_backpressure",
                                         blocked.admit_error is None, False, exact=True))
                checks.append(Comparison("trajectory.backpressure_is_retryable",
                                         bool(blocked.admit_error and not blocked.admit_error.terminal), True, exact=True))
                if observations:
                    raise RuntimeError("backpressured request unexpectedly executed")
                # No plan/submission is performed by this failed proposal.
                remove(2)
                remove(0)
                for key, pages in first_pages.items():
                    for layer in resources[key].kv_cache.tensor:
                        for page in pages:
                            layer[page].fill_(float("nan"))
                    # The oracle follows the new logical request's initial
                    # state; only the runtime chooses its new page identity.
                    reference_state.resources[key].kv_cache.tensor[:, list(pages)] = 0
                ingest(2)
                more, _, _, reassigned = transition(2, "trajectory.reassigned.")
                checks.extend(more)
                checks.append(Comparison("trajectory.reuses_released_blocks", reassigned, first_pages, exact=True))
                observed = self._observe({"retained": retained,
                                          KV_CACHE: resources[KV_CACHE].kv_cache.tensor,
                                          CROSS_KV_CACHE: resources[CROSS_KV_CACHE].kv_cache.tensor})
                checks.append(Comparison("trajectory.retained_output_after_reassignment",
                                         observed["retained"], expected_retained, exact=True))
                for key, pages in second_pages.items():
                    checks.append(Comparison("trajectory.other_request_preserved." + key,
                                             observed[key][:, list(pages)],
                                             survivor_before[key][:, list(pages)], exact=True))
                    checks.append(Comparison("trajectory.other_request_accuracy." + key,
                                             observed[key][:, list(pages)],
                                             reference_state.resources[key].kv_cache.tensor[:, list(pages)],
                                             promoted=lambda index=survivor_reference_index, key=key: higher(index, key)))
                checks.extend((
                    Comparison("trajectory.other_request_preserved.position",
                               resources[POS].position(1, "main"), survivor_position, exact=True),
                    Comparison("trajectory.other_request_preserved.sampling",
                               resources[SAMPLER]._sampler._step_offset[1], survivor_sampling, exact=True),
                ))
                remove(1)
                remove(2)
            else:
                remove(0)
        for key in (KV_CACHE, CROSS_KV_CACHE):
            checks.append(Comparison(("trajectory." if companion else "") + key + ".cleanup_pages",
                                     resources[key]._arena.num_free,
                                     resources[key].config.max_num_pages - 1, exact=True))
        return checks


def adapters(model, features, **kwargs):
    adapter = WhisperCaseAdapter(model, features, **kwargs)
    return {WhisperEncoderSubmodule: adapter, WhisperDecoderSubmodule: adapter}
