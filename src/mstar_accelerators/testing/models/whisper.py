"""Real-declaration cases for Whisper's logical encoder/decoder references.

This adapter interprets logical inputs and state for the differential consumer.
It owns no serving policy and is not selected by a deployment. Each registration
gets a fresh independent case: decode starts from a CPU-reference prefill. The
same adapter also samples a compatible registered prefill before decode, using
one runtime state owner. A declaration covering both walks can supply its own
prefill when the corpus has no distinct compatible producer.
"""
from __future__ import annotations

import importlib.util
import hashlib
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
from mstar.engine.execution import InvocationFacts, LogicalInvocation, Support
from mstar.engine.resources.binding import merge_resource_requests, resolve_bound_specs
from mstar.engine.resources.runner import StepRunner
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
    _run,
    _runtime_resources,
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
                 language="en", task="transcribe", scope=None, parallel_groups=None):
        self.model, self.features, self.session = model, features, session
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
        binding = NodeBinding(self.model, shell, node_name, self.device, torch.float32,
                              tp_group, sp_group, {}, specs)
        prompt = tuple(self.model.config.decoder_prompt_ids(self.language, self.task))
        domains = {"batch_size": (1,), "graph_walk": ("prefill",) if encoder else ("prefill", "decode")}
        if not encoder:
            domains["token_lengths"] = ((len(prompt),), (1,))
            domains["resources"] = policy_domains(acceleration.when, SAMPLER)
        facts = static_facts(binding)
        reference, states = None, None
        # Resolve and run the CPU reference before allocating native state or
        # preparing a physical program. A statically rejected binding needs no
        # reference construction: the common solver will diagnose that skip.
        if acceleration.when.evaluate(facts) is not Support.NO:
            if encoder:
                states = self._reference(binding, dtype=torch.float64).run(self.features)
            else:
                reference = self._reference(binding)
                encoder_shell = self.model.create_submodule("audio_encoder")
                encoder_binding = replace(binding, submodule=encoder_shell,
                                          node_name="audio_encoder", prepared={})
                states = self._reference(encoder_binding, dtype=torch.float64).run(self.features).float()
        trajectories = {"policy": "one-compatible-prefill-per-decode-witness-v2",
                        "selection": "first distinct compatible registration by sorted name, then the decode registration's own prefill",
                        "scope": "bounded registered trajectories, not all pairs",
                        "cases": []}

        def settle():
            if self.stream is not None:
                self.stream.synchronize()
            self.session.close()

        def execute(witness):
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
                    comparisons.extend(self._execute(
                        acceleration, binding, witness, prompt, reference, states,
                        companion=companion, trajectory=record))
                    settle()
                    record["executed"] = True
            return comparisons

        return CasePlan(
            facts=facts, domains=domains,
            execute=execute,
            reference=("model-owned HF encoder, CPU FP64 on logical FP32 values" if encoder else
                       "model-owned M* decoder, CPU FP32, exact invocation plans and independent physical state"),
            scope=self.scope,
            details={"case": "independent logical state per consumer-selected witness", "state_seed": "none" if encoder else "CPU reference",
                     "preparation": "exact registered implementation.prepare; no diagnostic replacement",
                     "internal_logits": "not exposed or compared by this consumer",
                     "sampling": {"policy": "categorical-cdf-dkw-v1",
                                  "draws": len(DRAW_SEEDS), "alpha": ALPHA,
                                  "domain": "canonical greedy or positive temperature; top_k=0, top_p=1, min_p=0, repetition_penalty=1",
                                  "inputs": "bounded policy witnesses from when; exact seeded replay; identical logical state for distribution draws",
                                  "state": "full independent and paired numerical state checks; draw trials retain tokens and exact position/sampler effects",
                                  "independent_decode_history": "CPU-reference prefill with argmax teacher-forcing token; this is case construction, not a stochastic draw oracle",
                                  "trajectory": "actual implementation samples teacher-force the independent reference",
                                  "scope": "bounded seeded empirical CDF; no cross-implementation identical-draw promise or arbitrary-input guarantee"},
                     "trajectories": trajectories},
            constraints=None if encoder else _InvocationShape(len(prompt)),
        )

    def _observe(self, values):
        if self.device.type == "cpu":
            return {key: tensor.detach().clone() for key, tensor in values.items()}
        return _observe(values, self.stream)

    def _reserve(self, capabilities, facts, request_ids):
        if self.agreement is None:
            return capabilities.reserve(facts)
        return self.agreement.reserve(capabilities, facts,
                                      tuple(f"qualification:{rid}" for rid in request_ids))

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

    def _bind(self, assembly, resources):
        with self._executing():
            capabilities = assembly.bind(resources)
            # Retain before the completion rendezvous can report a peer failure.
            self.session.prepared.append(capabilities)
            self._complete()
        return capabilities

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
            if self.agreement is not None:
                self.agreement.agree(assembly.execution_names, "qualification implementation set")
            capabilities = self._bind(assembly, {})
            logical = NodeInputs(tensor_inputs={"audio_features": self.features})
            invocation = LogicalInvocation(walk, [0], [logical], ModelInputsFromEngine([0], {}),
                                           row_walks=(walk,))
            selected = self._reserve(capabilities, InvocationFacts(walk, 1, row_walks=(walk,)), (0,))
            if selected is None:
                raise RuntimeError("the fresh qualification program has no available workspace")
            try:
                with self._executing():
                    result, _ = _run(selected, invocation, self.stream)
                    self._complete()
            finally:
                selected.close()
            self.session.retained.extend(result.encoder_states)
            actual = self._observe({"encoder": result.encoder_states[0]})["encoder"]
            return [Comparison("encoder.rows", len(result.encoder_states), 1, exact=True),
                    Comparison("encoder", actual, states, dtype=torch.float32)]

        prefix_assembly = None
        requests = assembly.resource_requests
        if companion is not None:
            prefix_assembly = assembly if shared_program else Registry({companion[0]: companion[1]}).assemble(
                replace(binding, graph_walks=("prefill",)), require_coverage=False)
            requests = merge_resource_requests(prefix_assembly.resource_requests, requests)
        if self.agreement is not None:
            self.agreement.agree((assembly.execution_names,
                                  None if prefix_assembly is None else prefix_assembly.execution_names),
                                 "qualification implementation set")
        group = (None if self.parallel_groups is None else
                 self.parallel_groups.get_joint_group_for_node(binding.node_name))
        _, resources = _runtime_resources(self.model, self.device, requests,
                                          specs=binding.resource_specs, session=self.session,
                                          group=group, agreement=self.agreement)
        runner = StepRunner(resources)
        reference_state = ReferencePlanConsumer(resources)
        reference_shell = self.model.create_submodule("decoder")
        reference_shell.reference_program = reference
        reference_shell.bind_node_resources(reference_state.resources)
        capabilities = self._bind(assembly, resources)
        prefix_capabilities = None
        if prefix_assembly is not None:
            prefix_capabilities = capabilities if shared_program else self._bind(prefix_assembly, resources)
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
            step, logical = declare(rid, selected_walk, ids, encoder_states)
            if not self._admit(runner, step).ready:
                raise RuntimeError("the generated single-request case failed resource admission")
            with self._executing():
                plans = runner.plan(step)
                self._complete()
            reference_state.use(plans)
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

        def reference_logits(rid, selected_walk, engine, plans, ids, encoder_states, *, reuse=False):
            # Distribution trials repeat the same logical inputs and state.
            # Cache reference math once, relocating its physical copy through
            # each fresh runtime plan rather than assuming stable page numbers.
            if reuse and selected_walk in reference_cases:
                logits, saved = reference_cases[selected_walk]
                for key in (KV_CACHE, CROSS_KV_CACHE):
                    tensor = reference_state.resources[key].kv_cache.tensor
                    tensor.zero_()
                    tensor[:, list(live_pages(plans, key, rid))] = saved[key]
                return logits
            logits = reference.run(selected_walk, engine, ids, encoder_states, return_logits=True)
            if reuse:
                reference_cases[selected_walk] = (logits, {
                    key: reference_state.resources[key].kv_cache.tensor[
                        :, list(live_pages(plans, key, rid))].clone()
                    for key in (KV_CACHE, CROSS_KV_CACHE)
                })
            return logits

        def run_step(rid, selected_walk, ids, selected_capabilities, declaration, *, prefix="",
                     reuse_reference=False, sample_only=False):
            encoder_states = states if selected_walk == "prefill" else None
            self._check_when(declaration, binding, {}, selected_walk, (ids.numel(),),
                             policies={SAMPLER: resources[SAMPLER].policy_facts([rid])})
            before_position = resources[POS].position(rid, "main")
            before_sampling = resources[SAMPLER]._sampler._step_offset[rid]
            # Immutability compares implementation state with its own snapshot;
            # accuracy against independent reference state remains numerical.
            cross_before = None
            if selected_walk == "decode" and not sample_only:
                cross_before = self._observe({
                    CROSS_KV_CACHE: resources[CROSS_KV_CACHE].kv_cache.tensor,
                })[CROSS_KV_CACHE]
            facts = InvocationFacts(selected_walk, 1, (ids.numel(),),
                                    {SAMPLER: resources[SAMPLER].policy_facts([rid])},
                                    row_walks=(selected_walk,))
            # As in serving, workspace acquisition precedes admission and the
            # ordered initialization/materialization that can submit work.
            selected = self._reserve(selected_capabilities, facts, (rid,))
            if selected is None:
                raise RuntimeError("the fresh qualification program has no available workspace")
            try:
                step, invocation, reference_engine = plan(rid, selected_walk, ids, encoder_states)
                with self._executing():
                    result, _ = _run(selected, invocation, self.stream)
                    self._complete()
            finally:
                selected.close()
            cloned = result.tokens.clone()
            self.session.retained.append(result.tokens)
            expected_logits = (reference_cases[selected_walk][0] if sample_only else reference_logits(
                rid, selected_walk, reference_engine, invocation.plan_results, ids, encoder_states,
                reuse=reuse_reference))
            expected_tokens = (expected_logits.argmax(dim=-1) if config.temperature == 0 else
                               CategoricalSupport(expected_logits))
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
                      Comparison("primitive.position_unchanged", resources[POS].position(rid, "main"),
                                 before_position, exact=True),
                      Comparison("primitive.sampling_unchanged", resources[SAMPLER]._sampler._step_offset[rid],
                                 before_sampling, exact=True)]
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
                                         reference_state.resources[key].kv_cache.tensor[:, columns]))
                sink = observed[key][:, 0]
                checks.append(Comparison(key + ".sink", sink, torch.zeros_like(sink), exact=True))
            if selected_walk == "decode" and not sample_only:
                columns = list(pages[CROSS_KV_CACHE])
                checks.append(Comparison("decode.cross_state_unchanged",
                                         observed[CROSS_KV_CACHE][:, columns],
                                         cross_before[:, columns], exact=True))
            # Reuse the model's existing result-to-resource mapping, exactly as
            # Engine does, instead of maintaining a second sampler commit path.
            effects = binding.submodule.execution_result(result, invocation).resource_results
            runner.accept_results(step, effects)
            runner.commit(step)
            checks.extend((Comparison("commit.position", resources[POS].position(rid, "main"),
                                      before_position + ids.numel(), exact=True),
                           Comparison("commit.sampling", resources[SAMPLER]._sampler._step_offset[rid],
                                      before_sampling + 1, exact=True)))
            return ([replace(check, name=prefix + check.name) for check in checks],
                    result.tokens, expected_tokens if config.temperature == 0 else observed["tokens"], pages)

        def ingest(rid, *, seed=None):
            runner.ingest_request(rid, {SAMPLER: replace(
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

        ids = torch.tensor(prompt, dtype=torch.long)
        ingest(0)
        if companion is None:
            if walk == "decode":
                ids = seed_decode(0)
            checks, _, first_sample, _ = run_step(
                0, walk, ids, capabilities, acceleration, reuse_reference=True)
            runner.remove_request(0)
            if config.temperature > 0:
                samples = [first_sample]
                for index, seed in enumerate((*DRAW_SEEDS[1:], DRAW_SEEDS[0]), 1):
                    for key in (KV_CACHE, CROSS_KV_CACHE):
                        reference_state.resources[key].kv_cache.tensor.zero_()
                    ingest(0, seed=seed)
                    if walk == "decode":
                        ids = seed_decode(0)
                    more, _, sampled, _ = run_step(
                        0, walk, ids, capabilities, acceleration,
                        prefix=f"sampling.draw{index}.", reuse_reference=True, sample_only=True)
                    checks.extend(more)
                    runner.remove_request(0)
                    if index < len(DRAW_SEEDS):
                        samples.append(sampled)
                    else:
                        checks.append(Comparison("sampling.seeded_replay", sampled, first_sample, exact=True))
                checks.append(Comparison("sampling.distribution", torch.cat(samples),
                                         CategoricalDistribution(reference_cases[walk][0], config.temperature)))
        else:
            def transition(rid, label):
                prefill_checks, output, expected, pages = run_step(
                    rid, "prefill", ids, prefix_capabilities, companion[1], prefix=label + "prefill.")
                decode_checks, decoded, expected_decode, pages = run_step(
                    rid, "decode", output, capabilities, acceleration, prefix=label + "decode.")
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
                survivor_before = self._observe({
                    key: resources[key].kv_cache.tensor for key in second_pages})
                survivor_position = resources[POS].position(1, "main")
                survivor_sampling = resources[SAMPLER]._sampler._step_offset[1]
                ingest(2)
                facts = InvocationFacts("prefill", 1, (len(prompt),),
                                        {SAMPLER: resources[SAMPLER].policy_facts([2])},
                                        row_walks=("prefill",))
                selected = self._reserve(prefix_capabilities, facts, (2,))
                if selected is None:
                    raise RuntimeError("backpressure probe could not reserve idle workspace")
                try:
                    blocked_step, _ = declare(2, "prefill", ids, states)
                    blocked = self._admit(runner, blocked_step)
                finally:
                    selected.close()
                checks.append(Comparison("trajectory.third_request_backpressure", blocked.ok, False, exact=True))
                checks.append(Comparison("trajectory.backpressure_is_retryable",
                                         bool(blocked.reason and not blocked.reason.terminal), True, exact=True))
                # No plan/submission is performed by this failed proposal.
                runner.remove_request(2)
                runner.remove_request(0)
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
                                             reference_state.resources[key].kv_cache.tensor[:, list(pages)]))
                checks.extend((
                    Comparison("trajectory.other_request_preserved.position",
                               resources[POS].position(1, "main"), survivor_position, exact=True),
                    Comparison("trajectory.other_request_preserved.sampling",
                               resources[SAMPLER]._sampler._step_offset[1], survivor_sampling, exact=True),
                ))
                runner.remove_request(1)
                runner.remove_request(2)
            else:
                runner.remove_request(0)
        for key in (KV_CACHE, CROSS_KV_CACHE):
            checks.append(Comparison(("trajectory." if companion else "") + key + ".cleanup_pages",
                                     resources[key]._arena.num_free,
                                     resources[key].config.max_num_pages - 1, exact=True))
        return checks


def adapters(model, features, **kwargs):
    adapter = WhisperCaseAdapter(model, features, **kwargs)
    return {WhisperEncoderSubmodule: adapter, WhisperDecoderSubmodule: adapter}
