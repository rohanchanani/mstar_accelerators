"""Specialize authored support, collect physical requests, then bind programs."""
from collections.abc import Mapping
from dataclasses import replace
import logging

from mstar.engine.execution import ExecutionCapabilities, ExecutionOption, InvocationFacts, Support
from mstar.engine.tensor_forward import TensorForward, bind_tensor_forward
from mstar.engine.resources.binding import merge_resource_requests
from mstar.model.binding import NodeAssembly

from .binding import invocation_facts, static_facts
from .compiler import PreparationScope, bind_tensor_result
from .conditions import Constant, Or, Range, batch_bound
from .definition import Acceleration, TensorForwardExecution

logger = logging.getLogger(__name__)


class Registry:
    def __init__(self, accelerations: Mapping[str, Acceleration]):
        self.accelerations = dict(accelerations)

    def assemble(self, binding, *, require_coverage=True) -> NodeAssembly:
        """Compile a placement and optionally build resource-independent code.

        Qualification uses the same stages with coverage disabled when binding
        one candidate for a witness. Its finite case domain is never a claim
        about the deployment's admitted requests.
        """
        static = static_facts(binding)
        candidates = []
        resource_requests = {}
        limits = {}
        walks = binding.graph_walks
        if walks is None:
            walks = tuple(
                walk for walk, graph in binding.model.get_graph_walk_graphs().items()
                if binding.node_name in graph.get_nodes()
            )
        declared = {spec.resource_key for spec in binding.resource_specs
                    if binding.node_name in spec.nodes}
        for name, acceleration in self.accelerations.items():
            if acceleration.reference is not type(binding.submodule):
                continue
            condition = acceleration.when.specialize(static)
            local = {}
            for walk in walks:
                residual = condition.specialize({"graph_walk": walk})
                if residual.evaluate({}) is not Support.NO:
                    local[walk] = residual
            if not local:
                continue
            # All downstream work uses this candidate's actual local scope.
            admitted = binding.model_config.get("max_concurrent_requests")
            domain = None if admitted is None else Range("batch_size", 1, admitted)
            scoped_binding = replace(binding, graph_walks=tuple(local),
                                     services=PreparationScope(name, local, static, domain))
            for walk, residual in local.items():
                bound = batch_bound(residual)
                if admitted is not None:
                    bound = admitted if bound is None else min(bound, admitted)
                if bound is None:
                    raise ValueError(f"{name}/{walk}: a finite batch search bound must come from when or deployment admission")
                limits[walk] = max(limits.get(walk, 0), bound)
            requested = acceleration.implementation.resources
            requested = requested(scoped_binding) if callable(requested) else requested
            if foreign := requested.keys() - declared:
                raise ValueError(f"{name}: undeclared resource requests {sorted(foreign)}")
            resource_requests = merge_resource_requests(resource_requests, {
                key: value if isinstance(value, tuple) else (value,)
                for key, value in requested.items()
            })
            candidates.append((name, acceleration, condition, local, scoped_binding))
        if not candidates:
            raise ValueError(f"no acceleration binds {binding.node_name} on {static}")
        if require_coverage and (missing := set(walks) - limits.keys()):
            raise ValueError(f"no execution covers walks {sorted(missing)}")
        device_plan_resources = frozenset(
            key for key, requests in resource_requests.items()
            if any(getattr(request, "device_plan", False) for request in requests)
        )

        # Positive proof only: missing invocation facts must simplify away.
        fixed = None
        admitted = binding.model_config.get("max_concurrent_requests")
        if len(candidates) == 1 and admitted is not None and admitted > 0:
            name, acceleration, condition, local, _ = candidates[0]
            execution = acceleration.implementation.execution
            if (isinstance(execution, TensorForwardExecution) and execution.bind_result is None
                    and binding.submodule.execution_result_is_output_only
                    and set(local) == set(walks)
                    and binding.tp_group.world_size == binding.sp_group.world_size == 1
                    and all(condition.specialize({"batch_size": n}) == Constant(Support.YES)
                            for n in range(1, admitted + 1))):
                fixed = name

        early = {}
        def close_pending():
            for program in tuple(early.values()):
                close = getattr(program, "close", None)
                if close is not None:
                    close()
            early.clear()

        try:
            for name, acceleration, _, _, local_binding in candidates:
                execution = acceleration.implementation.execution
                if isinstance(execution, TensorForwardExecution) and execution.build is not None:
                    program = execution.build(local_binding)
                    program.requires_grad_(False)
                    early[name] = program
        except BaseException:
            close_pending()
            raise

        def bind(resources):
            options = []
            pending = None
            fixed_name = fixed
            try:
                for name, acceleration, condition, local, local_binding in candidates:
                    if name in early:
                        prepared = early.pop(name)
                        pending = prepared
                        resource_binder = getattr(prepared, "bind_resources", None)
                        if resource_binder is not None:
                            resource_binder(resources)
                        elif acceleration.implementation.resources:
                            # A stateless program can explicitly request no views.
                            requested = acceleration.implementation.resources
                            requested = requested(local_binding) if callable(requested) else requested
                            if requested:
                                raise TypeError(f"{name}: early program needs bind_resources")
                    else:
                        prepared = acceleration.implementation.prepare(local_binding, resources)
                        pending = prepared

                    def supports(facts, condition=condition, local=local):
                        values = invocation_facts(facts)
                        if facts.row_walks is not None:
                            if any(walk not in local for walk in facts.row_walks):
                                return Support.NO
                            if len(set(facts.row_walks)) > 1:
                                # Mixed composition is an invocation fact. Do not
                                # silently turn a supporting program into uniform
                                # batches, or mistake the queue label for a row walk.
                                return condition.evaluate(values)
                        if facts.graph_walk is None:
                            # Request admission has policy facts, not a proposed
                            # shape or walk. Only conclusive rejection is useful.
                            return Or(*local.values()).evaluate(values)
                        residual = local.get(facts.graph_walk)
                        return Support.NO if residual is None else residual.evaluate(values)

                    execution = acceleration.implementation.execution
                    # The common lifecycle prepares rows inline. Programs with
                    # only a late row adapter retain the general consumer.
                    if (name == fixed_name and callable(getattr(prepared, "prepare_rows", None))
                            and not callable(getattr(prepared, "input_preparation", None))):
                        fixed_name = None
                    if isinstance(execution, TensorForwardExecution):
                        options.append(bind_tensor_forward(
                            name, supports, local_binding, resources,
                            TensorForward(
                                prepared, compile=execution.compile,
                                capture=execution.capture,
                                autocast_dtype=execution.autocast_dtype,
                                result=(execution.bind_result(local_binding)
                                        if execution.bind_result is not None else None),
                                close=getattr(prepared, "close", None),
                            ),
                            cost=acceleration.implementation.cost,
                        ))
                    elif execution is not None:
                        raise TypeError(f"unsupported execution description: {type(execution).__name__}")
                    else:
                        options.append(ExecutionOption(
                            name=name,
                            supports=supports,
                            acquire=prepared.acquire,
                            materialize=prepared.materialize,
                            primitive=prepared.primitive,
                            cost=acceleration.implementation.cost,
                            close=prepared.close,
                        ))
                    pending = None
            except BaseException:
                if pending is not None and (close := getattr(pending, "close", None)) is not None:
                    close()
                close_pending()
                for option in reversed(options):
                    if option.close is not None:
                        option.close()
                raise
            capabilities = ExecutionCapabilities(tuple(options), limits,
                fixed_option=next((option for option in options if option.name == fixed_name), None),
                fixed_admission=admitted if fixed_name is not None else 0)
            # A useful check for the current admission contract, not a universal
            # coverage proof. Unknown shape/policy facts remain unknown.
            if require_coverage:
                for walk in walks:
                    if not capabilities.possibilities(InvocationFacts(walk, 1)):
                        capabilities.close()
                        raise ValueError(f"{walk}: admitted single requests have no singleton execution")
            logger.info("Bound %s: implementations=%s batch_limits=%s", binding.node_name,
                        [option.name for option in options], limits)
            return capabilities

        return NodeAssembly(bind=bind, close=close_pending, resource_requests=resource_requests,
                            execution_names=tuple(name for name, *_ in candidates),
                            device_plan_resources=device_plan_resources)
