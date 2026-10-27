"""Test inputs and read-only observations around M*'s real Engine.

There is no qualification implementation of reserve/admit/plan/execute/commit.
The engine owns that sequence, resource construction, agreement and teardown.
"""
import torch
from dataclasses import replace

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.execution import ExecutionCapabilities
from mstar.engine.resources.binding import merge_resource_requests
from mstar.engine.resources.step import StepContext
from mstar.model.binding import NodeAssembly


def join_placements(*assemblies):
    """Combine case-selected, walk-scoped placements before engine allocation.

    This chooses which registration to test on each walk, not how to execute it.
    Registry.assemble has already scoped each option to its hosted walks.
    """
    if len(assemblies) == 1:
        return assemblies[0]

    def bind(resources):
        bound = []
        try:
            for assembly in assemblies:
                bound.append(assembly.bind(resources))
            limits = {}
            for capabilities in bound:
                for walk, limit in capabilities.batch_limits.items():
                    limits[walk] = max(limits.get(walk, 0), limit)
            return ExecutionCapabilities(tuple(option for cap in bound for option in cap.options), limits)
        except BaseException:
            for cap in reversed(bound):
                cap.close()
            raise

    def close():
        for assembly in assemblies:
            if getattr(assembly, "close", None) is not None:
                assembly.close()

    changes = dict(bind=bind, resource_requests=merge_resource_requests(*(a.resource_requests for a in assemblies)),
                   execution_names=tuple(name for a in assemblies for name in a.execution_names))
    if hasattr(assemblies[0], 'device_plan_resources'):
        changes['device_plan_resources'] = frozenset().union(*(a.device_plan_resources for a in assemblies))
    if hasattr(assemblies[0], 'close'):
        changes['close'] = close
    return replace(assemblies[0], **changes)


def load_engine(binding, assembly, session, parallel_groups=None):
    engine = Engine(graph_runtime=None, autocast_dtype=binding.dtype)
    session.engines.append(engine)  # Retain partial load until the managed drain.
    session.assemblies.append(assembly)
    binding.submodule.bind_execution(binding, None)
    engine.load_model(
        {binding.node_name: binding.submodule}, specs=list(binding.resource_specs),
        parallel_groups=parallel_groups or WorkerParallelGroups(1, 0),
        device=binding.device, transfer_engine_info=None, kv_cache_type=binding.dtype,
        model=binding.model, assemblies={binding.node_name: assembly},
        bindings={binding.node_name: binding}, resource_requests=assembly.resource_requests,
    )
    engine.warmup()
    return engine


@torch.no_grad()
def execute(engine, node, walk, rid, inputs, *, resource_configs=None, observe=None, metadata=None):
    info = CurrentForwardPassInfo(f"qualification:{rid}", walk, 0, 0, 448,
                                  rid_handle=rid, resource_configs=resource_configs or {}, step_metadata=metadata or {})
    batch = ExecutingBatch(node, {rid: info}, StepContext([rid], walk, 0, False),
                           per_request_input_tensors={rid: inputs})
    shell = engine._submodules[node].submodule
    original = shell.execution_result
    observations = []

    def record(result, invocation):
        # Observe the existing model adaptation boundary before Engine commits.
        # Always invoke the original method unchanged; no injected execution.
        extra = None if observe is None else observe(result, invocation)
        observations.append((result, invocation, extra))
        return original(result, invocation)

    had_override = "execution_result" in shell.__dict__
    shell.execution_result = record
    try:
        engine.prepare_inputs(batch)
        if batch.failed_requests or batch.skipped_rids:
            raise RuntimeError(f"qualification inputs rejected: {batch.failed_requests}; {batch.skipped_rids}")
        engine.exec(batch)
    finally:
        if had_override:
            shell.execution_result = original
        else:
            del shell.execution_result
    if batch.admit_error is None and len(observations) != 1:
        raise RuntimeError("qualification expected one actual engine invocation")
    return batch, observations
