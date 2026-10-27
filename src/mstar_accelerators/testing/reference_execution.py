"""Reference computation submitted through the same M* engine as candidates.

This adapter owns neither admission, planning, request state nor commit. Its
physical program is called exclusively by Engine's ordinary callable runner.
"""
from types import SimpleNamespace
from dataclasses import dataclass
from mstar.engine.execution import ExecutionCapabilities, ExecutionOption, Support
from mstar.model.binding import NodeAssembly
from .runtime import load_engine, execute


@dataclass(frozen=True)
class _ReferenceResource:
    factory: object
    def build(self, spec, info):
        return self.factory(spec, info)


class ReferenceExecution:
    def __init__(self, binding, program, session, *, providers=None):
        self.binding, self.program = binding, program
        def bind(resources):
            program.bind_resources(resources)
            return ExecutionCapabilities((ExecutionOption(
                name="logical-reference", supports=lambda facts: Support.YES,
                acquire=lambda facts: SimpleNamespace(close=lambda: None),
                materialize=lambda invocation, execution: invocation,
                primitive=lambda invocation, execution: program.run(invocation),
                close=getattr(program, "close", None)),),
                {walk: binding.model_config.get("max_concurrent_requests", 1)
                 for walk in binding.graph_walks})
        from mstar.engine.resources.kv.config import KVSpec
        # KV uses the runtime's ordinary allocator and layout resolver. Other
        # numerical resources use its existing physical-request extension point.
        sequence_keys = {spec.resource_key for spec in binding.resource_specs if isinstance(spec, KVSpec)}
        requests = {key: (_ReferenceResource(factory),) for key, factory in (providers or {}).items()
                    if key not in sequence_keys}
        assembly = NodeAssembly(bind=bind, resource_requests=requests, execution_names=("logical-reference",))
        self.engine = load_engine(binding, assembly, session)

    @property
    def resources(self):
        return self.engine._resources

    def add_request(self, rid, configs=None):
        self.engine.add_request(rid, configs)

    def remove_request(self, rid):
        self.engine.remove_request(rid)

    def run(self, walk, rid, inputs, *, metadata=None):
        batch, observations = execute(self.engine, self.binding.node_name, walk, rid, inputs,
                                      metadata=metadata)
        if batch.admit_error is not None:
            raise RuntimeError(f"reference admission failed: {batch.admit_error}")
        return batch, observations[0][0]
