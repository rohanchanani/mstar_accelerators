"""Create fresh acceleration source files from an M* deployment description."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


EXECUTION_TEMPLATES = ("callable", "tensor-forward")


def _plan_annotation(spec):
    from mstar.engine.resources import (
        AttentionSpec,
        CrossAttentionSpec,
        KVSpec,
        PositionSpec,
        SamplerSpec,
    )

    if isinstance(spec, KVSpec):
        return "dict[str, SequenceUse]", "from mstar.engine.resources.sequence_view import SequenceUse"
    if isinstance(spec, (AttentionSpec, CrossAttentionSpec)):
        return "dict[str, AttentionUse]", "from mstar.engine.resources.sequence_view import AttentionUse"
    if isinstance(spec, PositionSpec):
        return "dict[str, torch.Tensor]", None
    if isinstance(spec, SamplerSpec):
        return "SamplingPlan", "from mstar.engine.resources.sampler.plan import SamplingPlan"
    raise TypeError(f"the default call generator has no plan view for {type(spec).__qualname__}")


def _tensor_program_lines(contract):
    if contract.dimensions is not None or any(spec.present_when is not None or "tokens" in spec.shape for spec in contract.outputs):
        return _row_tensor_program_lines(contract)
    result_fields = []
    for spec in contract.outputs:
        if any(isinstance(dimension, str) and dimension != "batch" for dimension in spec.shape):
            raise ValueError(f"tensor-forward output {spec.name!r} needs an explicit physical shape")
        dimensions = ["batch_size" if dimension == "batch" else repr(dimension)
                      for dimension in spec.shape]
        shape = "(" + ", ".join(dimensions) + ("," if len(dimensions) == 1 else "") + ")"
        dtype = str(spec.dtypes(None)[0]) if spec.dtype is not None else "self.dtype"
        value = f"torch.zeros({shape}, dtype={dtype}, device=self.device)"
        if spec.per_request:
            value = f"tuple({value} for _ in range(batch_size))"
        result_fields.append(f"            {spec.name}={value},")
    return [
        "class Program(torch.nn.Module):",
        "    \"\"\"Implementation-owned tensor packing, computation and capture recipes.\"\"\"",
        "",
        "    def __init__(self, binding, resources):",
        "        super().__init__()",
        "        self.device = binding.device",
        "        self.dtype = binding.dtype",
        "        self.contract = binding.submodule.logical_contract()",
        "        self.resources = resources",
        "        # Prepare physical weights and resource views here; keep the logical shell unchanged.",
        "",
        "    def preprocess(self, graph_walk, engine_inputs, inputs):",
        "        # Pack the already-interpreted input rows described by Inputs into forward kwargs.",
        "        # Plans/Call describe logical uses; choose physical views through self.resources.",
        "        return {'batch_size': len(inputs)}",
        "",
        "    def forward_batched(self, graph_walk, engine_inputs, *, batch_size):",
        "        # Replace this executable zero program with the intended physical computation.",
        f"        result = {contract.result_type.__name__}(",
        *result_fields,
        "        )",
        "        # Logical field names only; the model owns graph-edge and request semantics.",
        "        return pack_tensor_result(result, self.contract)",
        "",
        "    def forward(self, graph_walk, engine_inputs, *, batch_size):",
        "        return self.forward_batched(graph_walk, engine_inputs, batch_size=batch_size)",
        "",
        "    def can_batch(self, batch, inputs):",
        "        return True",
        "",
        "    def get_accelerator_graph_configs(self, device, tp_world_size=1):",
        "        # Add explicit static shapes and a forward method before enabling capture below.",
        "        return []",
        "",
        "    def close(self):",
        "        pass",
        "", "",
        "def prepare(binding, resources):",
        "    return Program(binding, resources)",
    ]


def render_acceleration(node, contract, specs, *, node_name, contexts, device_type, dtype,
                        execution="callable"):
    if execution not in EXECUTION_TEMPLATES:
        raise ValueError(f"unknown execution template {execution!r}; choose from {EXECUTION_TEMPLATES}")
    reference = type(node)
    imports = {
        "from dataclasses import dataclass",
        "from typing import Literal",
        "import torch",
        f"from {reference.__module__} import {reference.__name__}",
        f"from {contract.result_type.__module__} import {contract.result_type.__name__}",
    }
    if contract.input_preparation is not None:
        protocol = contract.input_preparation
        imports.add(f"from {protocol.__module__} import {protocol.__name__}")
    if execution == "callable":
        imports.update({
            "from mstar.engine.execution import Execution",
            "from mstar_accelerators import Acceleration, Eq, Implementation, Or, PreparedImplementation, When",
            "from mstar_accelerators.compiler import bind_default_call, default_resource_requests",
        })
        program_lines = [
            "def prepare(binding, resources):",
            "    defaults = bind_default_call(binding, resources, input_type=Inputs, call_type=Call, plan_type=Plans)",
            "",
            f"    def primitive(call: Call, execution: Execution) -> {contract.result_type.__name__}:",
            "        return defaults.zero(call, execution)",
            "",
            "    return PreparedImplementation(",
            "        materialize=defaults.materialize, primitive=primitive,",
            "        acquire=defaults.acquire, close=defaults.close,",
            "    )",
        ]
        implementation_lines = [
            "    # Callable execution; the generated call is representation-neutral.",
            "    implementation=Implementation(prepare=prepare, resources=default_resource_requests),",
        ]
    else:
        imports.update({
            "from mstar.engine.tensor_forward import CapturePolicy, CompilePolicy",
            "from mstar_accelerators import Acceleration, Eq, Implementation, Or, TensorForwardExecution, When",
            "from mstar_accelerators.backends.torch.tensor_forward import tensor_resource_requests",
            "from mstar_accelerators.compiler import pack_tensor_result",
        })
        program_lines = _tensor_program_lines(contract)
        implementation_lines = [
            "    implementation=Implementation(",
            "        prepare=prepare,",
            "        # Editable physical view and runtime numerical resource choices.",
            "        resources=tensor_resource_requests,",
            "        execution=TensorForwardExecution(",
            "            # Optional: build(binding) instead of prepare(binding, resources).",
            "            # Optional: bind_result=None for established named logical outputs.",
            "            compile=CompilePolicy(enabled=False),",
            "            capture=CapturePolicy(",
            "                enabled=False, required=False, compile=CompilePolicy(enabled=False),",
            "            ),",
            "            autocast_dtype=None,",
            "        ),",
            "    ),",
        ]
    plan_fields = []
    for spec in specs:
        annotation, statement = _plan_annotation(spec)
        if statement:
            imports.add(statement)
        plan_fields.append(f"    {spec.resource_key}: {annotation}")
    input_fields = []
    for spec in contract.inputs:
        detail = f"per-row shape {spec.shape!r}"
        if spec.walks:
            detail += f"; present on {', '.join(spec.walks)}; absent rows are None, all absent is empty"
        row_type = "torch.Tensor | None" if spec.walks else "torch.Tensor"
        input_fields.append(f"    {spec.name}: tuple[{row_type}, ...]  # {detail}")
    walks = tuple(sorted({walk for walk, _, _ in contexts}))
    branches = [f"When(Eq('graph_walk', {walk!r}), Eq('tp_size', {tp}), Eq('sp_size', {sp}))"
                for walk, tp, sp in sorted(contexts)]
    circumstances = branches[0] if len(branches) == 1 else "Or(" + ", ".join(branches) + ")"
    lines = [
        '"""Editable implementation of one logical node. Assessment belongs to the harness."""',
        "from __future__ import annotations", "", *sorted(imports), "", "",
        "@dataclass(frozen=True)", "class Inputs:", *(input_fields or ["    pass"]), "", "",
        "class Resources:",
        "    \"\"\"Model-owned identities; physical views are chosen in implementation.resources.\"\"\"",
        *(f"    {spec.resource_key} = {spec.resource_key!r}" for spec in specs), "", "",
        "@dataclass(frozen=True)", "class Plans:", *(plan_fields or ["    pass"]), "", "",
        "@dataclass(frozen=True)", "class Call:",
        f"    graph_walk: Literal[{', '.join(repr(walk) for walk in walks)}]",
        "    inputs: Inputs", f"    outputs: {contract.result_type.__name__}",
        "    plan_results: Plans", "    row_walks: tuple[str, ...]", "", "",
        *program_lines, "", "",
        "acceleration = Acceleration(",
        f"    reference={reference.__name__},",
        "    when=When(",
        f"        Eq('node_name', {node_name!r}),",
        *([f"        Eq('device_type', {device_type!r}),"] if device_type is not None else []),
        f"        Eq('dtype', torch.{dtype}),",
        f"        {circumstances},",
        "    ),",
        *implementation_lines,
        ")", "",
    ]
    return "\n".join(lines)


def create_accelerations(deployment: Path, destination: Path, *, nodes=None,
                         execution="callable") -> list[Path]:
    if execution not in EXECUTION_TEMPLATES:
        raise ValueError(f"unknown execution template {execution!r}; choose from {EXECUTION_TEMPLATES}")
    import torch
    import yaml
    from mstar.model.binding import resolve_autocast_dtype
    from mstar.model.registry import get_model_class, model_init_kwargs

    config = yaml.safe_load(deployment.read_text())
    model_name = config["model"]
    model_type = get_model_class(model_name)
    model = model_type.describe(**(model_init_kwargs(model_name) | config.get("model_kwargs", {})))
    worker_graphs = model.get_worker_graphs(str(deployment))
    # Placement and sharding are interpreted by M*, including walk-specific
    # groups. A physical worker occurrence never creates a second source file.
    model.get_sharding_config(str(deployment))
    contexts = defaultdict(set)
    for graph in worker_graphs:
        for name in graph.section.get_nodes():
            for walk in graph.graph_walks:
                contexts[name].add((walk, graph._tp_comm_size, graph.sp_size))
    selected = nodes if nodes is not None else (config.get("runtime_extension") or {}).get("nodes")
    if selected is not None:
        if (not isinstance(selected, (list, tuple)) or not selected
                or any(not isinstance(name, str) for name in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("selected nodes must be nonempty and unique")
        if unknown := set(selected) - contexts.keys():
            raise ValueError(f"selected nodes are absent from the deployment: {sorted(unknown)}")
        contexts = {name: contexts[name] for name in selected}
    specs = model.get_node_resources()
    dtype = str(resolve_autocast_dtype(model, config) or torch.float32).removeprefix("torch.")
    sources = {}
    for name in sorted(contexts):
        node = model.create_submodule(name)
        contract = node.logical_contract()
        if not name.isidentifier():
            raise ValueError(f"logical node {name!r} needs a filesystem name encoding")
        path = destination / (name + ".py")
        sources[path] = render_acceleration(
            node, contract, [spec for spec in specs if name in spec.nodes],
            node_name=name, contexts=contexts[name], device_type=config.get("device_type"), dtype=dtype,
            execution=execution,
        )
        compile(sources[path], str(path), "exec")
    existing = [str(path) for path in sources if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace acceleration source: {', '.join(existing)}")
    destination.mkdir(parents=True, exist_ok=True)
    for path, source in sources.items():
        with path.open("x") as output:
            output.write(source)
    return list(sources)


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="write one new acceleration file per logical node")
    create.add_argument("--deployment", type=Path, required=True)
    create.add_argument("--out-dir", type=Path, required=True)
    create.add_argument("--nodes", nargs="+", help="generate only these logical nodes (defaults to deployment selection)")
    create.add_argument("--execution", choices=EXECUTION_TEMPLATES, default="callable",
                        help="author-selected execution template; independent of deployment device (default: callable)")
    args = parser.parse_args(argv)
    try:
        paths = create_accelerations(args.deployment.resolve(), args.out_dir.resolve(),
                                     nodes=args.nodes, execution=args.execution)
    except (FileExistsError, FileNotFoundError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"created": [str(path) for path in paths]}, sort_keys=True))
    return 0



def _row_tensor_program_lines(contract):
    return [
        "class Program(torch.nn.Module):",
        "    def __init__(self, binding, resources):",
        "        super().__init__()",
        "        self.contract = binding.submodule.logical_contract()",
        "        self.dtype, self.device = binding.dtype, binding.device",
        "",
        "    def prepare_rows(self, inputs):",
        "        # Optional physical preparation before runtime padding. Preserve logical geometry.",
        "        return tuple(inputs)",
        "",
        "    def preprocess(self, graph_walk, engine_inputs, inputs):",
        "        return {'inputs': inputs}",
        "",
        "    def forward_batched(self, graph_walk, engine_inputs, inputs):",
        "        from mstar_accelerators.compiler import allocate_results",
        "        values = allocate_results(self.contract, inputs, engine_inputs, self.dtype, self.device, zero=True)",
        "        result = self.contract.result_type(**values)",
        "        return pack_tensor_result(result, self.contract, engine_inputs.request_ids)",
        "",
        "    forward = forward_batched",
        "",
        "    def can_batch(self, batch, inputs):",
        "        return True",
        "",
        "    def get_accelerator_graph_configs(self, device, tp_world_size=1):",
        "        return []",
        "",
        "def prepare(binding, resources):",
        "    return Program(binding, resources)",
    ]

if __name__ == "__main__":
    raise SystemExit(main())
