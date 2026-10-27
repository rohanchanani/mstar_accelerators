"""Optional singleton graph bindings over explicitly chosen resource views.

Named graph ports are logical input names, logical weight identities, and
``resource.label.view`` names. This physical ABI is one implementation choice;
ordinary preparation callbacks can provide other representations.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
import os

import torch
from mstar.engine.resources.attn.config import AttentionSpec, CrossAttentionSpec
from mstar.engine.resources.kv.config import KVSpec
from mstar.engine.resources.position.config import PositionSpec
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.resources.sequence import DenseSequenceView, PagedSequenceView

from mstar_accelerators.compiler import bind_default_call
from mstar_accelerators.definition import PreparedImplementation
from mstar_accelerators.backends.weights import prepare_weights

from .paged import read_rows, write_rows


@dataclass(frozen=True)
class PagedLayout:
    configs: Mapping[str, object]

    @classmethod
    def from_resources(cls, binding, resources):
        return cls({spec.resource_key: resources[spec.resource_key].config
                    for spec in binding.resource_specs if isinstance(spec, KVSpec)
                    and binding.node_name in spec.nodes})

    def rows(self, resource):
        config = self.configs[resource]
        return config.max_num_pages * 2 * config.page_size


class GraphBuilder(Protocol):
    """Native preparation calls build(); explicit interpretation supplies graph=.

    Graph implementations are optional: this is the helper's calling convention,
    not an acceleration or CLI contract.
    """

    def __call__(self, *, graph: Any | None = None) -> Any: ...


@dataclass(frozen=True)
class GraphProgram:
    build: GraphBuilder
    slots: int = 2
    extra_outputs: tuple = ()


def paged_views(binding, resources):
    """Bind the implementation's physical access once during preparation."""
    return {spec.resource_key: resources[spec.resource_key].view(PagedSequenceView())
            for spec in binding.resource_specs if isinstance(spec, KVSpec)
            and binding.node_name in spec.nodes}


def dense_views(binding, resources):
    """Bind contiguous sequence aliases without allocating or admitting state."""
    return {spec.resource_key: resources[spec.resource_key].view(DenseSequenceView())
            for spec in binding.resource_specs if isinstance(spec, KVSpec)
            and binding.node_name in spec.nodes}


def _logical_graph_inputs(defaults, call, *, sampling=False):
    """Logical inputs and owner-plan snapshots, independent of cache layout."""
    values = {}
    for spec in defaults.contract.inputs:
        rows = getattr(call.inputs, spec.name)
        if rows:
            if len(rows) != 1:
                raise ValueError("this graph view ABI supports singleton invocations")
            values[spec.name] = rows[0]
    for spec in defaults.binding.resource_specs:
        if isinstance(spec, PositionSpec) and defaults.binding.node_name in spec.nodes:
            for label, ids in getattr(call.plan_results, spec.resource_key).items():
                values[f"{spec.resource_key}.{label}.ids"] = ids.to(torch.int32)
        elif sampling and isinstance(spec, SamplerSpec) and defaults.binding.node_name in spec.nodes:
            plan = getattr(call.plan_results, spec.resource_key)
            # The owner resolves seeds and snapshots draw offsets in row order.
            # These are ordinary read-only graph inputs, never backend RNG state.
            values[f"{spec.resource_key}.seeds"] = torch.tensor(
                [config.seed for config in plan.configs], dtype=torch.int64)
            values[f"{spec.resource_key}.offsets"] = torch.tensor(plan.offsets, dtype=torch.int64)
            values[f"{spec.resource_key}.temperatures"] = torch.tensor(
                [config.temperature for config in plan.configs], dtype=torch.float32)
    return values


def _stage_graph_inputs(binding, values):
    # Storage aliases are already contiguous on the bound device. Only ordinary
    # logical inputs and host plan tensors need staging; no allocator is walked.
    return {name: value.contiguous().to(binding.device, non_blocking=True)
            for name, value in values.items()}


def graph_bindings(defaults, call, *, views=None, sampling=False):
    """Lower the paged graph ABI from admitted logical resource-use handles."""
    binding, resources = defaults.binding, defaults.resources
    views = paged_views(binding, resources) if views is None else views
    plans = vars(call.plan_results)
    values = _logical_graph_inputs(defaults, call, sampling=sampling)
    for spec in binding.resource_specs:
        key = spec.resource_key
        if binding.node_name not in spec.nodes:
            continue
        if isinstance(spec, KVSpec):
            resource = resources[key]
            config = resource.config
            access = views[key]
            width = config.num_kv_heads * config.head_dim
            for layer in range(config.num_layers):
                values[f"{key}.layer.{layer}"] = access.layer(layer).view(-1, width)
            for label, use in plans[key].items():
                plan = access.for_step(use)
                rows = write_rows(plan.write_locations, page_size=config.page_size,
                                  num_pages=config.max_num_pages)
                values[f"{key}.{label}.write.keys"] = rows.keys
                values[f"{key}.{label}.write.values"] = rows.values
        elif isinstance(spec, (AttentionSpec, CrossAttentionSpec)):
            config = resources[spec.config.kv_cache].config
            access = views[spec.config.kv_cache]
            for label, use in plans[key].items():
                plan = access.for_attention(use)
                rows = read_rows(plan, capacity=config.max_seq_len, num_pages=config.max_num_pages)
                values[f"{key}.{label}.read.keys"] = rows.keys
                values[f"{key}.{label}.read.values"] = rows.values
                values[f"{key}.{label}.read.mask"] = rows.mask
    return _stage_graph_inputs(binding, values)


def dense_graph_bindings(defaults, call, *, views, sampling=False):
    """A contiguous-cache graph ABI using the resource owner's selected views.

    ``<kv>.<label>.keys.<layer>`` and ``values.<layer>`` are aliased full-capacity
    matrices with Torch shape [capacity, local heads * head_dim]. The int64
    ``<kv>.<label>.write.positions`` identifies this admitted append. Attention
    ports ``<attn>.<label>.read.keys.<layer>``/``values.<layer>`` expose the same
    storage in query/context order; ``read.mask`` is [queries, capacity].

    A graph writing a cache must feed the returned write alias into its reads,
    just as the paged builder does. A second input alias alone does not establish
    that graph dependency. No cache data is gathered or copied by this adapter.

    ``sampling=True`` also stages ordered I64 seeds/offsets and F32 temperatures
    from the existing SamplingPlan. It neither selects a sampling algorithm nor
    advances offsets. Greedy programs need no extra plan transfers.
    """
    binding = defaults.binding
    plans = vars(call.plan_results)
    values = _logical_graph_inputs(defaults, call, sampling=sampling)
    for spec in binding.resource_specs:
        key = spec.resource_key
        if binding.node_name not in spec.nodes:
            continue
        if isinstance(spec, KVSpec):
            access = views[key]
            width = access.config.num_kv_heads * access.config.head_dim
            for label, use in plans[key].items():
                rows = access.for_step(use)
                if len(rows) != 1:
                    raise ValueError("this graph view ABI supports singleton sequence uses")
                row, = rows
                values[f"{key}.{label}.write.positions"] = row.positions
                for layer, (keys, vals) in enumerate(zip(row.keys, row.values)):
                    values[f"{key}.{label}.keys.{layer}"] = keys.view(-1, width)
                    values[f"{key}.{label}.values.{layer}"] = vals.view(-1, width)
        elif isinstance(spec, (AttentionSpec, CrossAttentionSpec)):
            access = views[spec.config.kv_cache]
            width = access.config.num_kv_heads * access.config.head_dim
            for label, use in plans[key].items():
                rows = access.for_attention(use)
                if len(rows) != 1:
                    raise ValueError("this graph view ABI supports singleton attention uses")
                row, = rows
                values[f"{key}.{label}.read.mask"] = row.mask
                for layer, (keys, vals) in enumerate(zip(row.keys, row.values)):
                    values[f"{key}.{label}.read.keys.{layer}"] = keys.view(-1, width)
                    values[f"{key}.{label}.read.values.{layer}"] = vals.view(-1, width)
    return _stage_graph_inputs(binding, values)


def categorical_tokens(graph, logits, sampler, *, rows=1):
    """An optional graph realization of a declared categorical sampling policy.

    The implementation selects this callback and admits its actual policy in
    ``when``. Static model suppression has already been applied to logits. Pair
    with either graph input adapter's ``sampling=True`` option; this helper owns
    only the physical port convention, not request policy or RNG lifetime.
    """
    return graph.categorical(
        logits,
        graph.tensor(f"{sampler}.seeds", (rows,), dtype="i64"),
        graph.tensor(f"{sampler}.offsets", (rows,), dtype="i64"),
        graph.tensor(f"{sampler}.temperatures", (rows,), dtype="f32"),
    )


def graph_outputs(defaults, call):
    outputs = {}
    for spec in defaults.contract.outputs:
        value = getattr(call.outputs, spec.name)
        if spec.per_request:
            if len(value) != 1:
                raise ValueError("this graph view ABI supports singleton outputs")
            value = value[0]
        outputs[spec.name] = value
    return outputs


class _ModeLease:
    def __init__(self, walk, lease):
        self.walk = walk
        self.lease = lease
        self.extra_outputs = {}

    def submit(self, inputs, outputs):
        self.lease.submit(inputs, outputs)

    def close(self):
        self.lease.close()


def prepare_graph_programs(binding, resources, programs: Mapping[str, GraphProgram], *, provider,
                           call_binding=None, input_bindings=None):
    """Prepare optional ggml programs; the implementation chooses its provider.

    input_bindings may replace this helper's paged port convention. It receives
    the ordinary default call binding and the materialized call, and returns the
    chosen program's input views. It does not own resource admission or state.
    """
    from mstar_accelerators.conditions import Eq, In
    scope = binding.services
    if scope is not None:
        scope.require(In("graph_walk", tuple(programs)), origin="graph program mapping")
        scope.require(Eq("batch_size", 1), origin="singleton graph input/output ABI")
    hosted = binding.graph_walks
    if hosted is not None:
        programs = {walk: program for walk, program in programs.items() if walk in hosted}
    defaults = call_binding or bind_default_call(binding, resources)
    if input_bindings is None:
        views = paged_views(binding, resources)
        input_bindings = lambda defaults, call: graph_bindings(defaults, call, views=views)
    weights = prepare_weights(binding, defaults.contract.weight_prefix)
    prepared = {}
    try:
        for walk, program in programs.items():
            constants = dict(weights)
            for spec in binding.resource_specs:
                if isinstance(spec, SamplerSpec) and binding.node_name in spec.nodes:
                    mask = torch.zeros((1, spec.vocab_size), dtype=binding.dtype)
                    suppressed = defaults.contract.suppressed_tokens.get(spec.resource_key, {}).get(walk, ())
                    if suppressed:
                        mask[:, suppressed] = float("-inf")
                    constants[f"{spec.resource_key}.logit_mask"] = mask.to(binding.device)
            prepared[walk] = provider.prepare(program.build, constants, slots=program.slots)
    except BaseException:
        for program in reversed(tuple(prepared.values())):
            program.close()
        raise

    def acquire(facts):
        lease = prepared[facts.graph_walk].try_acquire()
        return None if lease is None else _ModeLease(facts.graph_walk, lease)

    def materialize(invocation, execution):
        call = defaults.materialize(invocation, execution)
        execution.workspace.extra_outputs = {
            spec.name: torch.empty(spec.shape, dtype=spec.dtype or binding.dtype, device=binding.device)
            for spec in programs[invocation.graph_walk].extra_outputs
        }
        return call

    def primitive(call, execution):
        if call.graph_walk != execution.workspace.walk:
            raise ValueError("acquired graph program does not match the invocation walk")
        execution.workspace.submit(input_bindings(defaults, call),
                                   graph_outputs(defaults, call) | execution.workspace.extra_outputs)
        return defaults.result(call)

    def close():
        for program in reversed(tuple(prepared.values())):
            program.close()

    return PreparedImplementation(materialize, primitive, acquire, close)


class _NativeLease:
    def __init__(self, lease, inputs, outputs):
        self.lease, self.inputs, self.outputs = lease, inputs, outputs

    def submit(self, inputs, outputs):
        self.lease.submit({name: inputs[name] for name in self.inputs},
                          {name: outputs[name] for name in self.outputs})

    def close(self):
        self.lease.close()


class _NativeProgram:
    def __init__(self, program, inputs, outputs):
        self.program, self.inputs, self.outputs = program, inputs, outputs

    def try_acquire(self):
        lease = self.program.try_acquire()
        return None if lease is None else _NativeLease(lease, self.inputs, self.outputs)

    def close(self):
        self.program.close()


class NativeGraphProvider:
    """Adapt the explicit native preparation API without a device fallback."""

    def prepare(self, builder, constants, *, slots):
        from mstar_pi.ggml import prepare_graph
        info = None

        def factory():
            nonlocal info
            graph = builder()
            if info is None:
                info = graph.info()
            return graph

        program = prepare_graph(factory, constants, slots=slots)
        if os.environ.get("MSTAR_ACCELERATORS_GRAPH_DIAGNOSTICS") == "1":
            import json
            import logging
            logging.getLogger(__name__).warning("%s", json.dumps({
                "event": "native_graph_prepared", "graph": info,
                "native_program": program.info(),
            }, sort_keys=True))
        return _NativeProgram(program, tuple(name for name in info["inputs"] if name not in constants),
                              tuple(info["outputs"]))
