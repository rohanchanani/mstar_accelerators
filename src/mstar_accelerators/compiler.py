"""Default call binding derived from a model's logical node contract."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch


@dataclass(frozen=True)
class PreparationScope:
    """Placement-specialized claims, available to optional shared helpers.

    A requirement is a compiler diagnostic, never an extra serving predicate.
    The bounded analysis reports only actual counterexamples it can establish;
    no result is not a proof of compatibility.
    """

    name: str
    claims: Mapping[str, Any]
    facts: Mapping[str, Any]
    domain: Any = None

    def require(self, requirement, *, origin):
        from .conditions import All, contradiction
        for walk, claim in self.claims.items():
            if self.domain is not None:
                claim = All(self.domain, claim)
            witness = contradiction(claim, requirement, dict(self.facts) | {"graph_walk": walk})
            if witness is not None:
                raise ValueError(
                    f"{self.name}: implementation contradicts when; {origin}; "
                    f"when admits {witness!r}, requirement is {requirement!r}"
                )


@dataclass(frozen=True)
class BoundCall:
    graph_walk: str
    inputs: Any
    outputs: Any
    plan_results: Any
    row_walks: tuple[str, ...]


class CompletedLease:
    def close(self):
        pass


def _shape(shape, *, tokens, batch):
    dimensions = {"tokens": tokens, "batch": batch}
    return tuple(dimensions[value] if isinstance(value, str) else value for value in shape)


class DefaultBinding:
    """Compile named logical fields into ordinary Torch arguments and results.

    Inputs remain separated by request. Implementations may replace this
    materializer to choose packed, padded or native physical representations.
    """

    def __init__(self, binding, resources, *, input_type=SimpleNamespace,
                 output_type=None, call_type=BoundCall, plan_type=SimpleNamespace):
        self.binding = binding
        self.resources = resources
        self.contract = binding.submodule.logical_contract()
        self.input_type = input_type
        self.output_type = output_type or self.contract.result_type
        self.call_type = call_type
        self.plan_type = plan_type

    def materialize(self, invocation, execution):
        del execution
        batch = len(invocation.inputs)
        row_walks = invocation.row_walks or (invocation.graph_walk,) * batch
        inputs = {}
        for spec in self.contract.inputs:
            if spec.walks and not any(walk in spec.walks for walk in row_walks):
                inputs[spec.name] = ()
                continue
            rows = []
            for walk, logical in zip(row_walks, invocation.inputs):
                if spec.walks and walk not in spec.walks:
                    rows.append(None)
                    continue
                value = logical
                for component in spec.source.split("."):
                    value = value[component] if isinstance(value, Mapping) else getattr(value, component)
                expected = _shape(spec.shape, tokens=getattr(logical, "input_seq_len", None), batch=batch)
                if tuple(value.shape) != expected:
                    raise ValueError(f"{spec.name}: expected logical shape {expected}, got {tuple(value.shape)}")
                rows.append(value.to(dtype=spec.dtype or self.binding.dtype).contiguous().to(
                    self.binding.device, non_blocking=True))
            inputs[spec.name] = tuple(rows)
        outputs = {}
        for spec in self.contract.outputs:
            shape = _shape(spec.shape, tokens=None, batch=batch)
            values = tuple(torch.empty(shape, dtype=spec.dtype or self.binding.dtype,
                                       device=self.binding.device)
                           for _ in range(batch if spec.per_request else 1))
            outputs[spec.name] = values if spec.per_request else values[0]
        return self.call_type(graph_walk=invocation.graph_walk, inputs=self.input_type(**inputs),
                              outputs=self.output_type(**outputs),
                              plan_results=self.plan_type(**{
                                  key: self.resources[key].logical_use(plan)
                                  if hasattr(self.resources[key], "logical_use") else plan
                                  for key, plan in invocation.plan_results.items()
                              }), row_walks=row_walks)

    def result(self, call):
        return call.outputs

    def zero(self, call, execution):
        del execution
        for spec in self.contract.outputs:
            value = getattr(call.outputs, spec.name)
            for tensor in value if spec.per_request else (value,):
                tensor.zero_()
        return self.result(call)

    def acquire(self, facts):
        del facts
        return CompletedLease()

    def close(self):
        pass


def bind_default_call(binding, resources, **types):
    return DefaultBinding(binding, resources, **types)


def prepare_placeholder(binding, resources, **types):
    """Executable placeholder behavior; its assessment belongs to the harness."""
    from .definition import PreparedImplementation
    default = bind_default_call(binding, resources, **types)
    return PreparedImplementation(default.materialize, default.zero, default.acquire,
                                  default.close)


def default_resource_requests(binding):
    """An editable initial physical choice, not a model or call-ABI constraint."""
    from mstar.engine.resources.kv.config import KVSpec
    from mstar.engine.resources.sequence import PagedSequenceView
    return {spec.resource_key: PagedSequenceView() for spec in binding.resource_specs
            if binding.node_name in spec.nodes and isinstance(spec, KVSpec)}


def pack_tensor_result(result, contract):
    """Organize logical tensors for the tensor runner, using logical field names.

    Row organization is physical. Routing these values onto graph edges remains
    the logical shell's execution_result responsibility.
    """
    from mstar.model.submodule_base import BatchedModelOutput

    rows, packed = {}, {}
    for spec in contract.outputs:
        value = getattr(result, spec.name)
        if spec.per_request:
            rows[spec.name] = torch.stack(tuple(value))
        elif spec.shape and spec.shape[0] == "batch":
            rows[spec.name] = value
        else:
            packed[spec.name] = value
    return BatchedModelOutput(row_outputs=rows or None, packed_outputs=packed)


def bind_tensor_result(binding):
    """Compile physical output organization back to the logical result record."""
    from mstar.engine.execution import ExecutionResult
    from mstar.model.submodule_base import BatchedModelOutput

    contract = binding.submodule.logical_contract()

    def result(raw, invocation):
        if isinstance(raw, ExecutionResult):
            return raw
        if isinstance(raw, contract.result_type):
            return raw
        raw = BatchedModelOutput.coerce(raw)
        if (not raw.row_outputs and not raw.packed_outputs
                and not any(raw.per_rid_outputs.values())):
            # Some state-only walks produce no logical outputs. The model's
            # mapper knows that walk and does not manufacture unused tensors.
            return None
        values = {}
        count = len(invocation.request_ids)
        for spec in contract.outputs:
            if raw.row_outputs is not None and spec.name in raw.row_outputs:
                value = raw.row_outputs[spec.name]
                if spec.per_request:
                    value = tuple(value[index] for index in range(count))
            elif spec.name in raw.packed_outputs:
                value = raw.packed_outputs[spec.name]
            else:
                rows = []
                for rid in invocation.request_ids:
                    value = raw.per_rid_outputs[rid][spec.name]
                    if isinstance(value, (list, tuple)):
                        if len(value) != 1:
                            raise ValueError(f"{spec.name}: expected one tensor per row")
                        value = value[0]
                    rows.append(value)
                if spec.per_request:
                    value = tuple(rows)
                elif spec.shape and spec.shape[0] == "batch":
                    tail = tuple(spec.shape[1:])
                    value = (torch.stack(rows) if tuple(rows[0].shape) == tail
                             else torch.cat(rows, dim=0))
                elif len(rows) == 1:
                    value = rows[0]
                else:
                    raise ValueError(f"{spec.name}: a whole-invocation output needs packed storage")
            expected = _shape(spec.shape, tokens=None, batch=count)
            tensors = value if spec.per_request else (value,)
            if spec.per_request and len(tensors) != count:
                raise ValueError(f"{spec.name}: expected {count} result rows")
            for tensor in tensors:
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
                    raise ValueError(f"{spec.name}: expected logical output shape {expected}")
                dtype = spec.dtype or binding.dtype
                if tensor.dtype != dtype:
                    raise ValueError(f"{spec.name}: expected logical output dtype {dtype}, got {tensor.dtype}")
            values[spec.name] = value
        return contract.result_type(**values)

    return result
