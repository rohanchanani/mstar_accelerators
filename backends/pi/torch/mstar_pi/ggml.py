"""Thin bindings to actual ggml contexts, tensors and prepared Vulkan programs.

Dimensions are ggml-native: ``ne[0]`` is contiguous/fastest. A bound Torch tensor
has the reversed contiguous shape; size-one high dimensions may be omitted.
Graph construction is host metadata only and never opens a device. Preparation
requires the explicitly initialized Pi runtime and creates one exclusive mutable
ggml context and canonical intermediate arena per slot. Constants are imported
once. Invocation inputs/state/outputs are borrowed native views, retained until
checked retirement; no host completion wait or hidden provider copy is inserted.

``set_rows(values, indices, destination)`` returns the destination alias. Feed
that alias into later reads/writes to establish their graph dependency. Indices
come from the runtime's checked resource plan; destination row indices must be
unique and in range. I64-to-I32 casts are for indices known to fit I32.
"""
from collections.abc import Callable, Mapping
from torch import Tensor
from ._C import Graph, GraphTensor, PreparedGraph, GraphLease


def prepare_graph(factory: Callable[[], Graph], constants: Mapping[str, Tensor], *, slots: int = 2) -> PreparedGraph:
    """Prepare a distinct actual ggml graph per slot, without executing it.

    ``factory()`` returns an unprepared graph with output/effect roots. Only
    constants named by that graph are imported, permitting one logical weight
    mapping to supply several programs. Every remaining named tensor is a
    required invocation binding. Output tensors belong to the runtime/caller.
    ``try_acquire()`` reserves a slot before resource admission; ``close()``
    releases an unsubmitted lease. Submission transfers it into native Pending.
    """
    return PreparedGraph(factory, dict(constants), slots)


__all__ = ["Graph", "GraphTensor", "PreparedGraph", "GraphLease", "prepare_graph"]
