"""Ordinary tensor representations of model-owned logical weights."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorShard:
    """Partition one logical dimension over a binding's communication group."""

    axis: int
    group: str = "tp"


@dataclass(frozen=True)
class _WeightRepresentation:
    prefix: str
    dtype: torch.dtype
    device: str
    shards: tuple


def prepare_weights(binding, prefix, *, dtype=None,
                    shards: Mapping[str, TensorShard] | None = None):
    """Load logical values once per immutable representation, preserving aliases.

    The model owns checkpoint interpretation. The implementation selects dtype
    and logical dimensions to partition; role and participant count come from
    the bound groups. No reference compute module is constructed.
    """
    dtype = binding.dtype if dtype is None else dtype
    shards = dict(shards or {})
    partitions = {}
    for name, shard in shards.items():
        if shard.group not in ("tp", "sp"):
            raise ValueError(f"unknown weight partition group {shard.group!r}")
        group = getattr(binding, shard.group + "_group")
        partitions[name] = (shard.axis, group.rank, group.world_size)
    key = _WeightRepresentation(prefix, dtype, str(binding.device),
                                tuple(sorted(partitions.items())))
    if key not in binding.prepared:
        source = binding.model.logical_weights()
        names = [spec.name for spec in source.specs if spec.name.startswith(prefix)]
        if unknown := shards.keys() - set(names):
            raise ValueError(f"weight partition names are not in {prefix!r}: {sorted(unknown)}")
        logical = source.load(names)
        converted, identities = {}, {}
        for name, tensor in logical.items():
            partition = partitions.get(name)
            identity = (id(tensor), partition)
            if identity not in identities:
                if partition is not None:
                    axis, rank, count = partition
                    if not -tensor.ndim <= axis < tensor.ndim:
                        raise ValueError(f"{name}: partition axis {axis} is out of range")
                    extent = tensor.shape[axis]
                    if extent % count:
                        raise ValueError(f"{name}: extent {extent} is not divisible by {count} roles")
                    width = extent // count
                    tensor = tensor.narrow(axis, rank * width, width)
                identities[identity] = tensor.to(dtype=dtype).contiguous().to(binding.device)
            converted[name] = identities[identity]
        binding.prepared[key] = converted
    return binding.prepared[key]
