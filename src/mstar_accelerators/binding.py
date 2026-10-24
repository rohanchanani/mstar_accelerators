"""Binding facts are derived once; invocation facts are refined by the engine."""

from mstar.engine.execution import InvocationFacts
from dataclasses import fields, is_dataclass


def static_facts(binding):
    from mstar.platforms import current_platform

    facts = {
        "binding": binding,
        "node_name": binding.node_name,
        "device_type": binding.device.type,
        "platform": current_platform.name if binding.device.type != "cpu" else "cpu",
        "dtype": binding.dtype,
        "tp_size": binding.tp_group.world_size,
        "sp_size": binding.sp_group.world_size,
    }
    config = getattr(binding.submodule, "config", None)
    if is_dataclass(config):
        facts.update(("config." + field.name, getattr(config, field.name)) for field in fields(config))
    return facts


def invocation_facts(facts: InvocationFacts):
    return {
        "graph_walk": facts.graph_walk,
        "batch_size": facts.batch_size,
        "token_lengths": facts.token_lengths,
        "total_tokens": facts.total_tokens,
        "resources": facts.resource_facts,
        "row_walks": facts.row_walks,
    }
