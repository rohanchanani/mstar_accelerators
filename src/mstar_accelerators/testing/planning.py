"""Derive execution placements from logical bindings and discovered hardware.

No user plan is accepted. Names label receipts only; reference types and when
are the only acceleration information used to decide where cases can execute.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from mstar.engine.execution import Support
from mstar.model.binding import NodeBinding
from mstar_accelerators.binding import static_facts
from ..backends.discovery import Environment
from .corpus import ModelCase, logical_nodes
from .harness import _fact_description
from .policy import TP_SIZES, dtype_for


@dataclass(frozen=True)
class Job:
    case: ModelCase
    environment: Environment
    tp_size: int
    nodes: tuple[str, ...]
    entries: tuple[str, ...]
    bindings: tuple[dict, ...]

    @property
    def key(self):
        return f'{self.case.key}/{self.environment.key}/tp{self.tp_size}'

    def unavailable_reason(self):
        if not self.environment.available:
            return self.environment.reason
        if self.tp_size > self.environment.max_ranks:
            return 'requested placement exceeds discovered rank capacity'
        if self.tp_size > 1 and self.environment.collective is None:
            return 'no collective backend for the requested placement'
        return None

    def receipt(self):
        return {'id': self.key, **self.case.identity(),
                'environment': self.environment.receipt(), 'tp_size': self.tp_size,
                'sp_size': 1, 'nodes': list(self.nodes), 'entries': list(self.entries),
                'bindings': list(self.bindings), 'unavailable': self.unavailable_reason()}


def placements(case, model, entries, environments):
    nodes, missing_shells = logical_nodes(model)
    matched = {name: [(node, shell, walks) for node, (shell, walks) in nodes.items()
                      if type(shell) is acceleration.reference]
               for name, acceleration in entries.items()}
    jobs = []
    for env in environments:
        for tp in TP_SIZES:
            records, selected, hosted = [], [], set()
            for name, acceleration in entries.items():
                for node, shell, walks in matched[name]:
                    group = SimpleNamespace(world_size=tp, rank=0)
                    binding = NodeBinding(model, shell, node,
                        torch.device(env.device) if env.available else torch.device('cpu'),
                        dtype_for(model), group,
                        SimpleNamespace(world_size=1, rank=0), {}, model.get_node_resources(),
                        graph_walks=walks)
                    facts = static_facts(binding) | env.facts()
                    if not env.available:
                        # An unavailable device has no real binding. Preserve
                        # partial facts; predicates needing binding stay unknown.
                        facts.pop('binding')
                    support = acceleration.when.specialize(facts).evaluate(facts)
                    if support is Support.NO:
                        continue
                    selected.append(name)
                    hosted.add(node)
                    records.append({'entry': name, 'node': node, 'walks': list(walks),
                                    'static_support': support.name,
                                    'facts': _fact_description({k: v for k, v in facts.items()
                                                                if k != 'binding'})})
            if selected:
                jobs.append(Job(case, env, tp, tuple(sorted(hosted)),
                                tuple(dict.fromkeys(selected)), tuple(records)))
    return jobs, {'model': case.identity(), 'logical_nodes': {
        node: {'reference': f'{type(shell).__module__}.{type(shell).__qualname__}',
               'walks': list(walks)} for node, (shell, walks) in nodes.items()},
        'missing_logical_shells': missing_shells,
        'matched_entries': [name for name, matches in matched.items() if matches]}
