"""Find a supported test witness in the current condition language.

This is an optimization-time consumer, not a serving dependency or a theorem
prover. Eq/In and integer Range constraints supply candidate values; a test
adapter can supply additional finite domains. Opaque predicates are evaluated
on those values or on bound facts, never guessed. Exhausting the search budget
or lacking an interpretable domain is UNRESOLVED, not a proof of unsatisfiability.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import chain, islice
from typing import Any

from mstar.engine.execution import Support

from mstar_accelerators.conditions import All, Condition, Eq, In, Or, Predicate, Range


class SolveStatus(str, Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class SolveResult:
    status: SolveStatus
    facts: Mapping[str, Any]
    reason: str
    attempts: int = 0


@dataclass(frozen=True)
class SweepResult:
    branches: tuple[SolveResult, ...]
    complete: bool


def solve_when(condition: Condition, facts: Mapping[str, Any] | None = None,
               domains: Mapping[str, Sequence[Any]] | None = None, *,
               max_trials: int = 1024) -> SolveResult:
    """Find one witness, retaining every supplied non-None fact.

    UNSAT is relative to the supplied facts and finite adapter domains. A SAT
    witness always passes the original condition's evaluate method and assigns
    every supplied case domain. Domains are reusable finite sequences (including
    ranges), not single-use iterators. The adapter must materialize and check
    the corresponding actual invocation rather than treating facts as inputs.
    """
    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    bound = dict(facts or {})
    supplied = dict(domains or {})
    for name, values in supplied.items():
        if bound.get(name) is not None and bound[name] not in values:
            return SolveResult(SolveStatus.UNSAT, bound,
                               f"the supplied {name} fact is outside its realizable domain")
    initial = condition.evaluate(bound)
    if initial is Support.YES and all(bound.get(name) is not None for name in supplied):
        return SolveResult(SolveStatus.SAT, bound, "the supplied facts satisfy when")
    if initial is Support.NO:
        return SolveResult(SolveStatus.UNSAT, bound, "when rejects the supplied facts")

    leaves = []

    def flatten(node):
        if isinstance(node, (All, Or)):
            for child in node.conditions:
                flatten(child)
        else:
            leaves.append(node)

    flatten(condition)
    constraints: dict[str, list[Predicate]] = {}
    for leaf in leaves:
        if isinstance(leaf, Predicate) and bound.get(leaf.fact) is None:
            constraints.setdefault(leaf.fact, []).append(leaf)

    def inferred_domains(node):
        """Finite supersets of each variable's satisfying values.

        A conjunct can restrict a variable alone. A disjunction bounds it only
        when every alternative does. Correlations remain in evaluate(), rather
        than being flattened into unrelated membership tests.
        """
        if isinstance(node, Eq):
            return {node.fact: ((node.value,),)}
        if isinstance(node, In):
            return {node.fact: (node.values,)}
        if isinstance(node, Range):
            return {node.fact: (range(node.minimum, node.maximum + 1),)}
        if isinstance(node, All):
            result = {}
            for child in node.conditions:
                for name, domain in inferred_domains(child).items():
                    result.setdefault(name, domain)
            return result
        if isinstance(node, Or) and node.conditions:
            children = [inferred_domains(child) for child in node.conditions]
            common = set(children[0]).intersection(*(set(child) for child in children[1:]))
            return {name: tuple(part for child in children for part in child[name])
                    for name in children[0] if name in common}
        return {}

    inferred = inferred_domains(condition)
    choices = {}
    missing = []
    for name in dict.fromkeys((*constraints, *supplied)):
        if bound.get(name) is not None:
            continue
        if name in supplied:
            # An adapter's domain is a restriction to cases it can construct.
            values = (supplied[name],)
        else:
            if name not in inferred:
                missing.append(name)
                continue
            values = inferred[name]
        # Range stays lazy: a huge declared shape domain must not be allocated
        # merely to find its first supported witness. Filtering belongs in the
        # bounded search so opaque predicate work is bounded as well.
        choices[name] = values

    names = tuple(choices)
    attempts = 0
    exhausted_budget = False
    saw_unknown = False
    witness = None

    def search(index):
        nonlocal attempts, exhausted_budget, saw_unknown, witness
        result = condition.evaluate(bound)
        if result is Support.NO:
            return False
        if index == len(names):
            if result is Support.YES:
                witness = dict(bound)
                return True
            saw_unknown = True
            return False
        name = names[index]
        old_present, old_value = name in bound, bound.get(name)
        for value in chain.from_iterable(choices[name]):
            if attempts >= max_trials:
                exhausted_budget = True
                break
            attempts += 1
            bound[name] = value
            if search(index + 1):
                return True
            if exhausted_budget:
                break
        if old_present:
            bound[name] = old_value
        else:
            bound.pop(name, None)
        return False

    if search(0):
        return SolveResult(SolveStatus.SAT, witness, "a supported witness was found", attempts)
    if exhausted_budget:
        return SolveResult(SolveStatus.UNRESOLVED, dict(facts or {}),
                           f"search budget of {max_trials} candidate assignments exhausted", attempts)
    if saw_unknown:
        detail = f"; missing domains for {', '.join(missing)}" if missing else ""
        return SolveResult(SolveStatus.UNRESOLVED, dict(facts or {}),
                           "when remains unknown after candidate evaluation" + detail, attempts)
    return SolveResult(SolveStatus.UNSAT, dict(facts or {}),
                       "no supported witness exists in the supplied domain", attempts)


def solve_alternatives(condition: Condition, facts: Mapping[str, Any] | None = None,
                       domains: Mapping[str, Sequence[Any]] | None = None, *,
                       constraints: Condition | None = None,
                       max_trials: int = 1024, max_branches: int = 32) -> SweepResult:
    """Find one witness per declared conjunction alternative, within bounds.

    Only All/Or are expanded. Other conditions remain opaque to this traversal;
    solve_when interprets each resulting conjunction as usual. Optional adapter
    constraints restrict each witness without adding declaration alternatives.
    Every supplied fact and finite domain retains solve_when's meaning.

    Branches preserve declaration order, including UNSAT and UNRESOLVED results.
    UNSAT remains relative to the supplied facts, domains and constraints.
    ``complete`` means every alternative was visited, not that every branch was
    satisfiable or resolved, nor that all invocations were tested. max_trials is
    a per-branch search budget. Expansion is lazy and visits at most max_branches
    alternatives plus one unsolved lookahead to detect truncation.
    """
    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    if max_branches < 1:
        raise ValueError("max_branches must be positive")

    def alternatives(node):
        if isinstance(node, Or):
            for child in node.conditions:
                yield from alternatives(child)
        elif isinstance(node, All):
            def combine(index, prefix):
                if index == len(node.conditions):
                    yield prefix
                    return
                for suffix in alternatives(node.conditions[index]):
                    yield from combine(index + 1, prefix + suffix)

            yield from combine(0, ())
        else:
            yield (node,)

    clauses = alternatives(condition)
    branches = []
    for clause in islice(clauses, max_branches):
        required = All(*clause) if constraints is None else All(*clause, constraints)
        branches.append(solve_when(required, facts, domains, max_trials=max_trials))
    return SweepResult(tuple(branches), complete=next(clauses, None) is None)
