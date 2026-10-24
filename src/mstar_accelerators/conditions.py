"""Small, partially evaluable conditions. Unknown facts remain unknown.

New predicates extend this producer/consumer boundary by implementing evaluate
and specializing their own facts.  The initial predicates are enough for the
first deployment; this is not a theorem prover or a qualification requirement.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice, product
from typing import Any, Protocol

from mstar.engine.execution import Support


class Condition(Protocol):
    def evaluate(self, facts: Mapping[str, Any]) -> Support: ...
    def specialize(self, facts: Mapping[str, Any]) -> "Condition": ...


@dataclass(frozen=True)
class Constant:
    value: Support

    def evaluate(self, facts):
        return self.value

    def specialize(self, facts):
        return self


class Predicate:
    fact: str

    def evaluate(self, facts):
        if self.fact not in facts or facts[self.fact] is None:
            return Support.UNKNOWN
        return Support.YES if self.matches(facts[self.fact]) else Support.NO

    def specialize(self, facts):
        result = self.evaluate(facts)
        return self if result is Support.UNKNOWN else Constant(result)


@dataclass(frozen=True)
class Eq(Predicate):
    fact: str
    value: Any

    def matches(self, value):
        return value == self.value


@dataclass(frozen=True)
class In(Predicate):
    fact: str
    values: tuple[Any, ...]

    def matches(self, value):
        return value in self.values


@dataclass(frozen=True)
class Range(Predicate):
    fact: str
    minimum: int
    maximum: int

    def matches(self, value):
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True, init=False)
class All:
    conditions: tuple[Condition, ...]

    def __init__(self, *conditions):
        object.__setattr__(self, "conditions", tuple(conditions))

    def evaluate(self, facts):
        results = tuple(condition.evaluate(facts) for condition in self.conditions)
        if Support.NO in results:
            return Support.NO
        return Support.UNKNOWN if Support.UNKNOWN in results else Support.YES

    def specialize(self, facts):
        residual = []
        for condition in self.conditions:
            value = condition.specialize(facts)
            if isinstance(value, Constant):
                if value.value is Support.NO:
                    return value
                if value.value is Support.YES:
                    continue
            residual.append(value)
        return All(*residual) if residual else Constant(Support.YES)


When = All


@dataclass(frozen=True, init=False)
class Or:
    conditions: tuple[Condition, ...]

    def __init__(self, *conditions):
        object.__setattr__(self, "conditions", tuple(conditions))

    def evaluate(self, facts):
        results = tuple(condition.evaluate(facts) for condition in self.conditions)
        if Support.YES in results:
            return Support.YES
        return Support.UNKNOWN if Support.UNKNOWN in results else Support.NO

    def specialize(self, facts):
        residual = []
        for condition in self.conditions:
            value = condition.specialize(facts)
            if isinstance(value, Constant):
                if value.value is Support.YES:
                    return value
                if value.value is Support.NO:
                    continue
            residual.append(value)
        return Or(*residual) if residual else Constant(Support.NO)


@dataclass(frozen=True)
class AllRows:
    """Apply a condition to each canonical row of one logical resource.

    The resource owner supplies these facts. This predicate never normalizes
    request policy, reads device state, or introduces another admission owner.
    """

    resource: str
    condition: Condition

    def evaluate(self, facts):
        rows = facts.get("resources", {}).get(self.resource)
        if rows is None:
            return Support.UNKNOWN
        results = tuple(self.condition.evaluate(row) for row in rows)
        if Support.NO in results:
            return Support.NO
        return Support.UNKNOWN if Support.UNKNOWN in results else Support.YES

    def specialize(self, facts):
        result = self.evaluate(facts)
        return self if result is Support.UNKNOWN else Constant(result)


def contradiction(claim, requirement, known):
    """Find a concrete contradiction in the small scalar predicate fragment.

    Candidate values come from predicate boundaries, not a qualification case
    domain. Every returned witness is evaluated against the complete claim and
    requirement. Opaque predicates (including row policies), a large search, or
    no witness leave analysis unresolved; callers must not infer compatibility.
    """
    domains = {}
    mentioned = set()

    def collect(condition):
        if isinstance(condition, Constant):
            return condition.value is not Support.UNKNOWN
        if isinstance(condition, (All, Or)):
            return all(collect(child) for child in condition.conditions)
        if not isinstance(condition, (Eq, In, Range)):
            return False
        mentioned.add(condition.fact)
        if condition.fact in known:
            return True
        if isinstance(condition, Eq):
            values = [condition.value]
        elif isinstance(condition, In):
            values = list(condition.values)
        else:
            values = [condition.minimum, condition.maximum]
        choices = domains.setdefault(condition.fact, [])
        for value in values:
            for candidate in ([value, value - 1, value + 1]
                              if isinstance(value, (int, float)) and not isinstance(value, bool)
                              else [value]):
                if candidate not in choices:
                    choices.append(candidate)
        return True

    if not collect(claim) or not collect(requirement):
        return None
    names = tuple(domains)
    for values in islice(product(*(domains[name] for name in names)), 256):
        facts = dict(known) | dict(zip(names, values))
        lengths = facts.get("token_lengths")
        if lengths is not None:
            facts["batch_size"] = len(lengths)
            facts["total_tokens"] = sum(lengths)
        if facts.get("batch_size", 1) < 1:
            continue
        if claim.evaluate(facts) is Support.YES and requirement.evaluate(facts) is Support.NO:
            return {key: value for key, value in facts.items()
                    if key in mentioned or key in ("graph_walk", "batch_size", "token_lengths", "total_tokens")}
    return None


def batch_bound(condition: Condition) -> int | None:
    """A sound upper bound, not a claim about downward-closed support."""
    if isinstance(condition, All):
        bounds = [bound for child in condition.conditions if (bound := batch_bound(child)) is not None]
        return min(bounds) if bounds else None
    if isinstance(condition, Or):
        bounds = [batch_bound(child) for child in condition.conditions]
        return max(bounds) if bounds and all(bound is not None for bound in bounds) else None
    if getattr(condition, "fact", None) == "batch_size":
        if isinstance(condition, Eq):
            return condition.value
        if isinstance(condition, In):
            return max(condition.values) if condition.values else 0
        if isinstance(condition, Range):
            return condition.maximum
    return None
