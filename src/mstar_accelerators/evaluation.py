"""Prepared evaluators for authored conditions, with live invocation values.

This is a derived assembly product, not another support declaration. Unknown
predicates retain their own evaluator. No fact or verdict is cached.
"""
from mstar.engine.execution import Support

from .conditions import All, AllRows, Constant, Eq, In, Or, Range


def prepare_evaluator(condition):
    kind = type(condition)
    if kind is Constant:
        value = condition.value
        return lambda _facts: value
    if kind in (Eq, In, Range):
        name, matches = condition.fact, condition.matches
        def predicate(facts):
            value = facts.get(name)
            if value is None:
                return Support.UNKNOWN
            return Support.YES if matches(value) else Support.NO
        return predicate
    if kind in (All, Or):
        # Flatten only the same associative operator. Retain evaluation order,
        # including custom predicates after a decisive earlier result: the
        # original implementation evaluates every child before combining.
        def children(node):
            if type(node) is kind:
                for child in node.conditions:
                    yield from children(child)
            else:
                yield prepare_evaluator(node)
        evaluators = tuple(children(condition))
        decisive = Support.NO if kind is All else Support.YES
        neutral = Support.YES if kind is All else Support.NO
        def combine(facts):
            outcome = neutral
            for evaluate in evaluators:
                value = evaluate(facts)
                if value is decisive:
                    outcome = decisive
                elif value is Support.UNKNOWN and outcome is neutral:
                    outcome = Support.UNKNOWN
            return outcome
        return combine
    if kind is AllRows:
        resource, evaluate = condition.resource, prepare_evaluator(condition.condition)
        def rows(facts):
            values = facts.get('resources', {}).get(resource)
            if values is None:
                return Support.UNKNOWN
            outcome = Support.YES
            for row in values:
                value = evaluate(row)
                if value is Support.NO:
                    outcome = Support.NO
                elif value is Support.UNKNOWN and outcome is Support.YES:
                    outcome = Support.UNKNOWN
            return outcome
        return rows
    return condition.evaluate
