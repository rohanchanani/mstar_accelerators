"""Witness soundness and honest limits of the optimization-time solver."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from mstar.engine.execution import Support

from mstar_accelerators.conditions import (
    All,
    Constant,
    Eq,
    In,
    Or,
    Predicate,
    Range,
    batch_bound,
)
from mstar_accelerators.testing.solve import SolveStatus, solve_alternatives, solve_when


def test_nested_constraints_find_a_supported_exact_two_witness():
    condition = All(Eq("device_type", "cpu"),
                    All(In("batch_size", (1, 2, 4)), Range("batch_size", 2, 3)),
                    Eq("token_lengths", (1, 1)))
    result = solve_when(condition, {"device_type": "cpu"})
    assert result.status is SolveStatus.SAT
    assert result.facts == {"device_type": "cpu", "batch_size": 2, "token_lengths": (1, 1)}
    assert condition.evaluate(result.facts) is Support.YES


def test_target_facts_are_never_replaced_to_make_a_witness():
    condition = All(Eq("device_type", "mstar_pi"), Eq("batch_size", 1))
    result = solve_when(condition, {"device_type": "cpu"}, {"device_type": ("mstar_pi",)})
    assert result.status is SolveStatus.UNSAT
    assert result.facts["device_type"] == "cpu"


def test_domains_restrict_cases_to_what_the_adapter_can_realize():
    condition = All(Range("tokens", 1, 8), Eq("tokens", 3))
    result = solve_when(condition, {}, {"tokens": (1, 2, 4)})
    assert result.status is SolveStatus.UNSAT
    result = solve_when(condition, {}, {"tokens": (2, 3, 4)})
    assert result.status is SolveStatus.SAT and result.facts["tokens"] == 3
    result = solve_when(condition, {"tokens": 3}, {"tokens": (1, 2, 4)})
    assert result.status is SolveStatus.UNSAT


def test_contradictory_constraints_and_empty_domains_are_unsat():
    assert solve_when(All(Eq("x", 1), Eq("x", 2))).status is SolveStatus.UNSAT
    assert solve_when(Range("x", 9, 2)).status is SolveStatus.UNSAT
    assert solve_when(In("x", ())).status is SolveStatus.UNSAT
    assert solve_when(Eq("x", 1), domains={"x": ()}).status is SolveStatus.UNSAT


@dataclass(frozen=True)
class Geometry(Predicate):
    fact: str = "binding"

    def matches(self, value):
        return value.width == 384


def test_model_owned_opaque_fact_is_evaluated_not_invented():
    condition = All(Geometry(), Eq("batch_size", 1))
    result = solve_when(condition)
    assert result.status is SolveStatus.UNRESOLVED
    assert "binding" in result.reason
    result = solve_when(condition, {"binding": SimpleNamespace(width=384)})
    assert result.status is SolveStatus.SAT
    assert condition.evaluate(result.facts) is Support.YES
    result = solve_when(condition, {"binding": SimpleNamespace(width=768)})
    assert result.status is SolveStatus.UNSAT


@dataclass(frozen=True)
class Odd(Predicate):
    fact: str = "x"

    def matches(self, value):
        return value % 2 == 1


def test_range_generation_can_find_a_later_opaque_match():
    condition = All(Range("x", 2, 9), Odd())
    result = solve_when(condition)
    assert result.status is SolveStatus.SAT and result.facts["x"] == 3


class SumFive:
    def evaluate(self, facts):
        if "x" not in facts or "y" not in facts:
            return Support.UNKNOWN
        return Support.YES if facts["x"] + facts["y"] == 5 else Support.NO

    def specialize(self, facts):
        return self


def test_adapter_domains_can_realize_a_new_cross_fact_predicate():
    result = solve_when(SumFive(), domains={"x": (1, 2), "y": (2, 3)})
    assert result.status is SolveStatus.SAT
    assert result.facts == {"x": 2, "y": 3}
    assert SumFive().evaluate(result.facts) is Support.YES
    assert solve_when(SumFive()).status is SolveStatus.UNRESOLVED


def test_large_range_is_lazy_and_budget_exhaustion_is_not_unsat():
    class AtEnd(Predicate):
        fact = "x"

        def matches(self, value):
            return value == 10**18

    result = solve_when(All(Range("x", 1, 10**18), AtEnd()), max_trials=2)
    assert result.status is SolveStatus.UNRESOLVED
    assert result.attempts == 2
    assert "budget" in result.reason
    supplied = solve_when(AtEnd(), domains={"x": range(1, 10**18 + 1)}, max_trials=2)
    assert supplied.status is SolveStatus.UNRESOLVED
    assert supplied.attempts == 2


def test_constants_keep_their_three_valued_meaning():
    assert solve_when(Constant(Support.YES)).status is SolveStatus.SAT
    assert solve_when(Constant(Support.NO)).status is SolveStatus.UNSAT
    assert solve_when(Constant(Support.UNKNOWN)).status is SolveStatus.UNRESOLVED


def test_all_supplied_case_domains_are_materialized_even_after_when_is_true():
    result = solve_when(Constant(Support.YES), domains={"n": (3,)})
    assert result.status is SolveStatus.SAT and result.facts == {"n": 3}
    result = solve_when(Constant(Support.YES), domains={"n": ()})
    assert result.status is SolveStatus.UNSAT
    result = solve_when(Eq("x", 1), domains={"x": (1,), "n": (3,)})
    assert result.status is SolveStatus.SAT and result.facts == {"x": 1, "n": 3}
    result = solve_when(Eq("x", 1), domains={"x": (1,), "n": ()})
    assert result.status is SolveStatus.UNSAT


def test_none_is_unresolved_and_input_mapping_is_not_mutated():
    facts = {"batch_size": None, "device_type": "cpu"}
    result = solve_when(Eq("batch_size", 1), facts)
    assert result.status is SolveStatus.SAT
    assert result.facts["batch_size"] == 1
    assert facts["batch_size"] is None


def test_walk_shape_disjunction_preserves_correlations_in_serving_and_solving():
    condition = All(Eq("batch_size", 1),
                    Or(All(Eq("graph_walk", "prefill"), Eq("token_lengths", (4,))),
                       All(Eq("graph_walk", "decode"), Eq("token_lengths", (1,)))))
    assert condition.evaluate({"batch_size": 1, "graph_walk": "decode"}) is Support.UNKNOWN
    assert condition.evaluate({"batch_size": 1, "graph_walk": "decode", "token_lengths": (4,)}) is Support.NO
    for walk, lengths in (("prefill", (4,)), ("decode", (1,))):
        solved = solve_when(condition, {"graph_walk": walk})
        assert solved.status is SolveStatus.SAT
        assert solved.facts["token_lengths"] == lengths
        assert condition.evaluate(solved.facts) is Support.YES
    assert batch_bound(condition) == 1


def test_alternative_with_unbounded_shape_does_not_produce_a_false_unsat():
    condition = Or(All(Eq("mode", "a"), Eq("x", 1)), Eq("mode", "b"))
    solved = solve_when(condition, {"mode": "b"})
    assert solved.status is SolveStatus.SAT
    assert batch_bound(Or(Eq("batch_size", 2), Eq("graph_walk", "prefill"))) is None
    assert batch_bound(Or(Eq("batch_size", 2), Range("batch_size", 4, 8))) == 8


def test_alternative_sweep_preserves_correlated_modes_and_all_case_domains():
    condition = All(Eq("device_type", "cpu"),
                    Or(All(Eq("mode", "prefill"), Eq("tokens", 4)),
                       All(Eq("mode", "decode"), Eq("tokens", 1))))
    facts = {"device_type": "cpu"}
    domains = {"tokens": (1, 4), "sample": (17,)}
    sweep = solve_alternatives(condition, facts, domains)
    assert sweep.complete
    assert [branch.status for branch in sweep.branches] == [SolveStatus.SAT] * 2
    assert [(branch.facts["mode"], branch.facts["tokens"])
            for branch in sweep.branches] == [("prefill", 4), ("decode", 1)]
    for branch in sweep.branches:
        assert condition.evaluate(branch.facts) is Support.YES
        assert all(branch.facts[name] in values for name, values in domains.items())
        assert branch.facts["device_type"] == "cpu"
    assert facts == {"device_type": "cpu"}


def test_nested_all_or_visits_four_clauses_with_two_satisfiable():
    condition = All(Or(Eq("x", 1), Eq("x", 2)),
                    All(Or(All(Eq("y", 1), Eq("x", 1)),
                           All(Eq("y", 2), Eq("x", 2)))))
    sweep = solve_alternatives(condition)
    assert sweep.complete
    assert [branch.status for branch in sweep.branches] == [
        SolveStatus.SAT, SolveStatus.UNSAT, SolveStatus.UNSAT, SolveStatus.SAT,
    ]
    assert [dict(branch.facts) for branch in sweep.branches
            if branch.status is SolveStatus.SAT] == [{"x": 1, "y": 1}, {"x": 2, "y": 2}]
    for branch in sweep.branches:
        if branch.status is SolveStatus.SAT:
            assert condition.evaluate(branch.facts) is Support.YES


def test_alternative_sweep_is_lazy_and_lookahead_is_not_solved(monkeypatch):
    from mstar_accelerators.testing import solve

    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return solve_when(*args, **kwargs)

    monkeypatch.setattr(solve, "solve_when", counted)
    condition = All(*(Or(Eq(f"bit_{i}", 0), Eq(f"bit_{i}", 1)) for i in range(40)))
    sweep = solve_alternatives(condition, max_branches=2)
    assert not sweep.complete
    assert len(sweep.branches) == len(calls) == 2
    assert [branch.status for branch in sweep.branches] == [SolveStatus.SAT] * 2
    assert [branch.facts["bit_39"] for branch in sweep.branches] == [0, 1]
    assert all(branch.facts[f"bit_{i}"] == 0 for branch in sweep.branches for i in range(39))


def test_alternative_sweep_marks_exact_bound_complete_and_validates_budgets():
    condition = Or(Eq("x", 1), Eq("x", 2))
    assert solve_alternatives(condition, max_branches=2).complete
    assert not solve_alternatives(condition, max_branches=1).complete
    for invalid in (0, -1):
        with pytest.raises(ValueError, match="max_branches must be positive"):
            solve_alternatives(condition, max_branches=invalid)
        with pytest.raises(ValueError, match="max_trials must be positive"):
            solve_alternatives(Or(), max_trials=invalid)


def test_alternative_sweep_leaves_opaque_conditions_unresolved_without_domains():
    condition = Or(All(Eq("mode", "a"), Geometry()), SumFive())
    sweep = solve_alternatives(condition)
    assert sweep.complete
    assert [branch.status for branch in sweep.branches] == [SolveStatus.UNRESOLVED] * 2
    assert "binding" in sweep.branches[0].reason
    assert all("binding" not in branch.facts for branch in sweep.branches)


def test_adapter_constraints_restrict_each_declared_branch_without_becoming_coverage():
    condition = Or(All(Eq("mode", "a"), Eq("tokens", 4)),
                   All(Eq("mode", "b"), Eq("tokens", 1)))
    constraints = All(Eq("mode", "b"), Or(Eq("sample", 3), Eq("sample", 7)))
    domains = {"tokens": (1, 4), "sample": (7,), "extra": (11,)}
    sweep = solve_alternatives(condition, domains=domains, constraints=constraints)
    assert sweep.complete
    assert [branch.status for branch in sweep.branches] == [SolveStatus.UNSAT, SolveStatus.SAT]
    witness = sweep.branches[1].facts
    assert condition.evaluate(witness) is Support.YES
    assert constraints.evaluate(witness) is Support.YES
    assert all(witness[name] in values for name, values in domains.items())


def test_alternative_sweep_retains_static_rejection_and_domain_relative_unsat():
    condition = All(Eq("device_type", "mstar_pi"), Or(Eq("x", 1), Eq("x", 2)))
    sweep = solve_alternatives(condition, {"device_type": "cpu"})
    assert sweep.complete
    assert [branch.status for branch in sweep.branches] == [SolveStatus.UNSAT] * 2
    assert all(branch.attempts == 0 and branch.facts["device_type"] == "cpu"
               for branch in sweep.branches)
    restricted = solve_alternatives(Or(Eq("x", 1), Eq("x", 2)), domains={"x": (1,)})
    assert restricted.complete
    assert [branch.status for branch in restricted.branches] == [SolveStatus.SAT, SolveStatus.UNSAT]


def test_alternative_sweep_keeps_branch_search_exhaustion_unresolved():
    class AtEnd(Predicate):
        fact = "x"

        def matches(self, value):
            return value == 10**18

    condition = Or(All(Range("x", 1, 10**18), AtEnd()), Eq("x", 2))
    sweep = solve_alternatives(condition, max_trials=2)
    assert sweep.complete
    assert [branch.status for branch in sweep.branches] == [SolveStatus.UNRESOLVED, SolveStatus.SAT]
    assert sweep.branches[0].attempts == 2
    assert sweep.branches[1].facts["x"] == 2


def test_empty_conjunction_has_one_alternative_and_empty_disjunction_has_none():
    conjunction = solve_alternatives(All(), domains={"x": (1,)})
    assert conjunction.complete
    assert len(conjunction.branches) == 1
    assert conjunction.branches[0].status is SolveStatus.SAT
    assert conjunction.branches[0].facts == {"x": 1}
    disjunction = solve_alternatives(Or())
    assert disjunction.complete and disjunction.branches == ()
