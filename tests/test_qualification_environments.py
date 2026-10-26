"""Coverage aggregation controls; these produce no hardware evidence."""
import pytest

from mstar_accelerators import Acceleration, Eq, Implementation, When
from mstar_accelerators.testing.environments import combine, run
from mstar_accelerators.testing.harness import Result, Status, SuiteReport


def row(name, status, *, inactive=False):
    return Result(name, status, "synthetic aggregation control",
                  ({"name": "observed", "passed": status is Status.PASS},)
                  if status is not Status.SKIP else (),
                  {"code": "role_not_participating"} if inactive else {}).to_dict()


@pytest.mark.parametrize("other,expected", [(Status.FAIL, Status.FAIL), (Status.SKIP, Status.SKIP)])
def test_one_pass_cannot_hide_peer_failure_or_missing_evidence(other, expected):
    result = combine("decoder", {"0": row("decoder", Status.PASS),
                                 "1": row("decoder", other)}, scope="control")
    assert result.status is expected
    assert set(result.details["observations"]) == {"0", "1"}


def test_only_participating_roles_are_required_and_empty_evidence_stays_skip():
    result = combine("encoder", {"0": row("encoder", Status.PASS),
                                 "1": row("encoder", Status.SKIP, inactive=True)}, scope="control")
    assert result.status is Status.PASS
    assert result.details["nonparticipating"] == ["1"]
    assert combine("absent", {}, scope="control").status is Status.SKIP
