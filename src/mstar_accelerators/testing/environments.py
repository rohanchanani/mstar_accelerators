"""Consume the complete corpus in configured execution environments.

Environment configuration supplies assets and placement, never a registration
or mode filter. Each environment uses the same differential consumer and model
adapter. A missing applicable environment remains missing evidence; no observed
failure can be hidden by a pass elsewhere.
"""
from __future__ import annotations

from mstar.engine.execution import Support

from ..sources import load_accelerations, normalize_source
from .harness import Result, Status, run_registry
from .numerics import DEFAULT


def inactive(result):
    details = result.get("details", {})
    if details.get("code") == "role_not_participating":
        return True
    cases = details.get("cases", ())
    return bool(cases) and not details.get("executed", False) and all(
        case.get("solver", {}).get("status") == "UNSAT" for case in cases)


def combine(name, observations, *, scope, irrelevant=()):
    """Aggregate role/environment observations without erasing failed checks."""
    required = [(label, result) for label, result in observations.items()
                if result["status"] != "SKIP"
                or (label not in irrelevant and not inactive(result))]
    checks = tuple(dict(check, observation_path=[label, *check.get("observation_path", ())])
                   for label, result in observations.items() for check in result.get("checks", ()))
    statuses = [result["status"] for _, result in required]
    if "FAIL" in statuses or any(not check["passed"] for check in checks):
        status, reason = Status.FAIL, "a participating execution failed"
    elif "SKIP" in statuses:
        status, reason = Status.SKIP, "applicable execution coverage is incomplete"
    elif statuses and all(value == "PASS" for value in statuses):
        status, reason = Status.PASS, "all participating executions passed"
    else:
        status, reason = Status.SKIP, "no configured execution environment produced evidence"
    return Result(name, status, reason, checks, {
        "scope": scope,
        "observations": {label: {key: value for key, value in result.items() if key != "checks"}
                         for label, result in observations.items()},
        "nonparticipating": sorted(set(observations) - {label for label, _ in required}),
    })


def run(registry_spec, options):
    from . import distributed, pi, qwen3_omni

    registry_spec = normalize_source(registry_spec)
    entries = load_accelerations(registry_spec)
    environments = options.get("environments", [{"kind": "pi"}])
    shared = {key: value for key, value in options.items() if key != "environments"}
    reports = {}
    targets = {}
    for index, environment in enumerate(environments):
        kind = environment["kind"]
        driver, device = {"pi": (pi.run, "mstar_pi"),
                          "cpu": (distributed.run, "cpu"),
                          "rocm": (qwen3_omni.run, "cuda")}[kind]
        label = f"{index}:{kind}"
        config = {**shared, **{key: value for key, value in environment.items() if key != "kind"}}
        reports[label] = driver(registry_spec, config)
        targets[label] = {"device_type": device}
        names = [row["name"] for row in reports[label]["accelerations"]]
        if names != list(entries):
            raise ValueError(f"{label}: environment changed the corpus inventory")

    rows = {label: {row["name"]: row for row in report["accelerations"]}
            for label, report in reports.items()}

    def consume(name, acceleration):
        return combine(name, {label: values[name] for label, values in rows.items()},
                       scope="whole corpus across configured execution environments",
                       irrelevant={label for label, facts in targets.items()
                                   if acceleration.when.evaluate(facts) is Support.NO})

    report = run_registry(entries, consume).to_dict()
    report.update(policy=DEFAULT.name, environments={
        label: {key: value for key, value in result.items() if key != "accelerations"}
        for label, result in reports.items()})
    # Startup failures can happen before any registration is active.
    if any(result["status"] == "FAIL" for result in reports.values()):
        report["status"] = "FAIL"
    return report
