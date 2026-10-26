"""Automatic registry qualification: corpus × discovered execution environments."""
from __future__ import annotations

from ..sources import load_accelerations, normalize_source
from .harness import Result, Status, run_registry
from .numerics import DEFAULT
from .policy import identity


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
        status, reason = Status.SKIP, "no discovered execution environment produced evidence"
    return Result(name, status, reason, checks, {
        "scope": scope,
        "observations": {label: {key: value for key, value in result.items() if key != "checks"}
                         for label, result in observations.items()},
        "nonparticipating": sorted(set(observations) - {label for label, _ in required}),
    })


def run(registry_spec):
    from ..backends.discovery import discover
    from . import assets, corpus, hosting, planning
    from .harness import Unavailable

    registry_spec = normalize_source(registry_spec)
    entries = load_accelerations(registry_spec)
    environments = discover()
    plan = {'version': 1, 'environments': [env.receipt() for env in environments],
            'models': [], 'jobs': []}
    jobs, reports = [], {}
    for case in corpus.MODELS:
        try:
            model = case.describe()
            derived, description = planning.placements(case, model, entries, environments)
            plan['models'].append(description)
            jobs.extend(derived)
        except ModuleNotFoundError as exc:
            # Only genuinely absent optional packages are missing evidence;
            # missing internal modules and import bugs remain failures.
            plan['models'].append({'model': case.identity(), 'missing_dependency': str(exc)})
            status = Status.SKIP if exc.name in {'transformers', 'safetensors', 'huggingface_hub'} else Status.FAIL
            reports['description:' + case.key] = run_registry(entries, lambda name, acc: Result(
                name, status, str(exc), details={'code': 'reference_dependency'})).to_dict()
        except Exception as exc:
            reports['description:' + case.key] = run_registry(entries, lambda name, acc: Result(
                name, Status.FAIL, f'{type(exc).__name__}: {exc}',
                details={'code': 'invalid_corpus_description'})).to_dict()
    resolved = {}
    for job in jobs:
        receipt = job.receipt()
        plan['jobs'].append(receipt)
        subset = {name: entries[name] for name in job.entries}
        try:
            if (reason := job.unavailable_reason()) is not None:
                raise Unavailable('target_unavailable', reason)
            if job.case.key not in resolved:
                resolved[job.case.key] = assets.resolve(job.case)
            checkpoint, fixtures, asset_receipt = resolved[job.case.key]
            receipt['assets'] = asset_receipt
            reports[job.key] = hosting.execute(registry_spec, job, checkpoint, fixtures)
        except Unavailable as exc:
            receipt['unavailable'] = {'code': exc.code, 'reason': str(exc)}
            reports[job.key] = run_registry(subset, lambda name, acc: Result(
                name, Status.SKIP, str(exc), details={'code': exc.code, 'executed': False})).to_dict()
        except Exception as exc:
            receipt['error'] = f'{type(exc).__name__}: {exc}'
            reports[job.key] = run_registry(subset, lambda name, acc: Result(
                name, Status.FAIL, receipt['error'], details={'code': 'execution_or_integrity_failure'})).to_dict()
            # Do not initialize another job after uncertain native ownership.
            if job.environment.device_type != 'cpu':
                break
    attempted = set(reports)
    for job in jobs:
        if job.key not in attempted:
            reports[job.key] = run_registry({name: entries[name] for name in job.entries},
                lambda name, acc: Result(name, Status.SKIP, 'not executed after prior failure')).to_dict()
    rows = {label: {row['name']: row for row in report['accelerations']}
            for label, report in reports.items()}
    def consume(name, acceleration):
        observations = {label: values[name] for label, values in rows.items() if name in values}
        if not observations:
            matched = [item['model'] for item in plan['models']
                       if name in item.get('matched_entries', ())]
            return Result(name, Status.SKIP,
                'no supported placement in the central domain' if matched else
                'no corpus logical node matched this reference',
                details={'code': 'placement_unavailable' if matched else 'reference_unavailable',
                         'matched_models': matched, 'executed': False})
        return combine(name, observations, scope='platform corpus across discovered supported placements')

    report = run_registry(entries, consume).to_dict()
    report.update(policy=DEFAULT.name, qualification_policy=identity(),
                  execution_plan=plan, executions=reports)
    if any(result['status'] == 'FAIL' for result in reports.values()):
        report['status'] = 'FAIL'
    return report
