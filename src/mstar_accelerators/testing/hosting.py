"""Execute an inferred placement through M*'s managed worker lifecycle.

This host has no model names, model constructors, reference policy, tolerances,
or case recipes. It transports a derived job to actual workers and preserves
release/uncertainty semantics. The central model consumer owns the cases.
"""
from datetime import timedelta
import json
import multiprocessing as mp
from pathlib import Path
import socket
import tempfile
import time

from ..sources import load_accelerations
from .harness import Result, Status, SuiteReport, Unavailable, run_registry
from . import policy


def _entries(source, job):
    entries = load_accelerations(source)
    return {name: entries[name] for name in job.entries}


def _write(path, report):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def _aborted(names, results, active, error):
    completed = {result.name: result for result in results}
    return SuiteReport(tuple(completed.get(name) or Result(
        name, Status.FAIL if name == active else Status.SKIP,
        error if name == active else 'not executed after worker failure',
        details={'scope': 'aborted qualification'}) for name in names)).to_dict() | {
            'status': 'FAIL', 'worker_error': error}


def groups_for(job, rank):
    from mstar.distributed.communication import CommGroup, WorkerParallelGroups
    members = list(range(job.tp_size))
    groups = WorkerParallelGroups(job.tp_size, rank, any_parallelism=job.tp_size > 1,
                                  world_parallel_groups=[tuple(members)] if job.tp_size > 1 else [])
    for node in job.nodes:
        groups.add(node, CommGroup(rank, rank, members))
    return groups


class Worker:
    def __init__(self, *, worker_id, device, runtime_extension, source, job,
                 checkpoint, fixtures, record, rank, rendezvous):
        self.extension, self.record = runtime_extension, record
        self.source, self.job, self.device = source, job, device
        self.checkpoint, self.fixtures = checkpoint, fixtures
        self.rank, self.rendezvous = rank, rendezvous
        self.session = None
        self.parallel_groups = groups_for(job, rank)
        self.stopped = False

    def request_stop(self):
        self.stopped = True

    def run(self):
        try:
            self._run()
        except BaseException:
            import traceback
            traceback.print_exc()
            self.parallel_groups.mark_failed()
            raise

    def _run(self):
        self.parallel_groups.init_dist(self.rendezvous, self.device, timeout=timedelta(seconds=180))
        entries = _entries(self.source, self.job)
        # The full selected corpus remains available to the centrally owned
        # trajectory constructor for compatible companion selection.
        corpus = load_accelerations(self.source)
        model = self.job.case.describe(self.checkpoint)
        control = self.parallel_groups.get_joint_group_for_node(self.job.nodes[0])
        inventories = control.all_gather_object(tuple(entries))
        if any(item != inventories[0] for item in inventories):
            raise ValueError('workers loaded different registry inventories')
        try:
            self.session, consumer, identity = policy.context(model,
                fixtures=self.fixtures, device=self.device, entries=corpus,
                stream=self.job.environment.stream(self.device), parallel_groups=self.parallel_groups,
                model_config={**self.job.case.deployment_config, 'runtime_extension': self.job.environment.runtime_options},
                model_identity=self.job.case.identity())
        except Unavailable as exc:
            self.record['report'] = run_registry(entries, lambda name, acceleration: Result(
                name, Status.SKIP, str(exc), details={'code': exc.code, 'executed': False})).to_dict()
            return

        def consume(name, acceleration):
            if self.stopped:
                raise InterruptedError('qualification stopped before next registration')
            self.record['active'] = name
            result = consumer(name, acceleration)
            self.extension.drain()
            self.session.close()
            finished = control.all_gather_object(name)
            if any(value != name for value in finished):
                raise ValueError('workers left registry entry out of order')
            self.record['results'].append(result)
            self.record['active'] = None
            return result

        self.record['report'] = run_registry(entries, consume).to_dict() | identity

    def close(self):
        self.extension.drain()
        if self.session is not None:
            self.session.close()
        self.parallel_groups.close()
        self.record['worker_closed'] = True


def _managed(source, job, checkpoint, fixtures, rank, rendezvous, owner, path):
    from mstar.runtime_extensions import run_managed_worker
    job.environment.prepare_process()
    extension = job.environment.runtime(source)
    record = {'names': job.entries, 'results': [], 'active': None, 'worker_closed': False}
    failure = None
    try:
        run_managed_worker(Worker, {
            'worker_id': f'qualification-{rank}', 'device': job.environment.device,
            'source': source, 'job': job, 'checkpoint': checkpoint, 'fixtures': fixtures,
            'record': record, 'rank': rank, 'rendezvous': rendezvous,
        }, extension, owner)
    except Exception as exc:
        if owner is not None and owner.status() != 'released':
            raise
        if not record['worker_closed']:
            raise
        failure = f'{type(exc).__name__}: {exc}'
    if owner is not None and owner.status() != 'released':
        raise RuntimeError('managed worker returned without a native release acknowledgement')
    report = record.get('report')
    if failure is not None or report is None:
        report = _aborted(job.entries, record['results'], record['active'],
                          failure or 'worker ended without producing a registry report')
    report.update(runtime_released=True, role=rank, pid=mp.current_process().pid)
    _write(path, report)


def execute(source, job, checkpoint, fixtures):
    from .environments import combine
    extension = job.environment.runtime(source)
    admission = extension.admission_context()
    owners, started = [], []
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        rendezvous = f'tcp://127.0.0.1:{listener.getsockname()[1]}'
    context = mp.get_context('spawn')
    with tempfile.TemporaryDirectory(prefix='mstar-qualification-') as directory:
        paths = [Path(directory) / f'role-{rank}.json' for rank in range(job.tp_size)]
        failure = None
        try:
            for rank, path in enumerate(paths):
                owner = extension.reserve_worker(admission, f'qualification-{rank}')
                owners.append(owner)
                process = context.Process(target=_managed,
                    args=(source, job, checkpoint, fixtures, rank, rendezvous, owner, path),
                    name=f'qualification-{rank}')
                process.start()
                started.append(process)
            # Only CPU process groups may be terminated on a failed peer. A
            # native device's managed holder must establish actual settlement.
            deadline = time.monotonic() + 7200
            while any(process.is_alive() for process in started):
                if job.environment.device_type == 'cpu' and (
                    any(process.exitcode not in (None, 0) for process in started)
                    or any(path.exists() and 'worker_error' in json.loads(path.read_text())
                           for path in paths)
                    or time.monotonic() >= deadline):
                    failure = 'CPU worker failed or exceeded the qualification deadline'
                    break
                for process in started:
                    process.join(timeout=0.1)
        except BaseException:
            if job.environment.device_type == 'cpu':
                failure = 'CPU controller failed while starting or waiting for roles'
            for owner in owners:
                if owner is not None and owner.status() != 'released':
                    owner.unknown('qualification controller interrupted without acknowledged release')
            raise
        finally:
            if failure is not None:
                for process in started:
                    if process.is_alive():
                        process.terminate()
                for process in started:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join()
        for owner in owners:
            if owner is not None and owner.status() != 'released':
                owner.unknown('qualification worker exited without acknowledged release')
                raise RuntimeError('qualification owner did not acknowledge release')
        reports = {}
        for rank, path in enumerate(paths):
            process = started[rank]
            reports[str(rank)] = json.loads(path.read_text()) if path.exists() else _aborted(
                job.entries, (), None, f'no worker report; exit={process.exitcode}; {failure}')
            if process.exitcode != 0:
                failure = failure or f'worker {rank} exited {process.exitcode}'
        observations = {rank: {row['name']: row for row in report['accelerations']}
                        for rank, report in reports.items()}
        report = run_registry(_entries(source, job), lambda name, acceleration: combine(
            name, {rank: rows[name] for rank, rows in observations.items()},
            scope='all participating runtime roles')).to_dict()
        report['roles'] = reports
        report['runtime_released'] = all(row.get('runtime_released', False) for row in reports.values())
        if failure or any(row['status'] == 'FAIL' for row in reports.values()):
            report.update(status='FAIL', worker_error=failure or 'worker reported failure')
    if admission is not None:
        admission.close()
    return report
