"""Qwen qualification using the existing managed Torch worker lifecycle.

Every registration is consumed. Missing hardware/assets remain explicit SKIPs;
submitted numerical failures remain FAILs. The actual acceleration is never
replaced by the portable correctness oracle.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import tempfile

import torch

from ..sources import load_accelerations, normalize_source
from .harness import DifferentialConsumer, Result, Status, SuiteReport, run_registry
from .numerics import DEFAULT, differential_comparator


def _skip(entries, code, reason):
    return run_registry(entries, lambda name, acceleration: Result(
        name, Status.SKIP, reason, details={'code': code, 'scope': 'not executed', 'executed': False},
    )).to_dict() | {'policy': DEFAULT.name, 'target_execution': False}


class _Worker:
    def __init__(self, *, worker_id, device, runtime_extension, registry_spec, options, record):
        del worker_id
        from .models.qwen3_omni import QualificationSession
        self.extension = runtime_extension
        self.device, self.registry_spec, self.options, self.record = device, registry_spec, options, record
        self.session = QualificationSession()
        self.stopped = False
        torch.set_num_threads(options.get('threads', 4))

    def request_stop(self):
        self.stopped = True

    def run(self):
        from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
        from .models.qwen3_omni import adapters
        entries = load_accelerations(self.registry_spec)
        model = Qwen3OmniModel.describe(model_path_hf=self.options['checkpoint'])
        references = adapters(model, session=self.session, device=self.device, options=self.options)
        consumer = DifferentialConsumer(references, differential_comparator)

        def consume(name, acceleration):
            if self.stopped:
                raise InterruptedError('qualification stopped before the next declaration')
            self.record['active'] = name
            result = consumer(name, acceleration)
            self.extension.drain()
            self.session.close()
            self.record['results'].append(result)
            self.record['active'] = None
            return result

        report = run_registry(entries, consume).to_dict()
        report.update(policy=DEFAULT.name, checkpoint=str(Path(self.options['checkpoint']).resolve()),
                      device=str(self.device), torch=torch.__version__, hip=torch.version.hip,
                      scope='actual registered programs; bounded model-adapter trajectories and references',
                      target_execution=any(row.details.get('executed', False) for row in self.record['results']))
        self.record['report'] = report

    def close(self):
        self.extension.drain()
        self.session.close()
        self.record['released'] = True


def _managed(registry_spec, options, report_path):
    from mstar.runtime_extensions import run_managed_worker
    from mstar_accelerators.backends.torch.runtime import TorchRuntime
    device = torch.device(options.get('device', 'cuda:0'))
    if device.type == 'cuda':
        from mstar_accelerators.backends.torch.device_runtime import DeviceTorchRuntime
        extension = DeviceTorchRuntime({'accelerations': registry_spec})
    else:
        extension = TorchRuntime({'accelerations': registry_spec, 'threads': options.get('threads', 4)})
    entries = load_accelerations(registry_spec)
    record = {'results': [], 'active': None, 'released': False}
    failure = None
    try:
        run_managed_worker(_Worker, {
            'worker_id': 'qwen-registry-qualification', 'device': str(device),
            'registry_spec': registry_spec, 'options': options, 'record': record,
        }, extension, None)
    except Exception as exc:
        if not record['released']:
            raise  # no successful drain/close receipt means no publishable verdict
        failure = f'{type(exc).__name__}: {exc}'
    report = record.get('report')
    if failure is not None or report is None:
        failure = failure or 'qualification ended without a report'
        results = {result.name: result for result in record['results']}
        rows = []
        for name in entries:
            if name in results:
                rows.append(results[name])
            else:
                active = name == record['active']
                rows.append(Result(name, Status.FAIL if active else Status.SKIP,
                                   failure if active else 'not executed after driver failure',
                                   details={'scope': 'aborted qualification'}))
        report = SuiteReport(tuple(rows)).to_dict() | {'status': 'FAIL', 'driver_error': failure,
                                                     'policy': DEFAULT.name}
    report['runtime_released'] = True
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')


def run(registry_spec, options):
    registry_spec = normalize_source(registry_spec)
    entries = load_accelerations(registry_spec)
    device = torch.device(options.get('device', 'cuda:0'))
    if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.version.hip):
        return _skip(entries, 'target_unavailable', 'the ROCm qualification environment needs an available AMD GPU')
    if device.type not in ('cpu', 'cuda'):
        return _skip(entries, 'target_unavailable', f'this driver does not provide {device.type} placement')
    checkpoint = options.get('checkpoint')
    if not checkpoint or not (Path(checkpoint) / 'config.json').is_file():
        return _skip(entries, 'reference_assets_unavailable', 'an available local Qwen checkpoint is required')
    if not any(Path(checkpoint).glob('*.safetensors')):
        return _skip(entries, 'reference_assets_unavailable', 'checkpoint safetensors are missing')
    batches = options.get('batches', [1, 2, 8])
    if not batches or any(type(x) is not int or x < 1 for x in batches) or len(set(batches)) != len(batches):
        raise ValueError('batches must be distinct positive integers')
    if options.get('decode_steps', 3) < 3:
        raise ValueError('decode_steps must cover text, EOS and padding (at least three)')
    if options.get('threads', 4) < 1:
        raise ValueError('threads must be positive')
    with tempfile.TemporaryDirectory(prefix='qwen-qualification-') as directory:
        report_path = str(Path(directory) / 'report.json')
        if options.get('in_process', False):
            _managed(registry_spec, options, report_path)
        else:
            process = mp.get_context('spawn').Process(target=_managed,
                args=(registry_spec, options, report_path), name='qwen-registry-qualification')
            process.start()
            process.join()
            if process.exitcode != 0 or not Path(report_path).is_file():
                raise RuntimeError(f'qualification child failed without a released report (exit={process.exitcode})')
        return json.loads(Path(report_path).read_text())
