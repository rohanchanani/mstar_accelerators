"""Ownership and inference boundaries. Synthetic cases are mechanism evidence."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mstar.engine.execution import Support
from mstar_accelerators import Acceleration, Eq, Implementation, When
from mstar_accelerators.backends.discovery import Environment
from mstar_accelerators.testing import assets, corpus, environments, hosting, planning
from mstar_accelerators.testing.harness import Result, Status, Unavailable, run_registry


class Logical:
    config = None


class DifferentLogical:
    config = None


class Graph:
    def get_nodes(self):
        return ['node']


class Model:
    def get_graph_walk_graphs(self):
        return {'walk': Graph()}

    def create_submodule(self, node):
        return Logical()

    def get_node_resources(self):
        return []

    def get_autocast_dtype(self):
        return torch.float32

    config = None


@dataclass(frozen=True)
class Case:
    key: str = 'model'

    def identity(self):
        return {'model': self.key, 'revision': 'pinned'}

    def describe(self):
        return Model()


def env(device='cpu', available=True):
    return Environment(device, device, device, device, 2, 'gloo', 'unused:Runtime', available)


def entry(device='cpu', reference=Logical):
    def forbidden(*args, **kwargs):
        raise AssertionError('inference must not prepare physical implementation')
    return Acceleration(reference, When(Eq('device_type', device), Eq('tp_size', 1),
                                       Eq('graph_walk', 'walk')), Implementation(forbidden))


def test_planning_uses_reference_and_when_not_name_filename_or_implementation():
    entries = {'invented/new-backend-looking-name': entry(), 'renamed-copy': entry(),
               'wrong-reference': entry(reference=DifferentLogical)}
    jobs, description = planning.placements(Case(), Model(), entries, [env()])
    assert len(jobs) == 1
    assert jobs[0].entries == tuple(entries)[:2]
    assert jobs[0].nodes == ('node',) and jobs[0].tp_size == 1
    assert description['matched_entries'] == list(entries)[:2]
    assert jobs[0].bindings[0]['facts']['dtype'] == 'torch.float32'


def test_backend_and_model_are_independent_axes():
    entries = {'a': entry('cpu'), 'b': entry('new_device')}
    jobs, _ = planning.placements(Case(), Model(), entries, [env(), env('new_device', available=False)])
    assert [(j.environment.device_type, j.entries) for j in jobs] == [
        ('cpu', ('a',)), ('new_device', ('b',))]


def test_opaque_condition_is_not_invented_or_narrowed():
    class Opaque:
        def evaluate(self, facts):
            return Support.UNKNOWN
        def specialize(self, facts):
            return self
    acc = entry()
    acc = Acceleration(acc.reference, Opaque(), acc.implementation)
    jobs, _ = planning.placements(Case(), Model(), {'opaque': acc}, [env()])
    assert jobs and all(j.bindings[0]['static_support'] == 'UNKNOWN' for j in jobs)
    # Actual bounded witness search, not planner guesses, diagnoses unresolved.
    from mstar_accelerators.testing.solve import solve_when, SolveStatus
    assert solve_when(acc.when, {}, {'graph_walk': ('walk',)}).status is SolveStatus.UNRESOLVED


def setup_coordinator(monkeypatch, entries, discovered):
    from mstar_accelerators.backends import discovery
    monkeypatch.setattr(environments, 'load_accelerations', lambda source: entries)
    monkeypatch.setattr(corpus, 'MODELS', (Case(),))
    monkeypatch.setattr(discovery, 'discover', lambda: tuple(discovered))


def test_missing_assets_never_reserves_hardware_and_plan_is_report_output(monkeypatch):
    setup_coordinator(monkeypatch, {'a': entry()}, [env()])
    def missing(case):
        raise Unavailable('reference_assets_unavailable', 'missing pinned checkpoint')
    monkeypatch.setattr(assets, 'resolve', missing)
    monkeypatch.setattr(hosting, 'execute', lambda *a: pytest.fail('must not reserve'))
    report = environments.run('synthetic:entries')
    assert report['status'] == 'SKIP' and not report['complete']
    assert report['execution_plan']['jobs'][0]['unavailable']['code'] == 'reference_assets_unavailable'
    assert report['execution_plan']['jobs'][0]['entries'] == ['a']


def test_unavailable_target_does_not_resolve_weights(monkeypatch):
    setup_coordinator(monkeypatch, {'a': entry()}, [env(available=False)])
    monkeypatch.setattr(assets, 'resolve', lambda *a: pytest.fail('unneeded weights'))
    assert environments.run('synthetic:entries')['status'] == 'SKIP'


def test_integrity_failure_cannot_become_missing_evidence(monkeypatch):
    setup_coordinator(monkeypatch, {'a': entry()}, [env()])
    def corrupt(case):
        raise assets.AssetIntegrityError('corrupt bytes')
    monkeypatch.setattr(assets, 'resolve', corrupt)
    assert environments.run('synthetic:entries')['status'] == 'FAIL'


def test_failed_environment_and_missing_supported_environment_cannot_hide(monkeypatch):
    acc = entry()
    # An unconstrained device supports both discovered environments.
    acc = Acceleration(Logical, When(Eq('tp_size', 1), Eq('graph_walk', 'walk')), acc.implementation)
    setup_coordinator(monkeypatch, {'a': acc}, [env(), env('other', available=False)])
    monkeypatch.setattr(assets, 'resolve', lambda case: ('unused', {}, {}))
    monkeypatch.setattr(hosting, 'execute', lambda source, job, *args:
        run_registry({'a': acc}, lambda name, item: Result(name, Status.PASS)).to_dict())
    report = environments.run('synthetic:entries')
    assert report['counts'] == {'PASS': 0, 'FAIL': 0, 'SKIP': 1}
    monkeypatch.setattr(hosting, 'execute', lambda source, job, *args:
        run_registry({'a': acc}, lambda name, item: Result(name, Status.FAIL)).to_dict())
    assert environments.run('synthetic:entries')['status'] == 'FAIL'


@pytest.mark.parametrize('algorithm', ['sha256', 'git_blob_sha1'])
def test_checksum_rejects_same_size_corruption(tmp_path, algorithm):
    value = b'fixed bytes'
    digest = (hashlib.sha256(value).hexdigest() if algorithm == 'sha256' else
              hashlib.sha1(f'blob {len(value)}\0'.encode() + value).hexdigest())
    path = tmp_path / 'asset'
    path.write_bytes(value)
    record = {'bytes': len(value), algorithm: digest}
    assets.verify(path, record)
    path.write_bytes(b'X' + value[1:])
    with pytest.raises(assets.AssetIntegrityError):
        assets.verify(path, record)


def test_real_corpus_matches_mstar_logical_nodes_and_pinned_descriptions():
    for case in corpus.MODELS:
        model = case.describe()
        nodes, absent = corpus.logical_nodes(model)
        assert nodes
        for name, (shell, walks) in nodes.items():
            assert walks and type(model.create_submodule(name)) is type(shell)
        assert len(case.manifest['revision']) == 40


def test_discovery_has_no_model_or_policy_dependencies():
    import ast
    from mstar_accelerators.backends import discovery
    source = Path(discovery.__file__).read_text()
    imports = [node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    assert not any(name and ('testing' in name or 'model' in name) for name in imports)
    assert 'Whisper' not in source and 'Qwen' not in source


def test_asset_resolution_requests_exact_manifest_files_not_unneeded_hub_files(tmp_path, monkeypatch):
    import huggingface_hub
    value = b'pinned config'
    record = {'path': 'config.json', 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
    (tmp_path / 'config.json').write_bytes(value)
    case = SimpleNamespace(manifest={'model': 'owner/model', 'revision': 'a' * 40, 'files': [record]})
    def snapshot(repo, **kwargs):
        assert repo == 'owner/model'
        assert kwargs == {'revision': 'a' * 40, 'allow_patterns': ['config.json'], 'local_files_only': True}
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    checkpoint, fixtures, receipt = assets.resolve(case)
    assert checkpoint == str(tmp_path) and not fixtures and receipt['verified'] == [record]


def test_failed_cpu_spawn_retires_already_started_peers(tmp_path, monkeypatch):
    events = []
    class Process:
        def __init__(self, index):
            self.index, self.live = index, False
        def start(self):
            if self.index:
                raise RuntimeError('spawn failure')
            self.live = True
            events.append('started')
        def is_alive(self):
            return self.live
        def terminate(self):
            self.live = False
            events.append('terminated')
        def join(self, timeout=None):
            events.append('joined')
    processes = []
    def process(**kwargs):
        result = Process(len(processes))
        processes.append(result)
        return result
    monkeypatch.setattr(hosting.mp, 'get_context', lambda mode: SimpleNamespace(Process=process))
    runtime = SimpleNamespace(admission_context=lambda: None, reserve_worker=lambda context, name: None)
    environment = SimpleNamespace(runtime=lambda source: runtime, device_type='cpu')
    job = SimpleNamespace(environment=environment, tp_size=2)
    with pytest.raises(RuntimeError, match='spawn failure'):
        hosting.execute('unused', job, 'unused', {})
    assert events == ['started', 'terminated', 'joined']


def test_unavailable_topology_cannot_disappear_behind_a_passing_placement(monkeypatch):
    from mstar_accelerators import Or
    base = entry()
    acc = Acceleration(Logical, When(Eq('device_type', 'cpu'),
        Or(Eq('tp_size', 1), Eq('tp_size', 2)), Eq('graph_walk', 'walk')), base.implementation)
    limited = Environment('cpu', 'cpu', 'cpu', 'cpu', 1, None, 'unused:Runtime')
    setup_coordinator(monkeypatch, {'a': acc}, [limited])
    monkeypatch.setattr(assets, 'resolve', lambda case: ('unused', {}, {}))
    monkeypatch.setattr(hosting, 'execute', lambda source, job, *args:
        run_registry({'a': acc}, lambda name, item: Result(name, Status.PASS)).to_dict())
    report = environments.run('synthetic:entries')
    assert report['counts'] == {'PASS': 0, 'FAIL': 0, 'SKIP': 1}
    assert len(report['execution_plan']['jobs']) == 2
    assert report['execution_plan']['jobs'][1]['unavailable']['code'] == 'target_unavailable'


def test_available_binding_uses_real_device_and_unavailable_binding_stays_unknown():
    class BindingCondition:
        def evaluate(self, facts):
            if 'binding' not in facts:
                return Support.UNKNOWN
            assert isinstance(facts['binding'].device, torch.device)
            return Support.YES
        def specialize(self, facts):
            self.evaluate(facts)
            return self
    base = entry()
    acc = Acceleration(Logical, BindingCondition(), base.implementation)
    jobs, _ = planning.placements(Case(), Model(), {'a': acc}, [env(), env('absent', available=False)])
    assert jobs
    assert all(record['static_support'] == 'UNKNOWN' for job in jobs
               if not job.environment.available for record in job.bindings)


def test_inferred_pi_lifecycle_preserves_required_native_validation(monkeypatch):
    import sys
    from mstar_accelerators.backends import discovery
    real_find = discovery.importlib.util.find_spec
    monkeypatch.setattr(discovery.importlib.util, 'find_spec',
                        lambda name: None if name == 'mstar_pi' else real_find(name))
    environment = next(env for env in discovery.discover() if env.device_type == 'mstar_pi')
    monkeypatch.setitem(sys.modules, 'mstar_accelerators.backends.pi.runtime',
                        SimpleNamespace(PiRuntime=lambda config: config))
    assert environment.runtime('chosen:entries') == {
        'accelerations': 'chosen:entries',
        'native': {'wait_timeout_ms': 180000, 'validation': True}}
    assert environment.receipt()['runtime_options']['native']['validation'] is True
