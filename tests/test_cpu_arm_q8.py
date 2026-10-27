"""Physical capability/ownership controls only; no native numerical claim."""
import hashlib
from types import SimpleNamespace

import pytest
import torch
from mstar.engine.execution import Support

from mstar_accelerators.backends.torch import q8


def binding(config=None, device='cpu'):
    return SimpleNamespace(device=torch.device(device), prepared={},
                           model_config={'runtime_extension': {'cpu_q8': config}})


def test_arm_q8_unknown_without_binding_and_no_library_required_on_other_hosts(monkeypatch):
    condition = q8.ArmQ8Available()
    assert condition.evaluate({}) is Support.UNKNOWN
    assert condition.specialize({}).evaluate({}) is Support.UNKNOWN
    monkeypatch.setattr(q8, '_arm_platform', lambda: False)
    assert condition.evaluate({'binding': binding()}) is Support.NO
    assert condition.evaluate({'binding': binding(device='meta')}) is Support.NO


def test_artifact_validation_does_not_silently_ignore_bad_cpu_q8():
    with pytest.raises(ValueError, match='exactly library and sha256'):
        q8.validate_artifact_config({'library': '/tmp/example'})
    with pytest.raises(ValueError, match='absolute'):
        q8.validate_artifact_config({'library': 'relative.so', 'sha256': '0' * 64})


def test_configured_library_handle_is_retained_by_the_actual_binding(tmp_path, monkeypatch):
    path = tmp_path / 'explicit-test-artifact'
    path.write_bytes(b'test only; native loading is replaced for this ownership unit test')
    config = {'library': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    loaded = []
    def load(name, **kwargs):
        assert name == str(path)
        handle = SimpleNamespace(q8cpu_capability=lambda: 1, q8cpu_readout=lambda *_args: 0)
        loaded.append(handle)
        return handle
    monkeypatch.setattr(q8, '_arm_platform', lambda: True)
    monkeypatch.setattr(q8.ctypes, 'CDLL', load)
    owner = binding(config)
    assert q8.ArmQ8Available().evaluate({'binding': owner}) is Support.YES
    assert q8.library(owner) is loaded[0] and len(loaded) == 1
    assert list(owner.prepared.values()) == loaded
    other = binding(config)
    assert q8.library(other) is loaded[1] and len(loaded) == 2
    config['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA256'):
        q8.library(binding(config))


def test_configured_non_dot_cpu_rejects_without_scalar_fallback(tmp_path, monkeypatch):
    path = tmp_path / 'explicit-test-artifact'
    path.write_bytes(b'capability test only')
    config = {'library': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(q8, '_arm_platform', lambda: True)
    monkeypatch.setattr(q8.ctypes, 'CDLL', lambda *_args, **_kwargs: SimpleNamespace(
        q8cpu_capability=lambda: 0, q8cpu_readout=lambda *_args: pytest.fail('compute must not run')))
    owner = binding(config)
    assert q8.ArmQ8Available().evaluate({'binding': owner}) is Support.NO
    assert owner.prepared == {}
