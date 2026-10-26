"""Hardware inventory and existing lifecycle factories; no model/test policy.

Discovery never reserves a device. Qualification and other tooling may consume
these facts; allocation and settlement remain with M* and the backend runtime.
"""
from dataclasses import dataclass, asdict
import importlib.util
import os


@dataclass(frozen=True)
class Environment:
    key: str
    device: str
    device_type: str
    platform: str
    max_ranks: int
    collective: str | None
    runtime_factory: str
    available: bool = True
    reason: str = ''
    memory_bytes: int | None = None
    host_threads: int = 4

    def facts(self):
        return {'device_type': self.device_type, 'platform': self.platform}

    def receipt(self):
        return asdict(self)

    def prepare_process(self):
        import torch
        torch.set_num_threads(self.host_threads)

    def runtime(self, registry):
        from importlib import import_module
        module, name = self.runtime_factory.split(':')
        return getattr(import_module(module), name)({'accelerations': registry})

    def stream(self, device):
        if self.device_type == 'mstar_pi':
            import mstar_pi
            return mstar_pi.current_stream(device)
        if self.device_type == 'cuda':
            import torch
            return torch.cuda.current_stream(device)
        return None


def discover():
    import torch
    gloo = torch.distributed.is_available() and torch.distributed.is_gloo_available()
    result = [Environment('cpu', 'cpu', 'cpu', 'cpu', os.cpu_count() or 1,
                          'gloo' if gloo else None,
                          'mstar_accelerators.backends.torch.runtime:TorchRuntime', host_threads=1)]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result.append(Environment(f'gpu:{index}', f'cuda:{index}', 'cuda',
                'rocm' if torch.version.hip else 'cuda', 1, None,
                'mstar_accelerators.backends.torch.device_runtime:DeviceTorchRuntime',
                memory_bytes=props.total_memory))
    # Known backend families remain represented when unavailable. Otherwise
    # a CPU pass could hide an untested ROCm alternative on a non-ROCm host.
    present = {item.platform for item in result if item.device_type == 'cuda'}
    for platform in ('cuda', 'rocm'):
        if platform not in present:
            result.append(Environment(f'{platform}:unavailable', 'cuda:0', 'cuda',
                platform, 1, None,
                'mstar_accelerators.backends.torch.device_runtime:DeviceTorchRuntime',
                available=False, reason=f'no available {platform} device in this Torch environment'))
    installed = importlib.util.find_spec('mstar_pi') is not None
    available, reason = False, 'mstar_pi is not installed'
    if installed:
        import mstar_pi
        from mstar_pi.admission import AdmissionContext
        available = mstar_pi.is_available()
        reason = '' if available else 'Pi capability probe unavailable'
        if available:
            context = AdmissionContext.from_environment()
            if context is None:
                available, reason = False, 'launch through run_hardware.py --managed'
            else:
                context.close()
    result.append(Environment('pi', 'mstar_pi:0', 'mstar_pi', 'cpu', 1, None,
        'mstar_accelerators.backends.pi.runtime:PiRuntime', available=available, reason=reason))
    return tuple(result)
