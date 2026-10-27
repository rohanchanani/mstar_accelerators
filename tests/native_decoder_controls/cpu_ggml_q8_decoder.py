"""Upstream Q8_REPACK decoder body with direct FP32-owner attention over runtime-owned FP32 KV."""
import ctypes as C
from dataclasses import dataclass
import logging

from mstar.engine.execution import Support
from mstar_accelerators.conditions import Constant
from mstar_accelerators.implementations.whisper.cpu_ggml import setting
from tests.native_decoder_controls.cpu_ggml_decoder import Body, library as body_library

logger = logging.getLogger(__name__)


def library(binding):
    value = setting(binding)
    if value is None or binding.device.type != 'cpu':
        return None
    key = (library, tuple(sorted(value.items())))
    if key not in binding.prepared:
        api = body_library(binding)
        if api is None or not all(hasattr(api, name) for name in
                ('cpu_decoder_q8_abi', 'cpu_decoder_q8_capability', 'cpu_decoder_q8_create')):
            return None
        for name in ('cpu_decoder_q8_abi', 'cpu_decoder_q8_capability'):
            getattr(api, name).argtypes = []
            getattr(api, name).restype = C.c_int
        if api.cpu_decoder_q8_abi() != 1 or api.cpu_decoder_q8_capability() != 1:
            return None
        binding.prepared[key] = api
    return binding.prepared[key]


@dataclass(frozen=True)
class CPUQ8DecoderAvailable:
    def evaluate(self, facts):
        binding = facts.get('binding')
        if binding is None:
            return Support.UNKNOWN
        return Support.YES if library(binding) is not None else Support.NO

    def specialize(self, facts):
        value = self.evaluate(facts)
        return self if value is Support.UNKNOWN else Constant(value)


class Q8Body(Body):
    def create_handle(self, physical, descriptors, error):
        api = self.library
        if not hasattr(api, 'cpu_decoder_q8_create'):
            raise ValueError('configured artifact does not implement the explicit Q8 decoder')
        api.cpu_decoder_q8_create.argtypes = api.cpu_decoder_create.argtypes
        api.cpu_decoder_q8_create.restype = C.c_void_p
        handle = api.cpu_decoder_q8_create(C.byref(physical), descriptors, len(descriptors), error, len(error))
        if handle:
            logger.info('CPU_GGML_Q8_DECODER_READY dense=Q8_0_CPU_REPACK source_rounding=F16 '
                        'attention=direct_FP32 gelu=tanh state=borrowed_unscaled_FP32 scale=in_flash '
                        'embedding=existing_FP32 readout=existing_ARM_Q8')
        return handle


def prepare_body(binding, weights, resources):
    if library(binding) is None:
        raise ValueError('explicit CPU_REPACK Q8 decoder capability is unavailable')
    key = (prepare_body, 'decoder')
    if key not in binding.prepared:
        binding.prepared[key] = Q8Body(binding, weights, resources)
    owner = binding.prepared[key]
    owner.users += 1
    released = False

    def close():
        nonlocal released
        if released:
            return
        if owner.users == 1:
            owner.close()
            binding.prepared.pop(key)
        owner.users -= 1
        released = True
    return owner.run, close
