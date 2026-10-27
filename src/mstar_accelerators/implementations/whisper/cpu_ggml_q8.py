"""Explicit upstream-compatible CPU Q8 encoder physical entrypoint."""
import ctypes as C
from dataclasses import dataclass
import logging

from mstar.engine.execution import Support
from mstar_accelerators.conditions import Constant
from mstar_accelerators.implementations.whisper.cpu_ggml import Encoder, load_library, setting

logger = logging.getLogger(__name__)


def library(binding):
    value = setting(binding)
    if value is None or binding.device.type != 'cpu':
        return None
    key = (library, tuple(sorted(value.items())))
    if key not in binding.prepared:
        api = load_library(value)
        if not all(hasattr(api, name) for name in ('cpu_encoder_q8_abi', 'cpu_encoder_q8_capability', 'cpu_encoder_q8_create')):
            return None
        for name in ('cpu_encoder_q8_abi', 'cpu_encoder_q8_capability'):
            getattr(api, name).argtypes = []
            getattr(api, name).restype = C.c_int
        if api.cpu_encoder_q8_abi() != 1 or api.cpu_encoder_q8_capability() != 1:
            return None
        binding.prepared[key] = api
    return binding.prepared[key]


@dataclass(frozen=True)
class CPUQ8EncoderAvailable:
    def evaluate(self, facts):
        binding = facts.get('binding')
        if binding is None:
            return Support.UNKNOWN
        return Support.YES if library(binding) is not None else Support.NO

    def specialize(self, facts):
        value = self.evaluate(facts)
        return self if value is Support.UNKNOWN else Constant(value)


class Q8Encoder(Encoder):
    def create_handle(self, physical, descriptors, memory, error):
        api = self.library
        if not hasattr(api, 'cpu_encoder_q8_create'):
            raise ValueError('configured artifact does not implement the explicit Q8 encoder')
        api.cpu_encoder_q8_create.argtypes = api.cpu_encoder_create.argtypes
        api.cpu_encoder_q8_create.restype = C.c_void_p
        handle = api.cpu_encoder_q8_create(C.byref(physical), descriptors, len(descriptors), C.byref(memory), error, len(error))
        if handle:
            logger.info('CPU_GGML_Q8_ENCODER_READY dense=Q8_0_CPU_REPACK source_rounding=F16 '
                        'conv=F16_im2col_KV=F16 gelu=tanh positions=1500 padding_keys=zero_unmasked_to_256')
        return handle


def prepare_encoder(binding):
    if library(binding) is None:
        raise ValueError('explicit CPU_REPACK Q8 encoder capability is unavailable')
    key = (prepare_encoder, 'encoder')
    if key in binding.prepared:
        raise RuntimeError('Q8 encoder is already prepared for this binding')
    encoder = Q8Encoder(binding, setting(binding))
    binding.prepared[key] = encoder

    def close():
        encoder.close()
        if binding.prepared.get(key) is encoder:
            binding.prepared.pop(key)
    return encoder.run, close
