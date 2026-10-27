"""Plain-Q8 public vocabulary MUL_MAT in the existing F16 decoder graph/pool."""
import ctypes as C
from dataclasses import dataclass
import logging

import torch

from mstar.engine.execution import Support
from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE
from mstar_accelerators import PreparedImplementation
from mstar_accelerators.backends.torch import TorchProgram, prepare_torch_programs
from mstar_accelerators.compiler import bind_default_call
from mstar_accelerators.conditions import Constant
from .cpu_ggml_decoder import size
from .cpu_ggml_storage import StorageBody, StorageCall, library as storage_library
from mstar_accelerators.implementations.whisper.cpu_q8 import prepared_weights

logger = logging.getLogger(__name__)


class ReadoutCall(C.Structure):
    _fields_ = [('body', StorageCall), ('logits', C.c_void_p), ('logits_bytes', C.c_uint64),
                ('token', C.c_void_p), ('token_bytes', C.c_uint64),
                ('suppressed', C.c_void_p), ('suppressed_count', C.c_uint64),
                ('suppressed_bytes', C.c_uint64)]


def library(binding, *, readout_format="q8_0"):
    if readout_format not in ("q8_0", "q4_0", "q4_0_kai"):
        raise ValueError("unsupported physical readout format")
    api = storage_library(binding)
    if api is None or not all(hasattr(api, name) for name in (
            'cpu_decoder_readout_abi', 'cpu_decoder_set_readout_q8',
            'cpu_decoder_storage_readout_run')):
        return None
    api.cpu_decoder_readout_abi.argtypes = []
    api.cpu_decoder_readout_abi.restype = C.c_int
    if api.cpu_decoder_readout_abi() != 1:
        return None
    api.cpu_decoder_set_readout_q8.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64,
        C.c_int64, C.c_int64, C.c_void_p, C.c_size_t]
    api.cpu_decoder_set_readout_q8.restype = C.c_int
    if readout_format in ('q4_0', 'q4_0_kai'):
        if not all(hasattr(api, name) for name in ('cpu_decoder_set_readout_q4', 'cpu_decoder_pack_readout_q4')):
            return None
        api.cpu_decoder_set_readout_q4.argtypes = api.cpu_decoder_set_readout_q8.argtypes
        api.cpu_decoder_set_readout_q4.restype = C.c_int
        api.cpu_decoder_pack_readout_q4.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64,
            C.c_int64, C.c_int64, C.c_void_p, C.c_size_t]
        api.cpu_decoder_pack_readout_q4.restype = C.c_int
        if readout_format == 'q4_0_kai':
            if not hasattr(api, 'cpu_decoder_set_readout_q4_kai'):
                return None
            api.cpu_decoder_set_readout_q4_kai.argtypes = api.cpu_decoder_set_readout_q8.argtypes
            api.cpu_decoder_set_readout_q4_kai.restype = C.c_int
    api.cpu_decoder_storage_readout_run.argtypes = [C.c_void_p, C.POINTER(ReadoutCall), C.c_void_p, C.c_size_t]
    api.cpu_decoder_storage_readout_run.restype = C.c_int
    return api


@dataclass(frozen=True)
class CPUGraphReadoutAvailable:
    def evaluate(self, facts):
        binding = facts.get('binding')
        if binding is None:
            return Support.UNKNOWN
        return Support.YES if library(binding) is not None else Support.NO

    def specialize(self, facts):
        support = self.evaluate(facts)
        return self if support is Support.UNKNOWN else Constant(support)


class GraphReadoutBody(StorageBody):
    """Existing storage body plus immutable projection, no new state owner."""
    def __init__(self, binding, weights, resources, packed, suppressed, shape, *, readout_format="q8_0"):
        if readout_format not in ("q8_0", "q4_0", "q4_0_kai"):
            raise ValueError("unsupported physical readout format")
        block_bytes = 34 if readout_format == "q8_0" else 18
        rows, width = shape
        if (packed.device.type != 'cpu' or packed.dtype != torch.int32 or not packed.is_contiguous()
                or width != binding.submodule.config.d_model or rows != binding.submodule.config.vocab_size
                or size(packed) != rows * (width // 32) * block_bytes or width % 32):
            raise ValueError('graph readout packed projection differs from logical shape')
        for mask in suppressed.values():
            if mask.device.type != 'cpu' or mask.dtype != torch.int64 or not mask.is_contiguous():
                raise ValueError('graph readout suppression storage differs')
        # Keep the selected canonical packed bytes alive until native close.
        # Both books share this same physical owner in binding.prepared.
        self.packed, self.suppressed, self.rows = packed, suppressed, rows
        super().__init__(binding, weights, resources)
        try:
            api = library(binding, readout_format=readout_format)
            if api is None:
                raise ValueError('configured artifact lacks graph readout ABI v1')
            self.library = api
            error = C.create_string_buffer(1024)
            setter = getattr(api, {"q8_0": "cpu_decoder_set_readout_q8", "q4_0": "cpu_decoder_set_readout_q4",
                                   "q4_0_kai": "cpu_decoder_set_readout_q4_kai"}[readout_format])
            if setter(self.handle, packed.data_ptr(), size(packed), rows, width, error, len(error)):
                raise RuntimeError('CPU graph readout preparation: ' + error.value.decode())
        except BaseException:
            self.close()
            raise
        logger.info('CPU_GGML_GRAPH_READOUT_READY rows=%d width=%d provider=%s format=%s '
                    'state=existing_resolved_storage projection=binding_immutable', rows, width, 'CPU_KLEIDIAI' if readout_format == 'q4_0_kai' else 'CPU', readout_format.upper())

    def execute_native(self, actual, hidden, call):
        logits = torch.empty(self.rows, dtype=torch.float32)
        token = torch.empty(1, dtype=torch.int64)
        mask = self.suppressed[call.graph_walk]
        bound = ReadoutCall(actual, logits.data_ptr(), size(logits), token.data_ptr(), size(token),
                            mask.data_ptr(), mask.numel(), size(mask))
        error = C.create_string_buffer(1024)
        if self.library.cpu_decoder_storage_readout_run(self.handle, C.byref(bound), error, len(error)):
            raise RuntimeError('CPU graph readout execution: ' + error.value.decode())
        return logits, token

    def close(self):
        super().close()
        self.packed = None  # only after synchronous native destruction succeeds


def prepare_body(binding, weights, resources, packed, suppressed, shape):
    key = (prepare_body, tuple(resources[name].config.encoding for name in (KV_CACHE, CROSS_KV_CACHE)))
    if key not in binding.prepared:
        binding.prepared[key] = GraphReadoutBody(binding, weights, resources, packed, suppressed, shape)
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

