"""Native decoder specialization from the runtime's resolved plain KV encoding."""
import ctypes as C
from dataclasses import dataclass
import logging

import torch

from mstar.engine.execution import Support
from mstar.engine.resources.sequence import DenseSequenceView, PlainKVEncoding, Replicated
from mstar.engine.resources.kv.config import KVSpec
from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE
from mstar_accelerators.conditions import Constant
from .cpu_ggml_decoder import Body, KV, NativeCall, library as base_library
from .cpu_ggml_q8_decoder import library as q8_library

logger = logging.getLogger(__name__)


def resource_requests(binding):
    return {spec.resource_key: DenseSequenceView(
                partition=Replicated(), encoding=PlainKVEncoding(torch.float16))
            for spec in binding.resource_specs
            if isinstance(spec, KVSpec) and binding.node_name in spec.nodes}


class StoragePlan(C.Structure):
    _fields_ = [('self_encoding', C.c_uint32), ('cross_encoding', C.c_uint32),
                ('self_capacity', C.c_int32), ('cross_capacity', C.c_int32)]


class StorageKV(C.Structure):
    # A distinct ABI type even though these v1 plain formats need the same fields.
    _fields_ = KV._fields_


class StorageCall(C.Structure):
    _fields_ = [(name, C.POINTER(StorageKV) if name == 'kv' else kind)
                for name, kind in NativeCall._fields_]


def library(binding, *, quantized=True):
    api = q8_library(binding) if quantized else base_library(binding)
    if api is None or not all(hasattr(api, name) for name in (
            'cpu_decoder_storage_abi', 'cpu_decoder_storage_create',
            'cpu_decoder_q8_storage_create', 'cpu_decoder_storage_run')):
        return None
    api.cpu_decoder_storage_abi.argtypes = []
    api.cpu_decoder_storage_abi.restype = C.c_int
    if api.cpu_decoder_storage_abi() != 1:
        return None
    for name in ('cpu_decoder_storage_create', 'cpu_decoder_q8_storage_create'):
        function = getattr(api, name)
        function.argtypes = [api.cpu_decoder_create.argtypes[0], C.POINTER(StoragePlan),
                             *api.cpu_decoder_create.argtypes[1:]]
        function.restype = C.c_void_p
    api.cpu_decoder_storage_run.argtypes = [C.c_void_p, C.POINTER(StorageCall), C.c_void_p, C.c_size_t]
    api.cpu_decoder_storage_run.restype = C.c_int
    return api


@dataclass(frozen=True)
class CPUStorageDecoderAvailable:
    def evaluate(self, facts):
        binding = facts.get('binding')
        if binding is None:
            return Support.UNKNOWN
        return Support.YES if library(binding) is not None else Support.NO

    def specialize(self, facts):
        support = self.evaluate(facts)
        return self if support is Support.UNKNOWN else Constant(support)


class StorageBody(Body):
    """One prepared program; no cache allocation, mirror, or request ownership."""
    kv_type = StorageKV
    call_type = StorageCall
    quantized = True

    def bind_views(self, resources):
        self.cache_dtypes, self.cache_capacities = {}, {}
        formats, capacities, views = [], [], {}
        ids = {torch.float32: 1, torch.float16: 2}  # decoder_storage.h ABI v1
        for key in (KV_CACHE, CROSS_KV_CACHE):
            resource = resources[key]
            encoding = resource.config.encoding
            if not isinstance(encoding, PlainKVEncoding) or encoding.dtype not in ids:
                raise ValueError('native storage ABI v1 requires resolved plain F32/F16 encoding')
            view = resource.view(DenseSequenceView(encoding=encoding))
            logical_capacity = (self.config.max_target_positions if key == KV_CACHE
                                else self.config.max_source_positions)
            if (view.config.max_seq_len != logical_capacity or
                    (view.config.num_layers, view.config.num_kv_heads, view.config.head_dim) !=
                    (self.config.decoder_layers, self.config.decoder_attention_heads,
                     self.config.d_model // self.config.decoder_attention_heads)):
                raise ValueError('native storage ABI v1 requires full unpadded replicated logical geometry')
            self.cache_capacities[key] = view.config.max_seq_len
            self.cache_dtypes[key] = encoding.dtype
            formats.append(ids[encoding.dtype])
            capacities.append(view.config.max_seq_len)
            views[key] = view
        self.storage_plan = StoragePlan(*formats, *capacities)
        return views

    def create_handle(self, physical, descriptors, error):
        api = library(self._binding, quantized=self.quantized)
        if api is None:
            raise ValueError('configured artifact lacks the explicit decoder storage ABI')
        self.library = api
        creator = api.cpu_decoder_q8_storage_create if self.quantized else api.cpu_decoder_storage_create
        handle = creator(C.byref(physical), C.byref(self.storage_plan), descriptors,
                         len(descriptors), error, len(error))
        if handle:
            logger.info('CPU_GGML_STORAGE_DECODER_READY self=%s cross=%s capacities=%s '
                        'state=single_runtime_owner scale=unscaled padding=none',
                        self.cache_dtypes[KV_CACHE], self.cache_dtypes[CROSS_KV_CACHE],
                        (self.storage_plan.self_capacity, self.storage_plan.cross_capacity))
        return handle

    def __init__(self, binding, weights, resources):
        self._binding = binding
        try:
            super().__init__(binding, weights, resources)
        finally:
            del self._binding

    def run_native(self, actual, error):
        return self.library.cpu_decoder_storage_run(self.handle, C.byref(actual), error, len(error))


def prepare_body(binding, weights, resources):
    # Greedy and stochastic books use the same body and actual owner resources.
    key = (prepare_body, tuple(resources[name].config.encoding for name in (KV_CACHE, CROSS_KV_CACHE)))
    if key not in binding.prepared:
        binding.prepared[key] = StorageBody(binding, weights, resources)
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
