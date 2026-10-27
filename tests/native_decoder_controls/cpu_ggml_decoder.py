"""F32 ggml decoder body over admitted, runtime-owned dense KV views."""
import ctypes as C
from dataclasses import dataclass
import logging

import torch
import torch.nn.functional as F

from mstar.engine.execution import Support
from mstar.engine.resources.sequence import DenseSequenceView
from mstar.model.whisper.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, POS
from mstar_accelerators.conditions import Constant
from mstar_accelerators.implementations.whisper.cpu_ggml import Weight, load_library, setting, native_threads

logger = logging.getLogger(__name__)


class Config(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ('width', 'layers', 'heads', 'ffn', 'source', 'target', 'threads')]


class KV(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in ('self_k', 'self_v', 'cross_k', 'cross_v')] + [
        ('self_bytes', C.c_uint64), ('cross_bytes', C.c_uint64)]


class NativeCall(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ('queries', 'self_length', 'cross_length')] + [
        ('x', C.c_void_p), ('x_bytes', C.c_uint64), ('encoder', C.c_void_p), ('encoder_bytes', C.c_uint64),
        ('self_positions', C.c_void_p), ('self_position_bytes', C.c_uint64),
        ('cross_positions', C.c_void_p), ('cross_position_bytes', C.c_uint64),
        ('self_mask', C.c_void_p), ('self_mask_bytes', C.c_uint64),
        ('cross_mask', C.c_void_p), ('cross_mask_bytes', C.c_uint64),
        ('kv', C.POINTER(KV)), ('layer_count', C.c_uint64), ('hidden', C.c_void_p), ('hidden_bytes', C.c_uint64)]


def library(binding):
    value = setting(binding)
    if value is None or binding.device.type != 'cpu':
        return None
    key = (library, tuple(sorted(value.items())))
    if key not in binding.prepared:
        api = load_library(value)
        if not hasattr(api, 'cpu_decoder_abi'):
            return None
        api.cpu_decoder_abi.argtypes = []
        api.cpu_decoder_abi.restype = C.c_int
        if api.cpu_decoder_abi() != 1:
            return None
        api.cpu_decoder_create.argtypes = [C.POINTER(Config), C.POINTER(Weight), C.c_size_t, C.c_void_p, C.c_size_t]
        api.cpu_decoder_create.restype = C.c_void_p
        api.cpu_decoder_run.argtypes = [C.c_void_p, C.POINTER(NativeCall), C.c_void_p, C.c_size_t]
        api.cpu_decoder_run.restype = C.c_int
        api.cpu_decoder_close.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
        api.cpu_decoder_close.restype = C.c_int
        binding.prepared[key] = api
    return binding.prepared[key]


@dataclass(frozen=True)
class CPUDecoderAvailable:
    def evaluate(self, facts):
        binding = facts.get('binding')
        if binding is None:
            return Support.UNKNOWN
        return Support.YES if library(binding) is not None else Support.NO

    def specialize(self, facts):
        value = self.evaluate(facts)
        return self if value is Support.UNKNOWN else Constant(value)


def checked(tensor, shape, dtype=torch.float32):
    if (tensor.device.type != 'cpu' or tensor.dtype != dtype
            or tuple(tensor.shape) != shape or not tensor.is_contiguous()):
        raise ValueError('admitted dense CPU storage differs from its physical contract')
    return tensor


def size(tensor):
    return tensor.numel() * tensor.element_size()


class Body:
    kv_type = KV
    call_type = NativeCall

    def bind_views(self, resources):
        self.cache_dtypes = {key: torch.float32 for key in (KV_CACHE, CROSS_KV_CACHE)}
        self.cache_capacities = {KV_CACHE: self.config.max_target_positions,
                                 CROSS_KV_CACHE: self.config.max_source_positions}
        return {key: resources[key].view(DenseSequenceView()) for key in self.cache_dtypes}

    def run_native(self, actual, error):
        return self.library.cpu_decoder_run(self.handle, C.byref(actual), error, len(error))

    def __init__(self, binding, weights, resources):
        threads = native_threads(binding)
        self.library = library(binding)
        if self.library is None:
            raise ValueError('the configured artifact has no native CPU decoder body')
        self.handle = None
        self.users = 0
        self.config = c = binding.submodule.config
        self.embedding = weights['decoder.embed_tokens.weight']
        self.positions = weights['decoder.embed_positions.weight']
        self.views = self.bind_views(resources)
        physical = Config(c.d_model, c.decoder_layers, c.decoder_attention_heads, c.decoder_ffn_dim,
                          c.max_source_positions, c.max_target_positions, threads)
        body = {n: t.detach().to(device='cpu', dtype=torch.float32).contiguous()
                for n, t in weights.items() if n not in ('decoder.embed_tokens.weight',
                    'decoder.proj_out.weight', 'decoder.embed_positions.weight')}
        descriptors = (Weight * len(body))()
        for i, (name, tensor) in enumerate(sorted(body.items())):
            descriptors[i] = Weight(name.encode(), tensor.data_ptr(), size(tensor), tensor.ndim,
                                    (C.c_int64 * 3)(*reversed(tensor.shape)))
        error = C.create_string_buffer(1024)
        self.handle = self.create_handle(physical, descriptors, error)
        if not self.handle:
            raise RuntimeError('CPU decoder preparation: ' + error.value.decode())
        cross_threads = binding.model_config.get('runtime_extension', {}).get('cpu_ggml_cross_threads')
        if cross_threads is not None:
            try:
                if type(cross_threads) is not int or not 1 <= cross_threads <= 64:
                    raise ValueError('cross preparation threads must be an integer in [1, 64]')
                api = self.library.cpu_decoder_set_cross_threads
                api.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t]
                api.restype = C.c_int
                if api(self.handle, cross_threads, error, len(error)):
                    raise RuntimeError('cross preparation: ' + error.value.decode())
            except BaseException:
                self.close()
                raise
        capture = binding.model_config.get('runtime_extension', {}).get('cpu_ggml_attention_capture')
        if capture is not None:
            try:
                from pathlib import Path
                if not isinstance(capture, str) or not Path(capture).is_absolute():
                    raise ValueError('attention capture must be an absolute fresh file prefix')
                paths = [Path(capture + suffix) for suffix in ('.json', '.q.f32', '.k.f32', '.v.f32', '.mask.f32')]
                present = [path.exists() for path in paths]
                if any(present) and not all(present):
                    raise ValueError('partial attention capture exists; preserve it and choose a fresh prefix')
                if not all(present):
                    api = self.library.cpu_decoder_capture_attention
                    api.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_size_t]
                    api.restype = C.c_int
                    if api(self.handle, capture.encode(), error, len(error)):
                        raise RuntimeError('attention capture: ' + error.value.decode())
            except BaseException:
                self.close()
                raise
        logger.info('CPU_GGML_DECODER_READY threads=%d body_weight_bytes=%d graph_allocation=first_admitted_call '
                    'borrowed_kv=True attention_mask=lossless_zero_neginf_f16',
                    physical.threads, sum(size(t) for t in body.values()))

    def create_handle(self, physical, descriptors, error):
        return self.library.cpu_decoder_create(C.byref(physical), descriptors, len(descriptors), error, len(error))

    @torch.no_grad()
    def run(self, call):
        if self.handle is None:
            raise RuntimeError('CPU decoder body is closed')
        c = self.config
        ids, = call.inputs.input_ids
        positions = getattr(call.plan_results, POS)['main']
        x = F.embedding(ids, self.embedding)
        if c.scale_embedding:
            x = x * c.d_model ** 0.5
        x = (x + F.embedding(positions.long(), self.positions)).contiguous()
        q = ids.numel()
        checked(x, (q, c.d_model))
        write, = self.views[KV_CACHE].for_step(getattr(call.plan_results, KV_CACHE)['main'])
        read, = self.views[KV_CACHE].for_attention(getattr(call.plan_results, ATTN)['main'])
        cross, = self.views[CROSS_KV_CACHE].for_attention(getattr(call.plan_results, CROSS_ATTN)['main'])
        prefill = call.graph_walk == 'prefill'
        if (q != (4 if prefill else 1) or write.request_id != read.request_id
                or cross.request_id != read.request_id):
            raise ValueError('decoder call and admitted row identities differ')
        self_positions = checked(write.positions, (q,), torch.int64)
        self_mask = checked(read.mask.contiguous(), (q, self.cache_capacities[KV_CACHE]))
        cross_mask = checked(cross.mask.contiguous(), (q, self.cache_capacities[CROSS_KV_CACHE]))
        encoder = cross_positions = cross_write = None
        if prefill:
            encoder, = call.inputs.encoder_states
            encoder = checked(encoder.contiguous(), (c.max_source_positions, c.d_model))
            cross_write, = self.views[CROSS_KV_CACHE].for_step(getattr(call.plan_results, CROSS_KV_CACHE)['main'])
            if cross_write.request_id != cross.request_id:
                raise ValueError('cross write/read row identities differ')
            cross_positions = checked(cross_write.positions, (c.max_source_positions,), torch.int64)
        kv = (self.kv_type * c.decoder_layers)()
        shape = (c.decoder_attention_heads, c.d_model // c.decoder_attention_heads)
        for layer in range(c.decoder_layers):
            sk = checked(read.keys[layer], (self.cache_capacities[KV_CACHE], *shape), self.cache_dtypes[KV_CACHE])
            sv = checked(read.values[layer], (self.cache_capacities[KV_CACHE], *shape), self.cache_dtypes[KV_CACHE])
            ck = checked(cross.keys[layer], (self.cache_capacities[CROSS_KV_CACHE], *shape), self.cache_dtypes[CROSS_KV_CACHE])
            cv = checked(cross.values[layer], (self.cache_capacities[CROSS_KV_CACHE], *shape), self.cache_dtypes[CROSS_KV_CACHE])
            # Both uses come from the model's same request/stream declaration;
            # validate the concrete alias before giving native SET_ROWS access.
            for writable, readable in ((write.keys[layer], sk), (write.values[layer], sv)):
                checked(writable, tuple(readable.shape), readable.dtype)
                if writable.data_ptr() != readable.data_ptr():
                    raise ValueError('self write/read dense aliases differ')
            if prefill:
                for writable, readable in ((cross_write.keys[layer], ck), (cross_write.values[layer], cv)):
                    checked(writable, tuple(readable.shape), readable.dtype)
                    if writable.data_ptr() != readable.data_ptr():
                        raise ValueError('cross write/read dense aliases differ')
            kv[layer] = self.kv_type(sk.data_ptr(), sv.data_ptr(), ck.data_ptr(), cv.data_ptr(), size(sk), size(ck))
        hidden = torch.empty((1, c.d_model), dtype=torch.float32)
        actual = self.call_type(q, read.valid_length, cross.valid_length,
            x.data_ptr(), size(x), encoder.data_ptr() if prefill else None, size(encoder) if prefill else 0,
            self_positions.data_ptr(), size(self_positions), cross_positions.data_ptr() if prefill else None,
            size(cross_positions) if prefill else 0, self_mask.data_ptr(), size(self_mask),
            cross_mask.data_ptr(), size(cross_mask), kv, c.decoder_layers, hidden.data_ptr(), size(hidden))
        return self.execute_native(actual, hidden, call)

    def execute_native(self, actual, hidden, call):
        error = C.create_string_buffer(1024)
        if self.run_native(actual, error):
            raise RuntimeError('CPU decoder execution: ' + error.value.decode())
        return hidden

    def close(self):
        if self.handle is not None:
            error = C.create_string_buffer(1024)
            if self.library.cpu_decoder_close(self.handle, error, len(error)):
                raise RuntimeError('CPU decoder close: ' + error.value.decode())
            self.handle = None


def prepare_body(binding, weights, resources):
    # Canonical helper identity shares one physical body across the two books;
    # no plan, request, sampler offset or borrowed KV pointer is retained here.
    key = (prepare_body, 'decoder')
    if key not in binding.prepared:
        binding.prepared[key] = Body(binding, weights, resources)
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
