"""ctypes boundary for the prepared native token decoder."""
import ctypes as C

import torch

from .cpu_ggml import Weight, load_library, setting

class Config(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ('width', 'layers', 'heads', 'ffn', 'source', 'target', 'threads')]

class StoragePlan(C.Structure):
    _fields_ = [('self_encoding', C.c_uint32), ('cross_encoding', C.c_uint32),
                ('self_capacity', C.c_int32), ('cross_capacity', C.c_int32)]

class Plane(C.Structure):
    _fields_ = [('self_data', C.c_void_p), ('self_bytes', C.c_uint64),
                ('cross_data', C.c_void_p), ('cross_bytes', C.c_uint64)]

class TokenPlan(C.Structure):
    _fields_ = [('embeddings', C.c_void_p), ('embedding_bytes', C.c_uint64),
                ('positions', C.c_void_p), ('position_bytes', C.c_uint64),
                ('self_pages', C.c_uint32), ('cross_pages', C.c_uint32),
                ('embedding_scale', C.c_float), ('layers', Plane * 4)]

class TokenCall(C.Structure):
    _fields_ = [(name, C.c_int32) for name in
        ('queries', 'self_page', 'cross_page', 'self_length', 'cross_length', 'self_begin')] + [
        ('tokens', C.c_void_p), ('token_input_bytes', C.c_uint64),
        ('positions', C.c_void_p), ('position_input_bytes', C.c_uint64),
        ('encoder', C.c_void_p), ('encoder_bytes', C.c_uint64),
        ('logits', C.c_void_p), ('logits_bytes', C.c_uint64),
        ('token', C.c_void_p), ('token_bytes', C.c_uint64),
        ('suppressed', C.c_void_p), ('suppressed_count', C.c_uint64), ('suppressed_bytes', C.c_uint64)]

def checked(tensor, shape, dtype=torch.float32):
    if (tensor.device.type != 'cpu' or tensor.dtype != dtype
            or tuple(tensor.shape) != shape or not tensor.is_contiguous()):
        raise ValueError('admitted dense CPU storage differs from its physical contract')
    return tensor

def size(tensor):
    return tensor.numel() * tensor.element_size()

def library(binding, *, readout_format="q8_0"):
    if readout_format not in ("q8_0", "q4_0", "q4_0_kai"):
        raise ValueError("unsupported physical readout format")
    value = setting(binding)
    if value is None or binding.device.type != "cpu":
        return None
    key = (library, tuple(sorted(value.items())), readout_format)
    if key in binding.prepared:
        return binding.prepared[key]
    api = load_library(value)
    versions = ("cpu_decoder_abi", "cpu_decoder_q8_abi", "cpu_decoder_q8_capability",
                "cpu_decoder_storage_abi", "cpu_decoder_readout_abi", "cpu_decoder_token_abi")
    # Preserve the accepted artifact capability requirements without chaining
    # loaders through superseded Python execution implementations.
    required = (*versions, "cpu_decoder_create", "cpu_decoder_run", "cpu_decoder_close",
                "cpu_decoder_q8_create", "cpu_decoder_storage_create", "cpu_decoder_q8_storage_create",
                "cpu_decoder_storage_run", "cpu_decoder_set_readout_q8", "cpu_decoder_storage_readout_run",
                "cpu_decoder_set_token_inputs", "cpu_decoder_token_run")
    if readout_format != "q8_0":
        required += ("cpu_decoder_set_readout_q4", "cpu_decoder_pack_readout_q4")
    if readout_format == "q4_0_kai":
        required += ("cpu_decoder_set_readout_q4_kai",)
    if not all(hasattr(api, name) for name in required):
        return None
    for name in versions:
        function = getattr(api, name)
        function.argtypes, function.restype = [], C.c_int
        if function() != 1:
            return None
    error = [C.c_void_p, C.c_size_t]
    signatures = {
        "cpu_decoder_q8_storage_create": ([C.POINTER(Config), C.POINTER(StoragePlan),
                                            C.POINTER(Weight), C.c_size_t, *error], C.c_void_p),
        "cpu_decoder_set_readout_q8": ([C.c_void_p, C.c_void_p, C.c_uint64, C.c_int64, C.c_int64, *error], C.c_int),
        "cpu_decoder_set_token_inputs": ([C.c_void_p, C.POINTER(TokenPlan), *error], C.c_int),
        "cpu_decoder_token_run": ([C.c_void_p, C.POINTER(TokenCall), *error], C.c_int),
        "cpu_decoder_close": ([C.c_void_p, *error], C.c_int),
    }
    if readout_format != "q8_0":
        signatures["cpu_decoder_set_readout_q4"] = signatures["cpu_decoder_set_readout_q8"]
        signatures["cpu_decoder_pack_readout_q4"] = ([C.c_void_p, C.c_uint64, C.c_void_p,
                                                      C.c_uint64, C.c_int64, C.c_int64, *error], C.c_int)
    if readout_format == "q4_0_kai":
        signatures["cpu_decoder_set_readout_q4_kai"] = signatures["cpu_decoder_set_readout_q8"]
    for name, (args, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = args, result
    binding.prepared[key] = api
    return api
