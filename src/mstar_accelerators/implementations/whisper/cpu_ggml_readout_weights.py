"""Explicit immutable readout formats within the existing binding weight owner."""
import ctypes as C

import torch

from mstar_accelerators.backends.weights import prepare_weights
from .cpu_ggml_decoder_abi import checked, size, library
from .cpu_q8 import prepared_weights as q8_weights
from .torch import column_major_weights


def pack_q4(binding, logical):
    if logical.ndim != 2 or logical.shape[1] % 32:
        raise ValueError('Q4 readout needs a matrix with block-aligned width')
    rows, width = logical.shape
    source = checked(logical, (rows, width))
    byte_count = rows * (width // 32) * 18
    if byte_count % 4:
        raise ValueError('Q4 readout extent must be int32 aligned')
    api = library(binding, readout_format='q4_0')
    if api is None:
        raise ValueError('configured artifact lacks explicit Q4 readout preparation')
    packed = torch.empty(byte_count // 4, dtype=torch.int32, device='cpu')
    error = C.create_string_buffer(1024)
    if api.cpu_decoder_pack_readout_q4(source.data_ptr(), size(source), packed.data_ptr(), size(packed),
                                     rows, width, error, len(error)):
        raise RuntimeError('CPU Q4 readout packing: ' + error.value.decode())
    return packed


def prepared_weights(binding, weight_prefix, *, readout_format='q8_0'):
    if readout_format == 'q8_0':
        return q8_weights(binding, weight_prefix)
    if readout_format not in ('q4_0', 'q4_0_kai'):
        raise ValueError('unsupported physical readout format')
    key = (prepared_weights, weight_prefix, readout_format)
    if key not in binding.prepared:
        weights = prepare_weights(binding, weight_prefix, dtype=torch.float32)
        embedding, projection = 'decoder.embed_tokens.weight', 'decoder.proj_out.weight'
        source_body = {name: weight for name, weight in weights.items() if name not in (embedding, projection)}
        body = column_major_weights(binding, source_body) | {embedding: weights[embedding]}
        packed = pack_q4(binding, weights[projection])
        suppressed = {walk: torch.tensor(binding.submodule.config.suppressed_tokens(walk),
                        dtype=torch.int64, device=binding.device) for walk in ('prefill', 'decode')}
        binding.prepared[key] = (source_body, body, packed, suppressed, weights[projection].shape)
    return binding.prepared[key]
