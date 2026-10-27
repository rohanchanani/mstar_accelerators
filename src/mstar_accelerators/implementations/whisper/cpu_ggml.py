"""Optional stateless F32 ggml CPU encoder, with one prepared physical owner."""
from collections.abc import Mapping
import ctypes as C
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path

import torch

from mstar.engine.execution import Support
from mstar_accelerators.conditions import Constant

logger = logging.getLogger(__name__)


def setting(binding):
    value = binding.model_config.get('runtime_extension', {}).get('cpu_ggml')
    if value is None:
        return None
    if (not isinstance(value, Mapping) or set(value) != {'library', 'sha256'}
            or not isinstance(value['library'], str) or not Path(value['library']).is_absolute()
            or not isinstance(value['sha256'], str) or len(value['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in value['sha256'])):
        raise ValueError('cpu_ggml requires an absolute library and lowercase SHA256')
    return dict(value)


def native_threads(binding):
    """Prepare a physical GGML pool independently of Torch's worker policy."""
    extension = binding.model_config.get('runtime_extension', {})
    if 'cpu_ggml_threads' not in extension:
        result, origin = torch.get_num_threads(), 'torch_default'
    else:
        mapping = extension['cpu_ggml_threads']
        if (not isinstance(mapping, Mapping) or any(
                not isinstance(node, str) or not node or type(count) is not int or not 1 <= count <= 64
                for node, count in mapping.items())):
            raise ValueError('cpu_ggml_threads requires node names mapped to integers in [1, 64]')
        if binding.node_name in mapping:
            result, origin = mapping[binding.node_name], 'provider_config'
        else:
            result, origin = torch.get_num_threads(), 'torch_default'
    if type(result) is not int or not 1 <= result <= 64:
        raise ValueError('native GGML thread count must be an integer in [1, 64]')
    logger.info('CPU_GGML_NATIVE_THREAD_POLICY threads=%d source=%s', result, origin)
    return result


@dataclass(frozen=True)
class CPUEncoderAvailable:
    def evaluate(self, facts):
        binding = facts.get('binding')
        if binding is None:
            return Support.UNKNOWN
        value = setting(binding)
        return (Support.YES if binding.device.type == 'cpu' and value is not None
                and Path(value['library']).is_file() else Support.NO)

    def specialize(self, facts):
        result = self.evaluate(facts)
        return self if result is Support.UNKNOWN else Constant(result)


class Config(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ('width', 'layers', 'heads', 'ffn', 'mels', 'positions', 'threads')]


class Weight(C.Structure):
    _fields_ = [('name', C.c_char_p), ('data', C.c_void_p), ('bytes', C.c_uint64),
                ('rank', C.c_int32), ('ne', C.c_int64 * 3)]


class Memory(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ('weights', 'graph', 'metadata', 'scratch')]


def load_library(value):
    path = Path(value['library']).resolve(strict=True)
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != value['sha256']:
            raise ValueError('CPU encoder artifact SHA256 differs')
    library = C.CDLL(str(path))
    library.cpu_encoder_abi.argtypes = []
    library.cpu_encoder_abi.restype = C.c_int
    if library.cpu_encoder_abi() != 1:
        raise ValueError('CPU encoder artifact ABI differs')
    library.cpu_encoder_create.argtypes = [C.POINTER(Config), C.POINTER(Weight), C.c_size_t,
                                         C.POINTER(Memory), C.c_void_p, C.c_size_t]
    library.cpu_encoder_create.restype = C.c_void_p
    library.cpu_encoder_run.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p,
                                      C.c_uint64, C.c_void_p, C.c_size_t]
    library.cpu_encoder_run.restype = C.c_int
    library.cpu_encoder_close.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    library.cpu_encoder_close.restype = C.c_int
    return library


class Encoder:
    def __init__(self, binding, value):
        threads = native_threads(binding)
        self.library = load_library(value)
        self.handle = None
        config = binding.submodule.config
        self.shape = (config.max_source_positions, config.d_model)
        self.input_shape = (config.num_mel_bins, 2 * config.max_source_positions)
        physical = Config(config.d_model, config.encoder_layers, config.encoder_attention_heads,
                          config.encoder_ffn_dim, config.num_mel_bins, config.max_source_positions,
                          threads)
        source = binding.model.logical_weights()
        logical = source.load(spec.name for spec in source.specs if spec.name.startswith('encoder.'))
        # These borrowed descriptors survive create, which copies all values into
        # the sole immutable ggml representation. No second checkpoint is loaded.
        tensors = {n: t.detach().to(device='cpu', dtype=torch.float32).contiguous() for n, t in logical.items()}
        descriptors = (Weight * len(tensors))()
        for index, (name, tensor) in enumerate(sorted(tensors.items())):
            if tensor.ndim not in (1, 2, 3):
                raise ValueError('CPU encoder weight rank differs')
            descriptors[index] = Weight(name.encode(), tensor.data_ptr(), tensor.numel() * tensor.element_size(),
                                        tensor.ndim, (C.c_int64 * 3)(*reversed(tensor.shape)))
        memory, error = Memory(), C.create_string_buffer(1024)
        self.handle = self.create_handle(physical, descriptors, memory, error)
        if not self.handle:
            raise RuntimeError('CPU encoder preparation: ' + error.value.decode())
        logger.info('CPU_GGML_ENCODER_READY library=%s sha256=%s threads=%d weights_bytes=%d '
                    'workspace_bytes=%d metadata_bytes=%d scratch_bytes=%d input_copy_bytes=%d output_copy_bytes=%d',
                    value['library'], value['sha256'], physical.threads, memory.weights, memory.graph,
                    memory.metadata, memory.scratch, 2 * physical.positions * physical.mels * 4,
                    physical.positions * physical.width * 4)

    def create_handle(self, physical, descriptors, memory, error):
        return self.library.cpu_encoder_create(C.byref(physical), descriptors, len(descriptors),
                                                C.byref(memory), error, len(error))

    def run(self, call):
        if self.handle is None:
            raise RuntimeError('CPU encoder is closed')
        features, = call.inputs.audio_features
        states, = call.outputs.encoder_states
        if (features.device.type != 'cpu' or features.dtype != torch.float32
                or tuple(features.shape) != self.input_shape
                or states.device.type != 'cpu' or states.dtype != torch.float32
                or tuple(states.shape) != self.shape or not states.is_contiguous()):
            raise ValueError('CPU encoder requires its generated F32 input/output contract')
        features = features.contiguous()
        error = C.create_string_buffer(1024)
        # Includes mandatory input staging, all CPU graph work, and final copy
        # directly into the generated owned output. No workspace alias escapes.
        status = self.library.cpu_encoder_run(self.handle, features.data_ptr(), features.numel() * 4,
                                             states.data_ptr(), states.numel() * 4, error, len(error))
        if status:
            raise RuntimeError('CPU encoder execution: ' + error.value.decode())
        return {'encoder_states': (states,)}

    def close(self):
        if self.handle is not None:
            error = C.create_string_buffer(1024)
            if self.library.cpu_encoder_close(self.handle, error, len(error)):
                raise RuntimeError('CPU encoder close: ' + error.value.decode())
            self.handle = None


def prepare_encoder(binding):
    value = setting(binding)
    if value is None:
        raise ValueError('explicit CPU encoder artifact is missing')
    key = (prepare_encoder, 'encoder')
    if key in binding.prepared:
        raise RuntimeError('CPU encoder already prepared for this binding')
    encoder = Encoder(binding, value)
    binding.prepared[key] = encoder

    def close():
        encoder.close()
        if binding.prepared.get(key) is encoder:
            binding.prepared.pop(key)
    return encoder.run, close
