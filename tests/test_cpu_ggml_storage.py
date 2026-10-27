"""Focused native storage ABI/borrowed-byte controls; not model qualification."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from mstar.engine.resources.sequence import PlainKVEncoding
from mstar.model.whisper.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, POS
from tests.native_decoder_controls import cpu_ggml_decoder as old
from tests.native_decoder_controls import cpu_ggml_storage as typed
from tests.test_cpu_ggml_decoder import Access, body_weights


class EncodedAccess(Access):
    def __init__(self, capacity, dtype):
        self.dtype = dtype
        self.config = NS(encoding=PlainKVEncoding(dtype), max_seq_len=capacity,
                         num_layers=4, num_kv_heads=6, head_dim=64)
        super().__init__(capacity)

    def reset(self, value):
        self.keys = tuple(torch.full((self.capacity, 6, 64), value, dtype=self.dtype) for _ in range(4))
        self.values = tuple(torch.full_like(t, value) for t in self.keys)

    def view(self, request):
        assert request.encoding == self.config.encoding
        return self


class PlainBody(typed.StorageBody):
    quantized = False


def test_half_family_preserves_logical_reference_support_and_shared_body():
    resources = {key: NS(config=NS(encoding=PlainKVEncoding(torch.float16)))
                 for key in (KV_CACHE, CROSS_KV_CACHE)}
    binding, closed = NS(prepared={}), []
    owner = NS(users=0, run=lambda call: call, close=lambda: closed.append(True))
    with patch.object(typed, 'StorageBody', return_value=owner) as create:
        _, first = typed.prepare_body(binding, {}, resources)
        _, second = typed.prepare_body(binding, {}, resources)
    assert create.call_count == 1
    first(); first()
    assert not closed and owner.users == 1
    second(); second()
    assert closed == [True] and not binding.prepared


@pytest.mark.skipif(not os.environ.get('MSTAR_CPU_DECODER_LIBRARY'), reason='optional native body artifact absent')
@pytest.mark.parametrize('dtype,body_type', [
    pytest.param(torch.float32, PlainBody, id='f32-plain'),
    pytest.param(torch.float16, PlainBody, id='f16-plain'),
    pytest.param(torch.float16, typed.StorageBody, id='f16-q8'),
])
@pytest.mark.parametrize('cross_threads', [None, 4])
def test_typed_storage_writes_rebinds_preserves_and_rejects_before_writes(dtype, body_type, cross_threads, tmp_path):
    path = Path(os.environ['MSTAR_CPU_DECODER_LIBRARY'])
    binding = NS(device=torch.device('cpu'), prepared={},
        model_config={'runtime_extension': {'cpu_ggml': {
            'library': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
            'cpu_ggml_cross_threads': cross_threads,
            'cpu_ggml_attention_capture': str(tmp_path / 'actual')}},
        submodule=NS(config=NS(d_model=384, decoder_layers=4, decoder_attention_heads=6,
            decoder_ffn_dim=1536, max_source_positions=1500, max_target_positions=448, scale_embedding=False)))
    if body_type.quantized:
        api = old.library(binding)
        api.cpu_decoder_q8_capability.argtypes = []
        api.cpu_decoder_q8_capability.restype = C.c_int
        if api.cpu_decoder_q8_capability() != 1:
            pytest.skip('configured artifact has no supported Q8 weight provider on this host')
    state = {KV_CACHE: EncodedAccess(448, dtype), CROSS_KV_CACHE: EncodedAccess(1500, dtype)}
    weights = body_weights()
    for layer in range(4):
        for attention in ('self_attn', 'encoder_attn'):
            weights[f'decoder.layers.{layer}.{attention}.k_proj.bias'].fill_(.125123)
            weights[f'decoder.layers.{layer}.{attention}.v_proj.bias'].fill_(.456789)
    owner = body_type(binding, weights, state)

    def call(walk, q, length):
        state[KV_CACHE].plan(q, length); state[CROSS_KV_CACHE].plan(q, 1500)
        state[CROSS_KV_CACHE].write.positions = torch.arange(1500, dtype=torch.int64)
        plans = NS(**{POS: {'main': torch.arange(length - q, length)},
            KV_CACHE: {'main': None}, ATTN: {'main': None},
            CROSS_KV_CACHE: {'main': None}, CROSS_ATTN: {'main': None}})
        return NS(graph_walk=walk, inputs=NS(input_ids=(torch.zeros(q, dtype=torch.int64),),
            encoder_states=(torch.zeros(1500, 384),) if q == 4 else ()), plan_results=plans)

    def tensors():
        return tuple(t for resource in state.values() for field in ('keys', 'values') for t in getattr(resource, field))

    def bytes_of(values):
        return tuple(t.contiguous().view(torch.uint8).clone() for t in values)

    try:
        result = owner.run(call('prefill', 4, 4))
        assert torch.equal(result, torch.full((1, 384), .25))
        for resource in state.values():
            length = 4 if resource is state[KV_CACHE] else 1500
            for values, scalar in ((resource.keys, .125123), (resource.values, .456789)):
                for tensor in values:
                    assert torch.equal(tensor[:length], torch.full_like(tensor[:length], scalar))
                    assert torch.equal(tensor[length:], torch.full_like(tensor[length:], 3.))
        cross_before = bytes_of(state[CROSS_KV_CACHE].keys + state[CROSS_KV_CACHE].values)
        decode = call('decode', 1, 5)
        all_before = bytes_of(tensors())
        native = owner.library.cpu_decoder_storage_run

        def bad_final_extent(handle, argument, error, size):
            argument._obj.kv[3].cross_bytes -= 2
            return native(handle, argument, error, size)

        with patch.object(owner.library, 'cpu_decoder_storage_run', side_effect=bad_final_extent):
            with pytest.raises(RuntimeError, match='extent'):
                owner.run(decode)
        assert all(torch.equal(a, b) for a, b in zip(all_before, bytes_of(tensors())))
        assert torch.equal(owner.run(decode), result)
        assert all(torch.equal(a, b) for a, b in zip(cross_before,
            bytes_of(state[CROSS_KV_CACHE].keys + state[CROSS_KV_CACHE].values)))
        metadata = json.loads((tmp_path / 'actual.json').read_text())
        assert metadata['keys'] == 1500 and metadata['queries'] == 1 and metadata['scale'] == .125
        assert (tmp_path / 'actual.k.f32').stat().st_size == 1500 * 384 * 4
        retained, saved = tensors(), bytes_of(tensors())
        for resource in state.values():
            resource.reset(-3.)
        assert torch.equal(owner.run(call('decode', 1, 8)), result)
        assert all(torch.equal(a, b) for a, b in zip(saved, bytes_of(retained)))
        error = C.create_string_buffer(1024)
        assert owner.library.cpu_decoder_run(owner.handle, C.byref(old.NativeCall()), error, len(error)) != 0
        assert b'legacy F32' in error.value
        owner.close()
        assert torch.equal(result, torch.full((1, 384), .25))
    finally:
        owner.close()


@pytest.mark.skipif(not os.environ.get('MSTAR_CPU_DECODER_LIBRARY'), reason='optional native body artifact absent')
def test_unknown_encoding_rejected_before_weight_preparation():
    path = Path(os.environ['MSTAR_CPU_DECODER_LIBRARY'])
    binding = NS(device=torch.device('cpu'), prepared={}, model_config={'runtime_extension': {
        'cpu_ggml': {'library': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}})
    api = typed.library(binding, quantized=False)
    config = old.Config(384, 4, 6, 1536, 1500, 448, 2)
    plan = typed.StoragePlan(999, 2, 448, 1500)
    error = C.create_string_buffer(1024)
    assert not api.cpu_decoder_storage_create(C.byref(config), C.byref(plan), None, 0, error, len(error))
    assert b'unsupported plain KV storage encoding' in error.value
