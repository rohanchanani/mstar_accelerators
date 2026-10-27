"""Mechanism checks for the explicit preparation-only Q8 readout format."""
from types import SimpleNamespace
import struct

import pytest
import torch

from mstar_accelerators.implementations.whisper.pi_v3d.q8 import (
    Q8ReadoutProvider, READOUT_Q8_0, READOUT_WEIGHT, pack_q8_0, q8_readout,
)


def test_q8_0_exact_block_bytes_and_roundf_ties():
    # Exact scale=1, halfway cases in both directions, then a zero block.
    values = [127, -127, .5, -.5, 1.5, -1.5, 2.5, -2.5, 126.5, -126.5] + [0] * 22
    weight = torch.tensor([values, [0] * 32], dtype=torch.float32)
    packed = pack_q8_0(weight)
    expected = struct.pack('<e32b', 1, 127, -127, 1, -1, 2, -2, 3, -3, 127, -127, *([0] * 22))
    expected += bytes(34)
    assert packed.dtype == torch.int32 and packed.shape == (17,)
    assert packed.is_contiguous() and packed.data_ptr() % 4 == 0
    assert bytes(packed.view(torch.uint8).tolist()) == expected


def test_q8_0_uses_fp32_scale_for_quantization_and_fp16_only_for_storage():
    weight = torch.zeros((2, 32), dtype=torch.float32)
    weight[:, :4] = torch.tensor([63.5, -63.5, .25, -.25])
    assert bytes(pack_q8_0(weight).view(torch.uint8).tolist()) == (
        struct.pack('<e32b', .5, 127, -127, 1, -1, *([0] * 28)) * 2
    )
    # The FP16 scale underflows, while integers still use its FP32 precursor.
    weight.zero_()
    weight[:, 0] = 1e-12
    assert bytes(pack_q8_0(weight).view(torch.uint8).tolist()) == (
        struct.pack('<e32b', 0, 127, *([0] * 31)) * 2
    )


@pytest.mark.parametrize('weight', [
    torch.zeros((1, 32)), torch.zeros((2, 31)), torch.zeros((2, 32), dtype=torch.float64),
    torch.full((2, 32), float('nan')), torch.full((2, 32), float('inf')),
])
def test_q8_0_rejects_invalid_storage_or_nonfinite_values(weight):
    with pytest.raises(ValueError):
        pack_q8_0(weight)


def test_provider_reuses_binding_owned_representation_and_preserves_f32_constants(monkeypatch):
    loads, preparations = [], []
    logical = torch.ones((2, 32), dtype=torch.float32)

    def load(names):
        loads.append(names)
        return {READOUT_WEIGHT: logical}

    class Native:
        def prepare(self, builder, constants, *, slots):
            preparations.append((builder, constants, slots))
            return constants

    monkeypatch.setattr('mstar_accelerators.implementations.whisper.pi_v3d.q8.NativeGraphProvider', Native)
    binding = SimpleNamespace(prepared={}, device='cpu', model=SimpleNamespace(
        logical_weights=lambda: SimpleNamespace(load=load)))
    first, second = Q8ReadoutProvider(binding), Q8ReadoutProvider(binding)
    constants = {'decoder.embed_tokens.weight': logical, READOUT_WEIGHT: logical}
    result = first.prepare('prefill', constants, slots=2)
    other = second.prepare('decode', constants, slots=2)
    assert loads == [(READOUT_WEIGHT,)]
    assert result[READOUT_Q8_0] is other[READOUT_Q8_0]
    assert result[READOUT_WEIGHT] is logical and result['decoder.embed_tokens.weight'] is logical
    assert READOUT_Q8_0 not in constants
    assert preparations[0][2] == preparations[1][2] == 2


def test_readout_explicitly_quantizes_last_query_and_dots_q8_weights():
    calls = []

    class Graph:
        def tensor(self, name, shape, *, dtype):
            calls.append((name, shape, dtype))
            return 'weight'

        def quantize_q8_1_x4(self, value):
            calls.append(('quantize', value))
            return 'activation'

        def q8_0_dot_x4(self, weight, activation):
            calls.append(('dot', weight, activation))
            return 'logits'

    assert q8_readout(Graph(), SimpleNamespace(d_model=384, vocab_size=51865), 'last') == 'logits'
    assert calls == [(READOUT_Q8_0, (384, 51865), 'q8_0'),
                     ('quantize', 'last'), ('dot', 'weight', 'activation')]
