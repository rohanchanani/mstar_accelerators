"""An omitted physical declaration remains an inherited storage requirement."""
from dataclasses import replace

import pytest
import torch

from mstar.engine.resources.binding import resolve_bound_specs
from mstar.engine.resources.kv.config import KVSpec, SequenceKVConfig, RingKVConfig, RingKVLayerConfig
from mstar.engine.resources.sequence import DenseSequenceView, PlainKVEncoding, SequenceBudget, SequenceEncoding
from mstar.engine.resources.spec import ResourceViewConflict
from mstar_accelerators import Registry
from tests.test_registry import binding, declaration


@pytest.mark.parametrize('omitted', [{}, {'cache': ()}])
def test_omitted_consumer_cannot_silently_adopt_siblings_half_cache(omitted):
    spec = KVSpec(resource_key='cache', nodes={'n'},
                  config=SequenceKVConfig(num_layers=1, num_kv_heads=2, head_dim=4, max_seq_len=8),
                  budget=SequenceBudget(3 * 2 * 8 * 2 * 4 * 4))
    events = []
    prefill = declaration('prefill', events=events)
    prefill = replace(prefill, implementation=replace(prefill.implementation, resources=omitted))
    decode = declaration('decode', events=events)
    decode = replace(decode, implementation=replace(decode.implementation,
        resources={'cache': DenseSequenceView(encoding=PlainKVEncoding(torch.float16))}))
    assembly = Registry({'prefill': prefill, 'decode': decode}).assemble(binding(resource_specs=(spec,)))
    assert SequenceEncoding() in assembly.resource_requests['cache']
    assert not events
    with pytest.raises(ResourceViewConflict, match='encoding'):
        resolve_bound_specs((spec,), assembly.resource_requests, dtype=torch.float32)
    # A family explicitly using half in both walks is coherent without changing
    # logical binding dtype or adding a paged-layout constraint to omission.
    prefill = replace(prefill, implementation=replace(prefill.implementation,
        resources={'cache': SequenceEncoding(PlainKVEncoding(torch.float16))}))
    assembly = Registry({'prefill': prefill, 'decode': decode}).assemble(binding(resource_specs=(spec,)))
    resolved, = resolve_bound_specs((spec,), assembly.resource_requests, dtype=torch.float32)
    assert resolved.config.encoding == PlainKVEncoding(torch.float16)
    assert resolved.config.dense


def test_omitted_ring_resource_keeps_its_existing_builder_path():
    spec = KVSpec(resource_key='ring', nodes={'n'},
                  config=RingKVConfig(num_layers=1, num_kv_heads=2, head_dim=4,
                      tokens_per_frame=4, layers=(RingKVLayerConfig(2, 1, 1),)))
    assembly = Registry({'p': declaration('prefill'), 'd': declaration('decode')}).assemble(
        binding(resource_specs=(spec,)))
    assert 'ring' not in assembly.resource_requests
