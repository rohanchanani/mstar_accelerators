"""Physical borrowed-storage/body-owner controls, not model qualification."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from mstar.engine.execution import Support
from mstar.model.whisper.config import ATTN, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, POS
from tests.native_decoder_controls import cpu_ggml_decoder as cpu


class BodyOwnerTests(unittest.TestCase):
    def test_optional_capability_is_unknown_before_binding_and_rejects_missing_symbol(self):
        condition = cpu.CPUDecoderAvailable()
        self.assertEqual(condition.evaluate({}), Support.UNKNOWN)
        b = NS(device=torch.device('cpu'), prepared={}, model_config={})
        self.assertEqual(condition.evaluate({'binding': b}), Support.NO)
        b.model_config = {'runtime_extension': {'cpu_ggml': {'library': '/artifact', 'sha256': '0' * 64}}}
        with patch.object(cpu, 'load_library', return_value=NS()):
            self.assertEqual(condition.evaluate({'binding': b}), Support.NO)
        self.assertFalse(b.prepared)

    def test_two_books_share_one_physical_body_and_close_after_last_user(self):
        b, calls = NS(prepared={}), []
        owner = NS(users=0, run=lambda call: call, close=lambda: calls.append('close'))
        with patch.object(cpu, 'Body', return_value=owner) as constructor:
            first, close_first = cpu.prepare_body(b, {}, {})
            second, close_second = cpu.prepare_body(b, {}, {})
        self.assertEqual(constructor.call_count, 1)
        self.assertIs(first, second)
        close_first(); close_first()
        self.assertEqual(owner.users, 1); self.assertFalse(calls)
        close_second(); close_second()
        self.assertEqual(calls, ['close']); self.assertFalse(b.prepared)

    def test_failed_native_close_keeps_owner_for_diagnosis_and_retry(self):
        b = NS(prepared={})
        owner = NS(users=0, run=lambda call: call, close=lambda: (_ for _ in ()).throw(RuntimeError('busy')))
        with patch.object(cpu, 'Body', return_value=owner): run, close = cpu.prepare_body(b, {}, {})
        with self.assertRaisesRegex(RuntimeError, 'busy'): close()
        self.assertEqual(owner.users, 1); self.assertIn(owner, b.prepared.values())
        owner.close = lambda: None
        close(); self.assertFalse(b.prepared)




class Access:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset(3.)
    def reset(self, value):
        self.keys = tuple(torch.full((self.capacity, 6, 64), value) for _ in range(4))
        self.values = tuple(torch.full_like(t, value) for t in self.keys)
    def view(self, request): return self
    def for_step(self, use): return (self.write,)
    def for_attention(self, use): return (self.read,)
    def plan(self, q, length):
        positions = torch.arange(length - q, length, dtype=torch.int64)
        self.write = NS(request_id='row', positions=positions, keys=self.keys, values=self.values)
        mask = torch.zeros(q, self.capacity).masked_fill(torch.arange(self.capacity)[None, :] >= length, -float('inf'))
        self.read = NS(request_id='row', valid_length=length, mask=mask, keys=self.keys, values=self.values)


def body_weights():
    weights = {'decoder.embed_tokens.weight': torch.zeros(1, 384), 'decoder.embed_positions.weight': torch.zeros(448, 384)}
    def norm(p):
        weights[p + '.weight'] = torch.ones(384); weights[p + '.bias'] = torch.zeros(384)
    def linear(p, inside, outside):
        weights[p + '.weight'] = torch.zeros(outside, inside); weights[p + '.bias'] = torch.zeros(outside)
    norm('decoder.layer_norm'); weights['decoder.layer_norm.bias'].fill_(.25)
    for layer in range(4):
        p = f'decoder.layers.{layer}'
        for name in ('self_attn_layer_norm', 'encoder_attn_layer_norm', 'final_layer_norm'): norm(p + '.' + name)
        for name in ('self_attn', 'encoder_attn'):
            for projection in ('q_proj', 'k_proj', 'v_proj', 'out_proj'): linear(p + '.' + name + '.' + projection, 384, 384)
        linear(p + '.fc1', 384, 1536); linear(p + '.fc2', 1536, 384)
    return weights


@unittest.skipUnless(os.environ.get('MSTAR_CPU_DECODER_LIBRARY'), 'optional built decoder body artifact absent')
class NativeBodyTests(unittest.TestCase):
    def test_borrowed_cache_rebinding_mask_refusal_and_surviving_rows(self):
        path = Path(os.environ['MSTAR_CPU_DECODER_LIBRARY'])
        binding = NS(device=torch.device('cpu'), prepared={},
            model_config={'runtime_extension': {'cpu_ggml': {'library': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}},
            submodule=NS(config=NS(d_model=384, decoder_layers=4, decoder_attention_heads=6,
                decoder_ffn_dim=1536, max_source_positions=1500, max_target_positions=448, scale_embedding=False)))
        state = {KV_CACHE: Access(448), CROSS_KV_CACHE: Access(1500)}
        owner = cpu.Body(binding, body_weights(), state)
        def call(walk, q, length):
            state[KV_CACHE].plan(q, length); state[CROSS_KV_CACHE].plan(q, 1500)
            state[CROSS_KV_CACHE].write.positions = torch.arange(1500, dtype=torch.int64)
            plan = NS(**{POS: {'main': torch.arange(length - q, length)},
                KV_CACHE: {'main': None}, ATTN: {'main': None}, CROSS_KV_CACHE: {'main': None}, CROSS_ATTN: {'main': None}})
            return NS(graph_walk=walk, inputs=NS(input_ids=(torch.zeros(q, dtype=torch.int64),),
                encoder_states=(torch.zeros(1500, 384),) if q == 4 else ()), plan_results=plan)
        def tensors(): return tuple(t for s in state.values() for name in ('keys', 'values') for t in getattr(s, name))
        try:
            prefill = call('prefill', 4, 4)
            result = owner.run(prefill)
            self.assertTrue(torch.equal(result, torch.full((1, 384), .25)))
            for t in state[KV_CACHE].keys + state[KV_CACHE].values:
                self.assertFalse(torch.count_nonzero(t[:4])); self.assertTrue(torch.equal(t[4:], torch.full_like(t[4:], 3.)))
            # Bad metadata in the last range is rejected before the first SET_ROWS.
            decode = call('decode', 1, 5); before = tuple(t.clone() for t in tensors())
            state[CROSS_KV_CACHE].read.mask[0, -1] = float('nan')
            with self.assertRaisesRegex(RuntimeError, 'mask'): owner.run(decode)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, tensors())))
            # All-masked attention remains finite/zero without reading old state.
            state[CROSS_KV_CACHE].read.mask.fill_(-float('inf'))
            self.assertTrue(torch.equal(owner.run(decode), torch.full((1, 384), .25)))
            old = tensors(); snapshot = tuple(t.clone() for t in old)
            for s in state.values(): s.reset(-3.)
            decode = call('decode', 1, 8)
            self.assertTrue(torch.equal(owner.run(decode), torch.full((1, 384), .25)))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, snapshot)))
            for t in state[KV_CACHE].keys + state[KV_CACHE].values:
                self.assertFalse(torch.count_nonzero(t[7])); self.assertTrue(torch.equal(t[:7], torch.full_like(t[:7], -3.)))
            # Decode never rewrites even a single cross-cache value.
            self.assertTrue(all(torch.equal(t, torch.full_like(t, -3.)) for t in state[CROSS_KV_CACHE].keys + state[CROSS_KV_CACHE].values))
            owner.close()
            self.assertTrue(torch.equal(result, torch.full((1, 384), .25)))
            with self.assertRaisesRegex(RuntimeError, 'closed'): owner.run(decode)
        finally: owner.close()


if __name__ == '__main__': unittest.main()
