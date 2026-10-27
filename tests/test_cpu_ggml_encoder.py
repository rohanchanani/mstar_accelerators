"""Physical encoder lifetime/storage controls; not model qualification."""
import ast
import ctypes as C
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from mstar.engine.execution import Support
from mstar_accelerators.implementations.whisper import cpu_ggml as cpu


def binding(weights, value):
    config = SimpleNamespace(d_model=384, encoder_layers=4, encoder_attention_heads=6,
        encoder_ffn_dim=1536, num_mel_bins=80, max_source_positions=1500)
    source = SimpleNamespace(specs=[SimpleNamespace(name=n) for n in weights],
                             load=lambda names: {n: weights[n] for n in names})
    return SimpleNamespace(device=torch.device('cpu'), prepared={},
        model_config={'runtime_extension': {'cpu_ggml': value}},
        model=SimpleNamespace(logical_weights=lambda: source), submodule=SimpleNamespace(config=config))


class CPUEncoderTests(unittest.TestCase):
    def test_condition_uses_existing_binding_without_loading_optional_code(self):
        condition = cpu.CPUEncoderAvailable()
        self.assertEqual(condition.evaluate({}), Support.UNKNOWN)
        b = binding({}, None)
        self.assertEqual(condition.evaluate({'binding': b}), Support.NO)
        b.model_config['runtime_extension']['cpu_ggml'] = {'library': '/missing/library', 'sha256': '0' * 64}
        with patch.object(C, 'CDLL', side_effect=AssertionError('condition loaded code')):
            self.assertEqual(condition.evaluate({'binding': b}), Support.NO)
            with tempfile.NamedTemporaryFile() as file:
                b.model_config['runtime_extension']['cpu_ggml']['library'] = file.name
                self.assertEqual(condition.evaluate({'binding': b}), Support.YES)
                b.device = torch.device('cuda')
                self.assertEqual(condition.evaluate({'binding': b}), Support.NO)

    def test_artifact_configuration_and_digest_are_explicit(self):
        for value in ({}, {'library': 'relative', 'sha256': '0' * 64},
                      {'library': '/x', 'sha256': 'X' * 64},
                      {'library': '/x', 'sha256': '0' * 64, 'fallback': True}):
            with self.assertRaises(ValueError): cpu.setting(binding({}, value))
        with tempfile.NamedTemporaryFile() as file:
            value = {'library': file.name, 'sha256': '0' * 64}
            with self.assertRaisesRegex(ValueError, 'SHA256'): cpu.load_library(value)

    def test_initialization_failure_preserves_exception_and_has_no_prepared_owner(self):
        b = binding({}, {'library': '/x', 'sha256': '0' * 64})
        original = RuntimeError('native preparation failure')
        with patch.object(cpu, 'Encoder', side_effect=original):
            with self.assertRaises(RuntimeError) as result: cpu.prepare_encoder(b)
        self.assertIs(result.exception, original)
        self.assertFalse(b.prepared)

    def test_preparation_has_one_owner_and_releases_only_after_successful_close(self):
        b = binding({}, {'library': '/x', 'sha256': '0' * 64})
        instance = SimpleNamespace(run=lambda call: call, close=lambda: None)
        with patch.object(cpu, 'Encoder', return_value=instance):
            run, close = cpu.prepare_encoder(b)
            self.assertIs(run, instance.run)
            with self.assertRaisesRegex(RuntimeError, 'already prepared'): cpu.prepare_encoder(b)
        failure = RuntimeError('busy')
        instance.close = lambda: (_ for _ in ()).throw(failure)
        with self.assertRaisesRegex(RuntimeError, 'busy'): close()
        self.assertEqual(list(b.prepared.values()), [instance])
        instance.close = lambda: None
        close(); close()
        self.assertFalse(b.prepared)



@unittest.skipUnless(os.environ.get('MSTAR_CPU_ENCODER_LIBRARY'), 'optional built CPU encoder artifact absent')
class NativeCPUEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ['MSTAR_CPU_ENCODER_LIBRARY'])
        cls.value = {'library': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_native_configuration_errors_return_without_handle(self):
        api = cpu.load_library(self.value)
        config, error = cpu.Config(384, 4, 6, 1536, 80, 1499, 2), C.create_string_buffer(1024)
        self.assertFalse(api.cpu_encoder_create(C.byref(config), None, 0, None, error, len(error)))
        self.assertIn(b'configuration', error.value)
        config.positions = 1500
        self.assertFalse(api.cpu_encoder_create(C.byref(config), None, 0, None, error, len(error)))
        self.assertIn(b'descriptors', error.value)

    def test_native_prepared_weight_copy_owned_outputs_and_storage_rejection(self):
        # Full physical shapes with zero projections and a known final norm bias;
        # this tests native packing/ownership and storage, not model accuracy.
        weights = {}
        def linear(p, inside, outside):
            weights[p + '.weight'] = torch.zeros(outside, inside)
            weights[p + '.bias'] = torch.zeros(outside)
        def norm(p):
            weights[p + '.weight'] = torch.ones(384)
            weights[p + '.bias'] = torch.zeros(384)
        weights['encoder.conv1.weight'] = torch.zeros(384, 80, 3)
        weights['encoder.conv1.bias'] = torch.zeros(384)
        weights['encoder.conv2.weight'] = torch.zeros(384, 384, 3)
        weights['encoder.conv2.bias'] = torch.zeros(384)
        weights['encoder.embed_positions.weight'] = torch.zeros(1500, 384)
        norm('encoder.layer_norm')
        weights['encoder.layer_norm.bias'].fill_(0.25)
        for layer in range(4):
            p = f'encoder.layers.{layer}'
            norm(p + '.self_attn_layer_norm'); norm(p + '.final_layer_norm')
            for projection in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
                linear(p + '.self_attn.' + projection, 384, 384)
            linear(p + '.fc1', 384, 1536); linear(p + '.fc2', 1536, 384)
        b = binding(weights, self.value)
        run, close = cpu.prepare_encoder(b)
        owner = next(iter(b.prepared.values()))
        try:
            # The native copy is immutable even after borrowed creation buffers change.
            weights['encoder.layer_norm.bias'].fill_(7.)
            features, output = torch.zeros(80, 3000), torch.empty(1500, 384)
            error = C.create_string_buffer(1024)
            result = owner.library.cpu_encoder_run(owner.handle, features.data_ptr(), 4,
                output.data_ptr(), output.numel() * 4, error, len(error))
            self.assertNotEqual(result, 0)
            self.assertIn(b'storage', error.value)
            call = SimpleNamespace(inputs=SimpleNamespace(audio_features=(features,)),
                outputs=SimpleNamespace(encoder_states=(output,)))
            self.assertIs(run(call)['encoder_states'][0], output)
            self.assertTrue(torch.equal(output, torch.full_like(output, 0.25)))
            snapshot = output.clone()
            call.outputs.encoder_states = (torch.empty_like(output),)
            features.fill_(-1.)
            run(call)
            self.assertTrue(torch.equal(snapshot, output))
        finally:
            close()
        self.assertFalse(b.prepared)
        with self.assertRaisesRegex(RuntimeError, 'closed'): run(call)


if __name__ == '__main__':
    unittest.main()
