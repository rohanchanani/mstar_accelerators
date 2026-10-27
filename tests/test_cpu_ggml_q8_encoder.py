"""Q8 physical capability/ownership controls, not numerical qualification."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from mstar.engine.execution import Support
from mstar_accelerators.implementations.whisper import cpu_ggml_q8 as q8


class Q8EncoderTests(unittest.TestCase):
    def test_optional_condition_does_not_claim_old_f32_only_artifact(self):
        c = q8.CPUQ8EncoderAvailable()
        self.assertEqual(c.evaluate({}), Support.UNKNOWN)
        b = NS(device=torch.device('cpu'), model_config={}, prepared={})
        self.assertEqual(c.evaluate({'binding': b}), Support.NO)
        b.model_config = {'runtime_extension': {'cpu_ggml': {'library': '/x', 'sha256': '0' * 64}}}
        with patch.object(q8, 'load_library', return_value=NS()):
            self.assertEqual(c.evaluate({'binding': b}), Support.NO)
        self.assertFalse(b.prepared)



@unittest.skipUnless(os.environ.get('MSTAR_CPU_ENCODER_LIBRARY'), 'optional Q8 artifact absent')
class NativeQ8Tests(unittest.TestCase):
    def test_real_repack_packing_and_owned_output_survive_close(self):
        path = Path(os.environ['MSTAR_CPU_ENCODER_LIBRARY'])
        weights = {}
        def norm(p):
            weights[p + '.weight'] = torch.ones(384); weights[p + '.bias'] = torch.zeros(384)
        def linear(p, inside, outside):
            weights[p + '.weight'] = torch.zeros(outside, inside); weights[p + '.bias'] = torch.zeros(outside)
        weights['encoder.conv1.weight'] = torch.zeros(384, 80, 3); weights['encoder.conv1.bias'] = torch.zeros(384)
        weights['encoder.conv2.weight'] = torch.zeros(384, 384, 3); weights['encoder.conv2.bias'] = torch.zeros(384)
        weights['encoder.embed_positions.weight'] = torch.zeros(1500, 384)
        norm('encoder.layer_norm'); weights['encoder.layer_norm.bias'].fill_(.25)
        for layer in range(4):
            p = f'encoder.layers.{layer}'
            norm(p + '.self_attn_layer_norm'); norm(p + '.final_layer_norm')
            for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'): linear(p + '.self_attn.' + name, 384, 384)
            linear(p + '.fc1', 384, 1536); linear(p + '.fc2', 1536, 384)
        config = NS(d_model=384, encoder_layers=4, encoder_attention_heads=6,
                    encoder_ffn_dim=1536, num_mel_bins=80, max_source_positions=1500)
        source = NS(specs=[NS(name=n) for n in weights], load=lambda names: {n: weights[n] for n in names})
        b = NS(device=torch.device('cpu'), prepared={}, submodule=NS(config=config),
            model=NS(logical_weights=lambda: source), model_config={'runtime_extension': {'cpu_ggml': {
                'library': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}})
        self.assertEqual(q8.CPUQ8EncoderAvailable().evaluate({'binding': b}), Support.YES)
        run, close = q8.prepare_encoder(b)
        try:
            weights['encoder.layer_norm.bias'].fill_(9.)
            out = torch.empty(1500, 384)
            call = NS(inputs=NS(audio_features=(torch.zeros(80, 3000),)), outputs=NS(encoder_states=(out,)))
            self.assertIs(run(call)['encoder_states'][0], out)
            self.assertTrue(torch.equal(out, torch.full_like(out, .25)))
            close()
            self.assertTrue(torch.equal(out, torch.full_like(out, .25)))
            with self.assertRaisesRegex(RuntimeError, 'closed'): run(call)
        finally: close()


if __name__ == '__main__': unittest.main()
