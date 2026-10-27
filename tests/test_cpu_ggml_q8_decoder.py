"""Q8 physical-owner/capability controls; numerical policy stays central."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from mstar.engine.execution import Support
from tests.native_decoder_controls import cpu_ggml_q8_decoder as q8
from tests import test_cpu_ggml_decoder as controls
from tests.test_cpu_ggml_decoder import cpu


class Q8DecoderTests(unittest.TestCase):
    def test_optional_capability_does_not_claim_old_artifact(self):
        condition = q8.CPUQ8DecoderAvailable()
        self.assertEqual(condition.evaluate({}), Support.UNKNOWN)
        binding = NS(device=torch.device('cpu'), prepared={}, model_config={})
        self.assertEqual(condition.evaluate({'binding': binding}), Support.NO)
        binding.model_config = {'runtime_extension': {'cpu_ggml': {'library': '/old', 'sha256': '0' * 64}}}
        with patch.object(q8, 'body_library', return_value=NS()):
            self.assertEqual(condition.evaluate({'binding': binding}), Support.NO)

    def test_both_books_share_one_physical_pack_and_retry_failed_close(self):
        binding, events = NS(prepared={}), []
        owner = NS(users=0, run=lambda c: c, close=lambda: events.append('close'))
        with patch.object(q8, 'library', return_value=True), patch.object(q8, 'Q8Body', return_value=owner) as constructor:
            first, close_first = q8.prepare_body(binding, {}, {})
            second, close_second = q8.prepare_body(binding, {}, {})
        self.assertEqual(constructor.call_count, 1)
        self.assertIs(first, second)
        close_first(); close_first()
        self.assertEqual(owner.users, 1)
        owner.close = lambda: (_ for _ in ()).throw(RuntimeError('busy'))
        with self.assertRaisesRegex(RuntimeError, 'busy'): close_second()
        self.assertEqual(owner.users, 1)
        owner.close = lambda: events.append('close')
        close_second(); close_second()
        self.assertFalse(binding.prepared)
        self.assertEqual(events, ['close'])


    @unittest.skipUnless(os.environ.get('MSTAR_CPU_DECODER_LIBRARY'), 'optional native artifact absent')
    def test_native_borrowed_rebinding_survivors_refusal_and_allmasked(self):
        # Reuse the exact existing CABI contract control, changing only physical
        # creation. This is not an alternative model numerical oracle.
        with patch.object(cpu, 'Body', q8.Q8Body):
            controls.NativeBodyTests('test_borrowed_cache_rebinding_mask_refusal_and_surviving_rows').test_borrowed_cache_rebinding_mask_refusal_and_surviving_rows()


if __name__ == '__main__': unittest.main()
