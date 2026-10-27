"""Physical builder controls; operator/model math remains separately tested."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / 'backends/cpu_ggml'
def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
build=load('build');overlay=load('flash_cross_q1')
ORIGINAL='static void ggml_compute_forward_flash_attn_ext_f16_one_chunk(\n) {\n    const bool write_partials = true;\n}\n'

class CrossQ1BuildTests(unittest.TestCase):
    def test_default_does_not_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(sys,'argv',['build','--whisper-source',str(root),'--build',str(root/'out'),'--plan-only']), patch.object(build,'source_identity',return_value=(build.PIN,'tree',None)), contextlib.redirect_stdout(io.StringIO()):
                build.main()
            plan=json.loads((root/'out/BUILD_PLAN.json').read_text())
            self.assertFalse(plan['requested_overlays']['decoder_cross_q1'])
            self.assertFalse(plan['physical_source_patches'])
    def test_enabled_plan_copies_local_stable_horner(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cpu=root/'ggml/src/ggml-cpu';cpu.mkdir(parents=True);(cpu/'ops.cpp').write_text(ORIGINAL)
            with patch.object(sys,'argv',['build','--whisper-source',str(root),'--build',str(root/'out'),'--decoder-cross-q1','--plan-only']), patch.object(build,'source_identity',return_value=(build.PIN,'tree',None)), patch.object(sys,'path',[str(HERE),*sys.path]),contextlib.redirect_stdout(io.StringIO()):
                build.main()
            plan=json.loads((root/'out/BUILD_PLAN.json').read_text());self.assertTrue(plan['requested_overlays']['decoder_cross_q1'])
            patched=Path(plan['physical_source'])/'src/ggml-cpu'
            self.assertEqual((patched/'flash_cross_exp_horner.h').read_bytes(),(HERE/'flash_cross_exp_horner.h').read_bytes())
            self.assertEqual((cpu/'ops.cpp').read_text(),ORIGINAL)
            self.assertNotIn('op_params[15]',(patched/'ops.cpp').read_text())
    def test_production_and_diagnostic_selectors_are_distinct(self):
        production=overlay.ops_overlay(ORIGINAL);diagnostic=overlay.ops_overlay(ORIGINAL,diagnostic=True)
        self.assertNotIn('mstar_cross_fast',production);self.assertNotIn('op_params[15]',production)
        self.assertIn('#define MSTAR_CROSS_Q1_DIAGNOSTIC 1',diagnostic)
        self.assertIn('op_params[15] == 1',diagnostic)
    def test_duplicate_or_wrong_entry_rejected(self):
        for text in ['',ORIGINAL*2,overlay.ops_overlay(ORIGINAL)]:
            with self.assertRaises(ValueError):overlay.ops_overlay(text)

if __name__=='__main__':unittest.main()
