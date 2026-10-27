"""Source validation and option controls for the optional physical builder."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
import sys
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    'cpu_ggml_build', Path(__file__).resolve().parents[1] / 'backends/cpu_ggml/build.py')
build = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build)


class SourceIdentityTests(unittest.TestCase):
    def test_q8_convolution_selection_reaches_plan_and_cmake(self):
        for selection in (None, 0, 1, 2, 3):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = ['build', '--whisper-source', str(root), '--build', str(root / 'build'), '--plan-only']
                if selection is not None:
                    argv += ['--encoder-q8-conv', str(selection)]
                with patch.object(sys, 'argv', argv), \
                     patch.object(build, 'source_identity', return_value=(build.PIN, 'tree', None)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    build.main()
                plan = json.loads((root / 'build/BUILD_PLAN.json').read_text())
                expected = selection or 0
                self.assertEqual(plan['requested_overlays']['encoder_q8_conv'], expected)
                self.assertIn('-DMSTAR_ENCODER_Q8_CONV=' + str(expected), plan['commands'][0])
                self.assertEqual(plan['status'], 'PLAN_ONLY_NO_COMPILATION')

    def test_invalid_q8_convolution_selection_rejected_before_source_validation(self):
        for selection in ('-1', '4', 'both'):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = ['build', '--whisper-source', str(root), '--build', str(root / 'build'),
                        '--encoder-q8-conv', selection, '--plan-only']
                with patch.object(sys, 'argv', argv), patch.object(build, 'source_identity') as identity, \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    build.main()
                identity.assert_not_called()
                self.assertFalse((root / 'build').exists())

    def test_ffn_selection_reaches_plan_and_cmake(self):
        for selection in (None, 0, 128):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = ['build', '--whisper-source', str(root), '--build', str(root / 'build'), '--plan-only']
                if selection is not None:
                    argv += ['--encoder-ffn-tile', str(selection)]
                with patch.object(sys, 'argv', argv), \
                     patch.object(build, 'source_identity', return_value=(build.PIN, 'tree', None)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    build.main()
                plan = json.loads((root / 'build/BUILD_PLAN.json').read_text())
                expected = selection or 0
                self.assertEqual(plan['requested_overlays']['encoder_ffn_tile'], expected)
                self.assertIn('-DMSTAR_ENCODER_FFN_TILE=' + str(expected), plan['commands'][0])
                self.assertEqual(plan['status'], 'PLAN_ONLY_NO_COMPILATION')

    def test_invalid_ffn_selection_rejected_before_source_validation(self):
        for selection in ('-1', '1', '127', '129', '128.0'):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = ['build', '--whisper-source', str(root), '--build', str(root / 'build'),
                        '--encoder-ffn-tile', selection, '--plan-only']
                with patch.object(sys, 'argv', argv), patch.object(build, 'source_identity') as identity, \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    build.main()
                identity.assert_not_called()
                self.assertFalse((root / 'build').exists())

    def test_inventory_verifies_every_file_before_accepting_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            root = source / 'ggml'
            root.mkdir()
            member = root / 'example.c'
            member.write_bytes(b'original\n')
            inventory = source / 'inventory.json'
            inventory.write_text(json.dumps({
                'whisper_revision': build.PIN, 'ggml_tree': 'tree',
                'files': {'example.c': {'bytes': member.stat().st_size,
                                       'sha256': build.sha(member)}}}))
            with patch.object(build, 'ARCHIVE_INVENTORY', build.sha(inventory)):
                self.assertEqual(build.source_identity(source, inventory),
                                 (build.PIN, 'tree', build.sha(inventory)))
                member.write_bytes(b'modified\n')
                with self.assertRaisesRegex(ValueError, 'source bytes differ'):
                    build.source_identity(source, inventory)
                member.write_bytes(b'original\n')
                (root / 'extra.c').touch()
                with self.assertRaisesRegex(ValueError, 'file inventory differs'):
                    build.source_identity(source, inventory)

    def test_default_source_identity_has_no_physical_option_dependency(self):
        with patch.object(build.subprocess, 'check_output',
                          side_effect=[build.PIN + '\n', '', 'tree\n']), \
             patch.object(build.subprocess, 'run') as run:
            self.assertEqual(build.source_identity(Path('/source'), None),
                             (build.PIN, 'tree', None))
            run.assert_called_once()

    def test_softmax_prerequisite_rejected_before_private_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'requires explicit QK SDOT'):
                build.physical_source(root, root / 'build', False, False,
                                      softmax_tile=True)
            self.assertFalse((root / 'build').exists())
            self.assertEqual(build.physical_source(root, root / 'build', False, False),
                             (root / 'ggml', {}))


if __name__ == '__main__':
    unittest.main()
