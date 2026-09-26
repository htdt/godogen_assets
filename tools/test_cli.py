"""Installed-launcher contract checks, no model generation: python3 tools/test_cli.py."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def run_cli(self, command, *args, env=None):
        return subprocess.run([str(ROOT / 'bin' / command), *map(str, args)], capture_output=True, text=True,
                              cwd='/tmp', env=env, timeout=60)

    def test_usage_errors_are_one_json_line_and_exit_one(self):
        for command in ('add-moves', 'gen-moves', 'gen3d', 'mia-rig', 'qwen-image', 'qwen-tts', 'stable-audio', 'lipsync'):
            with self.subTest(command=command):
                result = self.run_cli(command, '--invalid-option', '--json')
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(len(result.stdout.splitlines()), 1, result.stdout)
                self.assertIn('error', json.loads(result.stdout))

    def test_caller_python_environment_is_ignored(self):
        env = {**os.environ, 'PYTHONHOME': '/nonexistent', 'PYTHONPATH': '/nonexistent', 'VIRTUAL_ENV': '/nonexistent'}
        for command in ('add-moves', 'gen3d', 'mia-rig', 'qwen-image', 'qwen-tts', 'stable-audio'):
            with self.subTest(command=command):
                result = self.run_cli(command, '--help', env=env)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_blender_exception_and_progress_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp, 'check.py')
            script.write_text("print('progress')\nasset_result({'output': 'result.glb'})\n")
            result = self.run_cli('asset-blender', script, '--', '--json')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {'output': 'result.glb'})
            self.assertIn('progress', result.stderr)
            script.write_text("raise ValueError('broken animation')\n")
            result = self.run_cli('asset-blender', script, '--', '--json')
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(result.stdout), {'error': 'broken animation'})


if __name__ == '__main__':
    unittest.main()
