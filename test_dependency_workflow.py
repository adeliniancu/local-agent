"""Prevent the interpreter/package mixing seen in the first shared version."""
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from agent_core import Agent


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.agent = Agent(self.root, lambda *_: None, lambda *_: True, threading.Event())

    def tearDown(self):
        self.temp.cleanup()

    def make_environment(self):
        target = self.root / '.venv'
        (target / 'Scripts').mkdir(parents=True)
        (target / 'Scripts' / 'python.exe').write_bytes(b'test')
        (target / 'pyvenv.cfg').write_text('home = example')
        return target

    def test_install_requires_project_environment(self):
        with self.assertRaisesRegex(ValueError, 'setup_python_environment'):
            self.agent._check_dependency_command('python -m pip install pywin32')

    def test_targeting_runtime_is_refused(self):
        self.make_environment()
        for command in ('python -m pip install pywin32 --target runtime/python/Lib',
                        'python -m pip install pywin32 -t runtime/python/Lib'):
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.agent._check_dependency_command(command)

    def test_matching_interpreter_install_is_allowed(self):
        self.make_environment()
        self.agent._check_dependency_command('python -m pip install pywin32')
        with self.assertRaisesRegex(ValueError, 'same interpreter'):
            self.agent._check_dependency_command('pip install pywin32')

    def test_project_python_takes_priority(self):
        env_dir = self.make_environment()
        env = self.agent._command_environment(self.root)
        self.assertEqual(env['PATH'].split(os.pathsep)[0], str(env_dir / 'Scripts'))
        self.assertEqual(env['VIRTUAL_ENV'], str(env_dir))

    def test_existing_non_environment_not_overwritten(self):
        (self.root / '.venv').mkdir()
        with self.assertRaisesRegex(ValueError, 'Do not overwrite'):
            self.agent._setup_python_environment()

    def test_reuse_does_not_recreate_environment(self):
        env_dir = self.make_environment()
        with patch.object(self.agent, '_run_command') as run:
            result = self.agent._setup_python_environment()
        run.assert_not_called()
        self.assertEqual(result['python'], str(env_dir / 'Scripts' / 'python.exe'))

    def test_three_matching_errors_pause_and_success_resets(self):
        for _ in range(3):
            self.agent._record_command_result({'exit_code': 1, 'output': "ModuleNotFoundError: No module named 'pywintypes'"})
        self.assertTrue(self.agent._stuck)
        self.agent._record_command_result({'exit_code': 0, 'output': 'ok'})
        self.assertEqual(self.agent._command_failures, {})


if __name__ == '__main__':
    unittest.main()
