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

    def callback_result(self, marker='Exception in Tkinter callback', diagnostic='NameError: missing_handler'):
        return {'exit_code': 0, 'output': (
            marker + '\nTraceback (most recent call last):\n'
            '  File "app.py", line 2, in callback\n    missing_handler()\n' + diagnostic + '\n')}

    def test_exit_zero_callback_and_thread_failures_survive_unrelated_successes(self):
        (self.root / 'app.py').write_text('missing_handler()\n')
        for index, marker in enumerate(('Exception in Tkinter callback', 'Exception in thread Worker-1:', 'Exception in Tkinter callback')):
            result = self.callback_result(marker)
            self.agent._record_command_result(result, 'python app.py')
            self.assertEqual(result['exit_code'], 0)
            self.assertIn('traceback', result['error'])
            self.assertEqual(self.agent._stuck, index == 2)
            self.agent._record_command_result({'exit_code': 0, 'output': '["press", "release"]'}, 'python -c "print(dir(object))"')
        self.assertIn('paused', result)

    def test_clean_same_check_resolves_only_its_failure_counts(self):
        for _ in range(2):
            self.agent._record_command_result(self.callback_result(), 'python app.py')
        self.agent._record_command_result({'exit_code': 0, 'output': 'all callbacks passed'}, 'python app.py')
        self.agent._record_command_result(self.callback_result(), 'python app.py')
        self.assertFalse(self.agent._stuck)
        self.assertEqual(list(self.agent._command_failures.values()), [1])

    def test_successful_log_reads_help_and_plain_traceback_examples_are_not_failures(self):
        examples = [
            ('Get-Content crash.log', self.callback_result()['output']),
            ('rg Exception crash.log', self.callback_result()['output']),
            ('python -c "help(object)"', ':raises ValueError: if bad input\n'),
            ('python examples.py', 'Traceback (most recent call last):\nValueError: documented example\n'),
        ]
        for command, output in examples:
            result = {'exit_code': 0, 'output': output}
            self.agent._record_command_result(result, command)
            self.assertNotIn('error', result)
        self.assertEqual(self.agent._command_failures, {})

    def test_relevant_source_change_allows_retry_but_logs_and_other_sources_do_not(self):
        source = self.root / 'app.py'
        source.write_text('missing_handler()\n')
        for index in range(2):
            self.agent._record_command_result(self.callback_result(), 'python app.py')
            (self.root / 'diagnostic.log').write_text(str(index))
            (self.root / 'unrelated.py').write_text(str(index))
            cache = self.root / '__pycache__'
            cache.mkdir(exist_ok=True)
            (cache / 'app.pyc').write_bytes(bytes([index]))
        self.agent._record_command_result(self.callback_result(), 'python app.py')
        self.assertTrue(self.agent._stuck)
        self.agent._stuck = False
        source.write_text('def missing_handler(): pass\nmissing_handler()\n')
        result = self.callback_result()
        self.agent._record_command_result(result, 'python app.py')
        self.assertNotIn('recovery_hint', result)
        self.assertFalse(self.agent._stuck)

    def test_dependency_revision_allows_retry_but_pip_list_does_not(self):
        environment = self.make_environment()
        failure = {'exit_code': 1, 'output': "ModuleNotFoundError: No module named 'example'"}
        for _ in range(2):
            self.agent._record_command_result(dict(failure), 'python app.py')
            self.agent._record_command_result({'exit_code': 0, 'output': 'Package Version'}, 'python -m pip list')
        metadata = environment / 'Lib/site-packages/example-1.0.dist-info/METADATA'
        metadata.parent.mkdir(parents=True)
        metadata.write_text('Name: example\nVersion: 1.0\n')
        result = dict(failure)
        self.agent._record_command_result(result, 'python app.py')
        self.assertNotIn('recovery_hint', result)
        self.assertFalse(self.agent._stuck)

    def test_duplicate_callback_lines_count_once_per_command(self):
        result = self.callback_result(diagnostic='Exception: failed callback')
        result['output'] *= 4
        self.agent._record_command_result(result, 'python app.py')
        self.assertIn('error', result)
        self.assertEqual(list(self.agent._command_failures.values()), [1])
        self.assertFalse(self.agent._stuck)


if __name__ == '__main__':
    unittest.main()
