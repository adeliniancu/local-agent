"""Offline checks for synthetic evaluation tasks and harness restrictions."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import agent_core
import coding_benchmark as benchmark


class CodingBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def test_all_broken_fixtures_fail_and_reference_repairs_pass(self):
        for case in benchmark.CASES:
            with self.subTest(case=case):
                broken=benchmark.validate(*benchmark.prepare_case(case,self.root/(case+'-broken')))
                reference=benchmark.validate(*benchmark.prepare_case(case,self.root/(case+'-reference'),reference=True))
                self.assertNotEqual(broken['exit_code'],0,broken)
                self.assertEqual(reference['exit_code'],0,reference)

    def test_fixed_tests_outside_workspace_and_only_main_is_editable(self):
        workspace,tests=benchmark.prepare_case('ledger',self.root/'case')
        self.assertFalse(tests.is_relative_to(workspace))
        self.assertEqual((workspace/'fixed_tests.py').read_bytes(),tests.read_bytes())
        agent=benchmark.evaluation_agent(agent_core,workspace,tests,lambda *_:None,threading.Event(),'fake',False)
        self.assertIn('test_exact_cents',agent._execute('read_file',{'path':'fixed_tests.py'})['content'])
        with self.assertRaises(PermissionError): agent._execute('write_file',{'path':'fixed_tests.py','content':'pass'})
        with self.assertRaises(ValueError): agent._execute('read_file',{'path':'../fixed_tests.py'})
        with self.assertRaises(PermissionError): agent._execute('launch_python_app',{'path':'main.py'})

    def test_tampered_reference_cannot_change_external_validation(self):
        workspace,tests=benchmark.prepare_case('ledger',self.root/'case')
        original=tests.read_bytes()
        reference=workspace/'fixed_tests.py'
        reference.chmod(0o666)
        reference.write_text('print("all checks passed")\n',encoding='utf-8')
        result=benchmark.validate(workspace,tests)
        self.assertNotEqual(result['exit_code'],0)
        self.assertEqual(tests.read_bytes(),original)
        self.assertIn('test_exact_cents',result['output'])

    def test_only_fixed_command_runs_without_shell(self):
        workspace,tests=benchmark.prepare_case('ledger',self.root/'case')
        agent=benchmark.evaluation_agent(agent_core,workspace,tests,lambda *_:None,threading.Event(),'fake',False)
        with patch.object(benchmark,'validate',return_value={'exit_code':0,'output':'passed'}) as validate:
            self.assertEqual(agent._run_command(benchmark.VALIDATION_COMMAND)['exit_code'],0)
            validate.assert_called_once()
        with patch.object(benchmark,'validate') as validate:
            with self.assertRaises(PermissionError): agent._run_command('python -c "print(1)"')
            validate.assert_not_called()

    def test_equivalent_validation_commands_are_intercepted_without_shell_execution(self):
        workspace,tests=benchmark.prepare_case('ledger',self.root/'with spaces')
        app_root=self.root/'app with spaces'
        interpreter=app_root/'runtime/python/python.exe'
        import sys
        commands=[
            'python -m unittest', 'python.exe -m unittest -v',
            'python -m unittest fixed_tests.py -q', 'python fixed_tests.py',
            f'cd "{workspace}"; python -m unittest -q',
            f'cd {workspace}; python -m unittest -q',
            f"Set-Location -LiteralPath '{workspace}'; python.exe -m unittest -v",
            f'"{sys.executable}" -m unittest -q',
            f'& "{interpreter}" -m unittest -q',
            f'{interpreter} -m unittest -q',
        ]
        agent=benchmark.evaluation_agent(agent_core,workspace,tests,lambda *_:None,threading.Event(),'fake',False)
        with patch.object(agent_core,'APP_ROOT',app_root), patch.object(benchmark,'validate',return_value={'exit_code':0,'output':'passed'}) as validate:
            with patch.object(benchmark.subprocess,'Popen') as shell:
                for command in commands:
                    with self.subTest(command=command):
                        self.assertTrue(benchmark.is_validation_command(command,workspace,app_root=app_root))
                        self.assertEqual(agent._run_command(command)['exit_code'],0)
                shell.assert_not_called()
            self.assertEqual(validate.call_count,len(commands))

    def test_normalization_rejects_other_paths_expansions_and_commands(self):
        workspace,tests=benchmark.prepare_case('ledger',self.root/'case')
        commands=[
            'python -c "print(1)"', 'python -m unittest --help',
            'python -m unittest discover -s ..', 'python other_tests.py',
            'python -m unittest -q; echo passed',
            f'cd "{workspace}"; python -m unittest -q; echo passed',
            f'cd "{self.root}"; python -m unittest -q',
            f'cd "{workspace}" && python -m unittest -q',
            'python -m unittest -q | Out-String', 'python -m unittest -q > result.txt',
            '$python -m unittest -q', 'python -m unittest -q\necho passed',
            '`python -m unittest -q', 'python -m unittest -q -v',
            'python -m unittest fixed_tests.py fixed_tests.py',
            str(self.root/'unapproved/python.exe')+' -m unittest -q',
        ]
        with patch.object(benchmark,'validate') as validate:
            for command in commands:
                with self.subTest(command=command):
                    self.assertFalse(benchmark.is_validation_command(command,workspace))
            validate.assert_not_called()

    def test_evaluation_code_cannot_write_outside_fixture(self):
        workspace,tests=benchmark.prepare_case('ledger',self.root/'case')
        outside=self.root/'outside.txt'
        (workspace/'main.py').write_text('from pathlib import Path\nPath('+repr(str(outside))+').write_text("bad")\n',encoding='utf-8')
        result=benchmark.validate(workspace,tests)
        self.assertNotEqual(result['exit_code'],0)
        self.assertFalse(outside.exists())
        self.assertIn('PermissionError',result['output'])

    def test_live_pass_uses_external_check_not_assistant_claim(self):
        with patch.object(agent_core.Agent,'run',lambda self,prompt:self.history.append({'role':'assistant','content':'All tests passed!'})):
            result=benchmark.evaluate('ledger',agent_core,self.root/'case',model='fake',think=False,max_steps=1,max_seconds=3)
        self.assertFalse(result['passed'])
        self.assertTrue(result['baseline_failed'])
        self.assertEqual(result['harness_protocol'],benchmark.HARNESS_PROTOCOL)
        self.assertEqual(result['harness_sha256'],benchmark.HARNESS_SOURCE_SHA256)
        self.assertEqual(len(result['harness_sha256']),64)

    def test_limits_restore_even_on_exception(self):
        previous=(agent_core.MAX_RUN_STEPS,agent_core.MAX_RUN_SECONDS)
        with self.assertRaises(RuntimeError):
            with benchmark.run_limits(agent_core,2,3): raise RuntimeError('test')
        self.assertEqual((agent_core.MAX_RUN_STEPS,agent_core.MAX_RUN_SECONDS),previous)

    def test_evaluation_seed_preserves_model_runtime_options(self):
        workspace,tests=benchmark.prepare_case('ledger',self.root/'case')
        agent=benchmark.evaluation_agent(agent_core,workspace,tests,lambda *_:None,threading.Event(),'fake',False)
        with patch.object(agent_core.Agent,'_inference_options',return_value={'num_ctx':16384,'num_gpu':-1}):
            self.assertEqual(agent._inference_options(),{'num_ctx':16384,'num_gpu':-1,'seed':42})

    def test_core_fingerprint_matches_loaded_bytes_after_source_changes(self):
        source=self.root/'frozen_core.py'
        source.write_text('VALUE=1\n',encoding='utf-8')
        original=source.read_bytes()
        loaded=benchmark.load_core(source)
        source.write_text('VALUE=2\n',encoding='utf-8')
        import hashlib
        self.assertEqual(loaded.VALUE,1)
        self.assertEqual(loaded._benchmark_source_sha256,hashlib.sha256(original).hexdigest())


if __name__=='__main__': unittest.main()
