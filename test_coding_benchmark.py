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
        agent=benchmark.evaluation_agent(agent_core,workspace,tests,lambda *_:None,threading.Event(),'fake',False)
        with self.assertRaises(PermissionError): agent._execute('write_file',{'path':'fixed_tests.py','content':'pass'})
        with self.assertRaises(ValueError): agent._execute('read_file',{'path':'../fixed_tests.py'})
        with self.assertRaises(PermissionError): agent._execute('launch_python_app',{'path':'main.py'})

    def test_only_fixed_command_runs_without_shell(self):
        workspace,tests=benchmark.prepare_case('ledger',self.root/'case')
        agent=benchmark.evaluation_agent(agent_core,workspace,tests,lambda *_:None,threading.Event(),'fake',False)
        with patch.object(benchmark,'validate',return_value={'exit_code':0,'output':'passed'}) as validate:
            self.assertEqual(agent._run_command(benchmark.VALIDATION_COMMAND)['exit_code'],0)
            validate.assert_called_once()
        with patch.object(benchmark,'validate') as validate:
            with self.assertRaises(PermissionError): agent._run_command('python -c "print(1)"')
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

    def test_limits_restore_even_on_exception(self):
        previous=(agent_core.MAX_RUN_STEPS,agent_core.MAX_RUN_SECONDS)
        with self.assertRaises(RuntimeError):
            with benchmark.run_limits(agent_core,2,3): raise RuntimeError('test')
        self.assertEqual((agent_core.MAX_RUN_STEPS,agent_core.MAX_RUN_SECONDS),previous)


if __name__=='__main__': unittest.main()
