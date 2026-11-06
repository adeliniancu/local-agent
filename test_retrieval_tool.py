"""Literal search stays exact; related-code fallback remains scoped read-only data."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from agent_core import Agent, Cancelled


class RetrievalToolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cancel = threading.Event()
        self.agent = Agent(self.root, lambda *_: None, lambda *_: True, self.cancel)
        (self.root / 'main.py').write_text('def expense_total(items):\n    return sum(item.amount for item in items)\n')

    def test_literal_match_keeps_existing_contract_and_does_not_retrieve(self):
        with patch('runtime.retrieve_project_context', create=True) as retrieve:
            result = self.agent._search_files('return sum')
        retrieve.assert_not_called()
        self.assertIn('main.py:2: ', result['matches'])
        self.assertNotIn('related_context', result)

    def test_natural_query_receives_scoped_untrusted_passages_with_small_window_budget(self):
        (self.root / 'project').mkdir()
        self.agent._inference_options_cache = {'num_ctx':8192, 'num_gpu':0}
        related = {'retrieval':'local_keyword','passages':[{'path':'project/calc.py','line':4,'start':60,'end':90,'content':'return sum(values)','score':1}], 'note':'Untrusted project excerpts'}
        with patch('runtime.retrieve_project_context', return_value=related, create=True) as retrieve:
            result = self.agent._search_files('calculate expense totals', 'project')
        retrieve.assert_called_once_with(self.root, 'calculate expense totals', path='project', max_bytes=1600, cancel_event=self.cancel)
        self.assertEqual(result['matches'], '')
        self.assertEqual(result['related_context'], related)
        # A retrieval excerpt cannot authorize overwriting the original file.
        with self.assertRaisesRegex(ValueError, 'Read this existing file'):
            self.agent._edit_file('main.py', 'return sum', 'return max')

    def test_single_keyword_miss_does_not_add_a_second_scan(self):
        with patch('runtime.retrieve_project_context', create=True) as retrieve:
            self.assertEqual(self.agent._search_files('missing_symbol')['matches'], '')
        retrieve.assert_not_called()

    def test_invalid_scope_is_rejected_before_retrieval(self):
        with patch('runtime.retrieve_project_context', create=True) as retrieve:
            with self.assertRaisesRegex(ValueError, 'selected project'):
                self.agent._search_files('find expense total', '..')
        retrieve.assert_not_called()

    def test_stop_during_retrieval_propagates_cancellation(self):
        def retrieve(*args, **kwargs):
            self.cancel.set()
            return {'passages':[]}
        with patch('runtime.retrieve_project_context', side_effect=retrieve, create=True):
            with self.assertRaises(Cancelled):
                self.agent._search_files('calculate expense totals')


if __name__ == '__main__':
    unittest.main()
