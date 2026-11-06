"""Output-limit recovery must preserve evidence without executing partial tools."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import test_agent_core
from agent_core import Agent


def action(name, arguments):
    return {'message': {'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]}}


class ResponseRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = []
        self.cancel = threading.Event()
        self.approve = Mock(return_value=True)
        resolver = patch('agent_core.runtime.get_inference_options', return_value={'num_ctx': 16384, 'num_gpu': 0})
        resolver.start()
        self.addCleanup(resolver.stop)

    def agent_with_responses(self, responses, *, model='qwen3.5:4b-q4_K_M', emit=None):
        requests = []
        def respond(handler):
            index = len(requests)
            requests.append(handler.request_payload)
            handler.send_response(200)
            handler.end_headers()
            for chunk in responses[index]:
                handler.wfile.write(json.dumps(chunk).encode() + b'\n')
        endpoint = test_agent_core.StreamTests.run_server(self, respond)
        return Agent(self.root, emit or (lambda kind, data: self.events.append((kind, data))),
                     self.approve, self.cancel, model=model, base_url=endpoint, think=True), requests

    def test_partial_tool_is_saved_without_execution_and_recovery_is_one_turn(self):
        incomplete = action('write_file', '{"path":"bad.py","content":"unfinished')
        agent, requests = self.agent_with_responses([
            [{'message': {'thinking': 'RAW_REASONING_SENTINEL', 'content': 'Incomplete code draft'}},
             incomplete, {'done': True, 'done_reason': 'length'}],
            [action('write_file', {'path': 'good.py', 'content': "print('ready')\n"}), {'done': True, 'done_reason': 'stop'}],
            [{'message': {'content': 'Created the small file.'}}, {'done': True}],
        ])
        agent.run('Create a small Python file')
        self.assertFalse((self.root / 'bad.py').exists())
        self.assertEqual((self.root / 'good.py').read_text(), "print('ready')\n")
        self.assertEqual([request['think'] for request in requests], [True, False, True])
        self.assertTrue(agent.think)
        saved = next(item for item in agent.history if item.get('_response_truncated'))
        self.assertEqual(saved['thinking'], 'RAW_REASONING_SENTINEL')
        self.assertEqual(saved['_incomplete_tool_calls'][0]['function']['arguments'], incomplete['message']['tool_calls'][0]['function']['arguments'])
        self.assertIn('Incomplete code draft', saved['content'])
        self.assertNotIn('RAW_REASONING_SENTINEL', json.dumps(requests[1]))
        self.assertNotIn('unfinished', json.dumps(requests[1]))
        self.assertIn('RESPONSE-LIMIT RECOVERY', requests[1]['messages'][0]['content'])
        self.assertNotIn('RESPONSE-LIMIT RECOVERY', requests[2]['messages'][0]['content'])
        self.assertTrue(all(valid for _, _, valid in agent._groups(agent._records())))
        recalled = agent._read_working_history(message_index=agent._records().index(saved), max_chars=6000)
        self.assertIn('RAW_REASONING_SENTINEL', recalled['text'])

    def test_even_valid_calls_are_blocked_if_response_was_cut_off_and_retry_is_bounded(self):
        chunks = [action('write_file', {'path': 'bad.py', 'content': 'print(1)'}),
                  {'done': True, 'done_reason': 'length'}]
        agent, requests = self.agent_with_responses([chunks, chunks])
        agent.run('Make the app')
        self.assertEqual(len(requests), 2)
        self.assertFalse((self.root / 'bad.py').exists())
        self.assertIn('response limit again', agent.history[-1]['content'])
        self.assertEqual(sum(bool(item.get('_response_truncated')) for item in agent.history), 2)
        self.assertFalse(any(kind == 'error' for kind, _ in self.events))
        self.assertEqual(self.events[-1][0], 'done')

    def test_truncated_bare_json_cannot_turn_into_a_fallback_action(self):
        text = json.dumps({'name': 'write_file', 'arguments': {'path': 'bad.py', 'content': 'print(1)'}})
        agent, requests = self.agent_with_responses([
            [{'message': {'content': text}}, {'done': True, 'done_reason': 'length'}],
            [{'message': {'content': 'Unable to complete this task.'}}, {'done': True}],
        ], model='qwen2.5-coder:7b')
        agent.run('Create a file')
        self.assertFalse((self.root / 'bad.py').exists())
        self.assertEqual(len(requests), 2)
        self.assertFalse(any(item.get('role') == 'tool' for item in agent.history))

    def test_stop_before_recovery_keeps_partial_history_and_does_not_request_again(self):
        def emit(kind, data):
            self.events.append((kind, data))
            if kind == 'status' and data.get('text', '').startswith('Response limit reached.'):
                self.cancel.set()
        agent, requests = self.agent_with_responses([
            [{'message': {'thinking': 'unfinished reasoning'}}, {'done': True, 'done_reason': 'length'}],
        ], emit=emit)
        agent.run('Do work')
        self.assertEqual(len(requests), 1)
        self.assertTrue(any(item.get('_response_truncated') for item in agent.history))
        self.assertEqual(self.events[-1][0], 'done')
        self.assertFalse(agent._response_recovery)

    def test_recovery_still_requires_command_approval(self):
        agent, requests = self.agent_with_responses([
            [{'done': True, 'done_reason': 'length'}],
            [action('run_command', {'command': 'python check.py'}), {'done': True}],
        ])
        self.approve.return_value = False
        with patch('agent_core.subprocess.Popen') as process:
            agent.run('Run the checks')
        process.assert_not_called()
        self.approve.assert_called_once()
        self.assertEqual(len(requests), 2)
        self.assertFalse(any(kind == 'error' for kind, _ in self.events))

    def test_qwen_history_omits_old_reasoning_but_preserves_raw_saved_messages(self):
        agent = Agent(self.root, lambda *_: None, self.approve, self.cancel, model='qwen3.5:4b-q4_K_M')
        records = [{'role': 'user', 'content': 'Fix the program'},
                   {'role': 'assistant', 'content': 'Inspecting it.', 'thinking': 'OLD_THINKING_SENTINEL ' * 6000,
                    'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'path': 'main.py'}}}]},
                   {'role': 'tool', 'tool_name': 'read_file', 'content': '{"content":"print(1)"}'}]
        original = copy.deepcopy(records)
        agent.history = records
        messages = agent._messages()
        self.assertNotIn('thinking', messages[2])
        self.assertNotIn('OLD_THINKING_SENTINEL', json.dumps(messages))
        self.assertIn('Inspecting it.', json.dumps(messages))
        self.assertEqual(agent.history, original)
        self.assertTrue(all(valid for _, _, valid in agent._groups(agent._records())))


if __name__ == '__main__':
    unittest.main()
