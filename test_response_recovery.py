"""Bounded response recovery preserves evidence without replaying completed tools."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import test_agent_core
from test_attachment_core import png_bytes
import runtime
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
        with patch.object(agent, '_mark_images_processed') as mark_images:
            agent.run('Do work')
        mark_images.assert_not_called()
        self.assertEqual(len(requests), 1)
        self.assertTrue(any(item.get('_response_truncated') for item in agent.history))
        self.assertEqual(self.events[-1][0], 'done')
        self.assertFalse(agent._response_recovery)

    def test_recovery_sampling_uses_its_effective_reasoning_mode(self):
        agent, requests = self.agent_with_responses([
            [{'message': {'thinking': 'Incomplete thought'}}, {'done': True, 'done_reason': 'length'}],
            [{'message': {'content': 'Could not complete the task.'}}, {'done': True}],
        ])
        with patch('agent_core.runtime.get_generation_options', side_effect=lambda model, think: {'temperature': .6 if think else .2}) as sampling:
            agent.run('Inspect the issue')
        self.assertEqual([call.args[1] for call in sampling.call_args_list], [True, False])
        self.assertEqual([request['options']['temperature'] for request in requests], [.6, .2])
        self.assertTrue(agent.think)

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

    def test_empty_reasoning_after_success_retries_once_without_replaying_check(self):
        agent, requests = self.agent_with_responses([
            [action('run_command', {'command': 'python check.py'}), {'done': True}],
            [{'message': {'thinking': 'EMPTY_THINKING_SENTINEL', 'content': ' \n '}}, {'done': True, 'done_reason': 'stop'}],
            [{'message': {'content': 'The check passed.'}}, {'done': True}],
        ])
        old = [{'role': 'user', 'content': 'Earlier project work'},
               {'role': 'assistant', 'content': 'Earlier information. ' * 3000}]
        agent.history = copy.deepcopy(old)
        with patch.object(agent, '_run_command', return_value={'exit_code': 0, 'output': 'CHECK_PASSED_SENTINEL'}) as check:
            agent.run('Run the check, then report the result.')
        check.assert_called_once()
        self.assertEqual([request['think'] for request in requests], [True, True, False])
        self.assertTrue(agent.think)
        self.assertEqual(agent.history[-1]['content'], 'The check passed.')
        self.assertEqual(agent._records()[:2], old)
        self.assertIn('CHECK_PASSED_SENTINEL', json.dumps(requests[-1]))
        self.assertIn('EMPTY-RESPONSE RECOVERY', requests[-1]['messages'][0]['content'])
        self.assertNotIn('EMPTY_THINKING_SENTINEL', json.dumps(requests[-1]))
        self.assertFalse(any(item.get('role') == 'assistant' and not item.get('content', '').strip()
                             and not item.get('tool_calls') for item in requests[-1]['messages']))
        saved = next(item for item in agent._records() if item.get('_response_empty'))
        self.assertEqual(saved['content'], ' \n ')
        self.assertIn('EMPTY_THINKING_SENTINEL', agent._read_working_history(
            message_index=agent._records().index(saved))['text'])
        self.assertTrue(all(valid for _, _, valid in agent._groups(agent._records())))
        self.assertTrue(any(kind == 'memory' for kind, _ in self.events))
        self.assertFalse(any(kind == 'error' for kind, _ in self.events))

    def test_empty_and_output_limit_share_one_recovery_per_run(self):
        empty = [{'message': {'thinking': 'No final response', 'content': ''}}, {'done': True}]
        limited = [{'done': True, 'done_reason': 'length'}]
        for responses in ([empty, empty], [empty, limited], [limited, empty]):
            with self.subTest(responses=responses):
                agent, requests = self.agent_with_responses(responses)
                agent.run('Complete the task')
                self.assertEqual(len(requests), 2)
                self.assertIn('Paused because', agent.history[-1]['content'])
                self.assertIn('recovery', agent.history[-1]['content'])
                self.assertFalse(agent._response_recovery)
                self.assertTrue(agent.think)

    def test_empty_retry_preserves_image_until_a_usable_response(self):
        agent, requests = self.agent_with_responses([
            [{'message': {'thinking': 'Inspecting image'}}, {'done': True}],
            [{'message': {'content': 'The reference is red.'}}, {'done': True}],
        ])
        agent.vision_supported = True
        png = png_bytes()
        with patch.object(runtime, 'ROOT', self.root):
            stored = runtime._store_attachment_bytes(png, '.png')
            image = {'id': hashlib.sha256(png).hexdigest(), 'name': 'reference.png', 'kind': 'image',
                     'path': stored, 'image': stored, 'preview': stored, 'size': len(png), 'mime': 'image/png'}
            agent.run('Inspect the attached reference', attachments=[image])
        self.assertEqual([sum(len(item.get('images', [])) for item in request['messages']) for request in requests], [1, 1])
        self.assertTrue(agent._records()[0]['_images_processed'])
        self.assertNotIn(base64.b64encode(png).decode(), json.dumps(agent.history))

    def test_stop_before_empty_retry_prevents_another_request(self):
        def emit(kind, data):
            self.events.append((kind, data))
            if kind == 'status' and data.get('text', '').startswith('The model returned no answer or action.'):
                self.cancel.set()
        agent, requests = self.agent_with_responses([[{'done': True}]], emit=emit)
        with patch.object(agent, '_mark_images_processed') as mark:
            agent.run('Inspect the task')
        mark.assert_not_called()
        self.assertEqual(len(requests), 1)
        self.assertTrue(any(item.get('_response_empty') for item in agent.history))
        self.assertEqual(self.events[-1][0], 'done')

    def test_empty_retry_drains_queued_instruction_before_request(self):
        pending = []
        def emit(kind, data):
            if kind == 'status' and data.get('text', '').startswith('The model returned no answer or action.'):
                pending.append('Report the existing result without making further edits.')
        agent, requests = self.agent_with_responses([
            [{'done': True}], [{'message': {'content': 'Reported the existing result.'}}, {'done': True}],
        ], emit=emit)
        def drain():
            messages = pending[:]
            pending.clear()
            return messages
        agent.poll_messages = drain
        agent.run('Review the existing result')
        self.assertIn('Report the existing result without making further edits.', json.dumps(requests[1]))

    def test_empty_retry_keeps_normal_command_approval(self):
        agent, requests = self.agent_with_responses([
            [{'done': True}],
            [action('run_command', {'command': 'python check.py'}), {'done': True}],
        ])
        self.approve.return_value = False
        with patch('agent_core.subprocess.Popen') as process:
            agent.run('Run the check')
        process.assert_not_called()
        self.approve.assert_called_once()
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(valid for _, _, valid in agent._groups(agent._records())))

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
