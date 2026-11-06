import json
import struct
import unittest
from unittest.mock import patch

import adapter_smoke as smoke


class AdapterSmokeTests(unittest.TestCase):
    def test_synthetic_png_is_large_rgb_image(self):
        raw = smoke.probe_png()
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", raw[16:24]), (256, 128))

    def test_native_tool_requires_exact_complete_call_without_execution(self):
        response = {"done": True, "message": {"tool_calls": [{"function": {
            "name": "echo_probe", "arguments": {"value": smoke.NONCE}}}]}}
        self.assertTrue(smoke.check_response("native_tools", response))
        response["done_reason"] = "length"
        self.assertFalse(smoke.check_response("native_tools", response))
        response.pop("done_reason")
        response["message"]["tool_calls"][0]["function"]["arguments"] = {"value": "wrong"}
        self.assertFalse(smoke.check_response("native_tools", response))

    def test_correct_vision_answer_and_malformed_answer(self):
        response = {"done": True, "message": {"content": json.dumps({"left": "red", "right": "blue"})}}
        self.assertTrue(smoke.check_response("vision", response))
        response["message"]["content"] = "The image is red and blue"
        self.assertFalse(smoke.check_response("vision", response))

    def test_probe_calls_are_local_requests_and_errors_do_not_claim_compatibility(self):
        replies = [{"done": True, "message": {"content": "READY"}},
                   {"done": True, "message": {"tool_calls": [{"function": {"name": "echo_probe", "arguments": {"value": smoke.NONCE}}}]}},
                   RuntimeError("Projector unavailable")]
        with patch.object(smoke.runtime, "get_model_info", return_value={"capabilities": ["tools", "vision"]}), \
             patch.object(smoke.runtime, "get_inference_options", return_value={"num_ctx": 16384, "num_gpu": -1}), \
             patch.object(smoke.runtime, "_post", side_effect=replies) as request:
            result = smoke.smoke_model("candidate:local")
        self.assertFalse(result["passed"])
        self.assertEqual([p["passed"] for p in result["probes"]], [True, True, False])
        self.assertEqual(request.call_count, 3)
        self.assertTrue(all(call.args[0] == "/api/chat" for call in request.call_args_list))
        self.assertTrue(all(call.args[1]["think"] is False for call in request.call_args_list))
        self.assertTrue(all(call.args[1]["options"]["seed"] == 42 for call in request.call_args_list))


if __name__ == "__main__":
    unittest.main()
