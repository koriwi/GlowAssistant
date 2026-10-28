import asyncio
import json
import os
import sys
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tts_answer import answer_config, answer_context, generate_answer


class AnswerTests(unittest.TestCase):
    def test_configuration_is_optional_and_independent(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-key"}, clear=True):
            self.assertIsNone(answer_config())
        with patch.dict(os.environ, {
            "ANSWER_URL": "http://localhost:8322/v1/chat/completions",
            "ANSWER_MODEL": "small-chat-model", "OPENROUTER_API_KEY": "router-key",
            "DECIDER_MODEL": "decider", "DECIDER_API_KEY": "decider-key",
        }, clear=True):
            self.assertEqual(answer_config(), {
                "url": "http://localhost:8322/v1/chat/completions",
                "model": "small-chat-model", "api_key": None,
            })

    def test_openrouter_key_fallback_and_explicit_provider_key(self):
        environment = {
            "ANSWER_URL": "https://openrouter.ai/api/v1/chat/completions",
            "ANSWER_MODEL": "provider/small-model", "OPENROUTER_API_KEY": "router-key",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(answer_config()["api_key"], "router-key")
            os.environ["ANSWER_API_KEY"] = "answer-key"
            self.assertEqual(answer_config()["api_key"], "answer-key")

    def test_invalid_configuration_fails_early(self):
        for environment, message in (
            ({"ANSWER_URL": "localhost:8322"}, "ANSWER_URL"),
            ({"ANSWER_MODEL": "small-model"}, "ANSWER_URL"),
            ({"ANSWER_URL": "http://localhost:8322/v1/chat/completions"}, "ANSWER_MODEL"),
            ({"ANSWER_URL": "https://openrouter.ai/api/v1/chat/completions", "ANSWER_MODEL": "small"}, "ANSWER_API_KEY"),
        ):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(ValueError, message):
                    answer_config()

    def test_query_context_contains_selected_facts_and_omits_stale_values(self):
        result = {
            "prompt": "Wie warm ist es?", "service": None, "domain": "sensor",
            "target": {"type": "entity", "names": "Room temperature", "state": "22"},
            "state_info": [
                {"names": "Room temperature", "domain": "sensor", "state": "22", "attributes": {"unit_of_measurement": "°C"}},
                {"names": "Offline sensor", "domain": "sensor", "state": "unavailable", "attributes": {"temperature": 99}},
            ],
            "decisions": {"scores": "unneeded"}, "usage": {"cost": 1},
        }
        context = answer_context(result)
        self.assertEqual(context["operation"], "query_state")
        self.assertEqual(context["state_info"][0]["attributes"]["unit_of_measurement"], "°C")
        self.assertNotIn("attributes", context["state_info"][1])
        self.assertNotIn("state", context["target"])
        self.assertNotIn("decisions", context)
        self.assertNotIn("usage", context)

    def test_action_context_preserves_outcome_and_applied_arguments(self):
        for execution in (
            {"status": "completed", "arguments": {"temperature": 22}, "response": {"large": "raw MCP payload"}},
            {"status": "skipped", "reason": "Ambiguous target"},
            {"status": "error", "error": "Device outcome unconfirmed"},
        ):
            result = {
                "prompt": "Make it warmer", "service": "assist__HassClimateSetTemperature",
                "domain": "climate", "area": "Office", "target": {"type": "area", "name": "Office"},
                "execution": execution,
            }
            with self.subTest(status=execution["status"]):
                context = answer_context(result)
                self.assertEqual(context["execution"], {key: value for key, value in execution.items() if key != "response"})
                self.assertNotIn("state_info", context)

    def call_endpoint(self, data, key=None, http_error=False):
        response = SimpleNamespace(is_error=http_error, status_code=503, json=lambda: data)
        client = SimpleNamespace(post=AsyncMock(return_value=response))
        @asynccontextmanager
        async def connection(**kwargs):
            yield client
        config = {"url": "http://localhost:8322/v1/chat/completions", "model": "small-model", "api_key": key}
        result = {"prompt": "Is it on?", "service": None, "domain": "light", "target": None, "state_info": []}
        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=connection)}):
            answer = asyncio.run(generate_answer(result, config))
        return answer, client.post.await_args

    def test_chat_endpoint_payload_response_and_optional_auth(self):
        data = {"choices": [{"message": {"content": "  Die Lampe ist an.  "}, "finish_reason": "stop"}], "usage": {"completion_tokens": 8}}
        for key in (None, "answer-key"):
            with self.subTest(key=key):
                (answer, usage), request = self.call_endpoint(data, key)
                self.assertEqual(answer, "Die Lampe ist an.")
                self.assertEqual(usage, data["usage"])
                self.assertEqual(request.args, ("http://localhost:8322/v1/chat/completions",))
                self.assertEqual(request.kwargs["headers"], {"Authorization": "Bearer answer-key"} if key else {})
                payload = request.kwargs["json"]
                self.assertEqual(payload["model"], "small-model")
                self.assertEqual([message["role"] for message in payload["messages"]], ["system", "user"])
                self.assertEqual(json.loads(payload["messages"][1]["content"])["request"], "Is it on?")
                self.assertNotIn("tools", payload)

    def test_empty_malformed_and_truncated_answers_are_errors(self):
        for data in (
            {}, {"choices": []}, {"choices": [{"message": {"content": None}}]},
            {"choices": [{"message": {"content": " "}}]},
            {"choices": [{"message": {"content": "I have", "reasoning_content": "hidden"}, "finish_reason": "length"}]},
        ):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "Answer endpoint"):
                self.call_endpoint(data)
        with self.assertRaisesRegex(ValueError, "Answer endpoint HTTP 503"):
            self.call_endpoint({}, http_error=True)


if __name__ == "__main__":
    unittest.main()
