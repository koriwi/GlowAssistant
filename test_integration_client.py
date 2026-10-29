import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from custom_components.glow_assistant.client import ModelClient, ModelError
from custom_components.glow_assistant.settings import normalize_settings


class IntegrationClientTests(unittest.IsolatedAsyncioTestCase):
    def client(self, data, status=200):
        response = SimpleNamespace(status=status, json=AsyncMock(return_value=data))
        @asynccontextmanager
        async def request(*args, **kwargs):
            yield response
        session = SimpleNamespace(post=Mock(side_effect=request))
        return ModelClient(session), session, response

    async def test_native_decider_uses_exact_endpoint_model_payload_and_key(self):
        client, session, _ = self.client({"answers": {}})
        config = normalize_settings({"decider": {
            "url": "http://server:8321/v1/systemone", "model": "native-decider", "api_key": "decider-key",
        }})["decider"]
        questions = {"service": {"type": "choice", "criteria": {"unsure": "Unknown"}}}
        response = await client.decide(config, {"prompt": "test"}, questions)
        self.assertEqual(response, {"answers": {}})
        call = session.post.call_args
        self.assertEqual(call.args, (config["url"],))
        self.assertEqual(call.kwargs["json"], {"model": "native-decider", "state": {"prompt": "test"}, "questions": questions})
        self.assertEqual(call.kwargs["headers"], {"Authorization": "Bearer decider-key"})
        self.assertFalse(call.kwargs["allow_redirects"])

    async def test_chat_applies_its_own_limits_and_sends_no_tools(self):
        client, session, _ = self.client({"choices": [{"message": {"content": "  Hallo!  "}}], "usage": {"completion_tokens": 2}})
        config = normalize_settings({"general": {
            "url": "http://server:8323/v1/chat/completions", "model": "smart-model", "max_tokens": 1024,
            "temperature": 0.3, "timeout": 90,
        }})["general"]
        messages = [{"role": "user", "content": "Hi!"}]
        self.assertEqual(await client.chat(config, messages), ("Hallo!", {"completion_tokens": 2}))
        call = session.post.call_args
        self.assertEqual(call.kwargs["headers"], {})
        self.assertEqual(call.kwargs["timeout"].total, 90)
        self.assertEqual(call.kwargs["json"], {
            "model": "smart-model", "messages": messages, "max_tokens": 1024, "temperature": 0.3,
        })

    async def test_input_limit_rejects_without_network_or_silent_truncation(self):
        client, session, _ = self.client({})
        config = normalize_settings({"answer": {"max_input_chars": 100}})["answer"]
        with self.assertRaisesRegex(ModelError, "character limit"):
            await client.chat(config, [{"role": "user", "content": "x" * 101}])
        session.post.assert_not_called()

    async def test_http_errors_and_redirects_are_not_retried(self):
        for status in (302, 401, 429, 500):
            client, session, _ = self.client({}, status)
            with self.subTest(status=status), self.assertRaisesRegex(ModelError, f"HTTP {status}"):
                await client.chat(normalize_settings({})["answer"], [])
            session.post.assert_called_once()

    async def test_invalid_empty_and_truncated_chat_responses(self):
        for data in (
            {}, [], {"error": {"message": "provider error"}},
            {"choices": [{"message": {"content": None}}]},
            {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "unfinished"}, "finish_reason": "length"}]},
        ):
            client, session, _ = self.client(data)
            with self.subTest(data=data), self.assertRaises(ModelError):
                await client.chat(normalize_settings({})["answer"], [])
            session.post.assert_called_once()

    async def test_completion_supports_tool_calls_without_spoken_content(self):
        call = {"id": "call-1", "type": "function", "function": {"name": "mcp_001_001", "arguments": "{}"}}
        client, session, _ = self.client({"choices": [{"message": {"content": None, "tool_calls": [call]}}]})
        config = normalize_settings({})["general"]
        tools = [{"type": "function", "function": {"name": "mcp_001_001", "parameters": {"type": "object"}}}]
        message, _ = await client.completion(config, [{"role": "user", "content": "Find it"}], tools)
        self.assertEqual(message["tool_calls"], [call])
        self.assertEqual(message["role"], "assistant")
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertFalse(payload["parallel_tool_calls"])

    async def test_plain_chat_rejects_unrequested_tool_calls(self):
        client, _, _ = self.client({"choices": [{"message": {"tool_calls": [{"id": "unexpected"}]}}]})
        with self.assertRaisesRegex(ModelError, "Unexpected tool"):
            await client.chat(normalize_settings({})["answer"], [])

    async def test_empty_tool_call_list_can_accompany_a_normal_answer(self):
        client, _, _ = self.client({"choices": [{"message": {"content": "Hello", "tool_calls": []}}]})
        text, _ = await client.chat(normalize_settings({})["answer"], [])
        self.assertEqual(text, "Hello")


if __name__ == "__main__":
    unittest.main()
