import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.general import general_answer
from custom_components.glow_assistant.settings import normalize_settings


def tool_call(name="mcp_001_001", arguments=None, identifier="call-1"):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments if arguments is not None else {"query": "Eiffel Tower"}),
    }}


class FakeTools:
    def __init__(self, result=None):
        self.tools = [{"type": "function", "function": {"name": "mcp_001_001", "description": "Search", "parameters": {"type": "object"}}}]
        self.result = result or {"content": [{"type": "text", "text": "330 metres"}], "isError": False}
        self.call = AsyncMock(return_value=self.result)
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def describe(self, name):
        if name != "mcp_001_001":
            raise ModelError("Unknown MCP tool")
        return {"server": "web", "tool": "search"}


class FakeModel:
    def __init__(self, messages):
        self.responses = list(messages)
        self.requests = []

    async def completion(self, config, messages, tools):
        self.requests.append(copy.deepcopy((config, messages, tools)))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result, {"completion_tokens": 10}


class GeneralTests(unittest.IsolatedAsyncioTestCase):
    async def run_loop(self, responses, tools=None, config=None, trace=None):
        model = FakeModel(responses)
        tools = tools or FakeTools()
        config = config or normalize_settings({})["general"]
        messages = [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "How tall is the Eiffel Tower?"}]
        answer, usage = await general_answer(model, config, messages, trace=trace, tools_factory=lambda *_: tools)
        return answer, usage, model, tools

    async def test_model_receives_tool_results_and_preserves_provider_reasoning_metadata(self):
        first = {"role": "assistant", "content": None, "tool_calls": [tool_call()],
                 "reasoning_details": [{"type": "reasoning.encrypted", "data": "provider-state"}]}
        answer, usage, model, tools = await self.run_loop([first, {"role": "assistant", "content": "330 metres."}])
        self.assertEqual(answer, "330 metres.")
        self.assertEqual(len(usage), 2)
        tools.call.assert_awaited_once_with("mcp_001_001", {"query": "Eiffel Tower"})
        second_messages = model.requests[1][1]
        self.assertEqual(second_messages[-2], first)
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call-1")
        self.assertEqual(json.loads(second_messages[-1]["content"]), tools.result)
        self.assertEqual(sum(message["role"] == "system" for message in second_messages), 1)
        self.assertTrue(tools.closed)

    async def test_plain_answer_does_not_call_a_tool(self):
        answer, _, _, tools = await self.run_loop([{"role": "assistant", "content": "Hello."}])
        self.assertEqual(answer, "Hello.")
        tools.call.assert_not_awaited()

    async def test_input_limit_failure_preserves_character_counts_in_general_model_debug(self):
        events = []
        error = ModelError("Model input exceeds its configured character limit.",
                           input_chars=2500, max_input_chars=2000, model="general-model")
        with self.assertRaises(ModelError):
            await self.run_loop([error], trace=lambda *event: events.append(copy.deepcopy(event)))
        output = next(output for stage, _, output in events if stage == "general" and output is not None)
        self.assertEqual(output["input_chars"], 2500)
        self.assertEqual(output["max_input_chars"], 2000)
        self.assertEqual(output["model"], "general-model")

    async def test_unknown_tool_prevents_every_tool_in_the_batch(self):
        tools = FakeTools()
        with self.assertRaisesRegex(ModelError, "Unknown MCP"):
            await self.run_loop([{"tool_calls": [tool_call(), tool_call("invented", identifier="call-2")]}], tools=tools)
        tools.call.assert_not_awaited()
        self.assertTrue(tools.closed)

    async def test_invalid_arguments_are_rejected_without_execution(self):
        for value in ("not JSON", "[]", "null", "{\"query\": NaN}"):
            tools = FakeTools()
            call = tool_call()
            call["function"]["arguments"] = value
            with self.subTest(value=value), self.assertRaises(ModelError):
                await self.run_loop([{"tool_calls": [call]}], tools=tools)
            tools.call.assert_not_awaited()

    async def test_repeated_call_is_not_executed_twice(self):
        tools = FakeTools()
        with self.assertRaisesRegex(ModelError, "repeated"):
            await self.run_loop([
                {"tool_calls": [tool_call()]},
                {"tool_calls": [tool_call(identifier="call-2")]},
            ], tools=tools)
        tools.call.assert_awaited_once()

    async def test_duplicate_identifiers_are_rejected(self):
        tools = FakeTools()
        with self.assertRaisesRegex(ModelError, "malformed"):
            await self.run_loop([{"tool_calls": [tool_call(), tool_call(arguments={"query": "other"})]}], tools=tools)
        tools.call.assert_not_awaited()

    async def test_tool_round_limit_stops_more_execution(self):
        tools = FakeTools()
        config = normalize_settings({"general": {"mcp_max_rounds": 1}})["general"]
        with self.assertRaisesRegex(ModelError, "round limit"):
            await self.run_loop([
                {"tool_calls": [tool_call()]},
                {"tool_calls": [tool_call(arguments={"query": "other"}, identifier="call-2")]},
            ], tools=tools, config=config)
        tools.call.assert_awaited_once()

    async def test_total_call_limit_rejects_an_oversized_batch(self):
        tools = FakeTools()
        config = normalize_settings({"general": {"mcp_max_calls": 1}})["general"]
        with self.assertRaisesRegex(ModelError, "call limit"):
            await self.run_loop([{"tool_calls": [tool_call(), tool_call(identifier="call-2")]}], tools=tools, config=config)
        tools.call.assert_not_awaited()

    async def test_uncertain_tool_failure_does_not_request_another_completion(self):
        tools = FakeTools()
        tools.call.side_effect = ModelError("MCP tool call failed; outcome unconfirmed.")
        model = FakeModel([{"tool_calls": [tool_call()]}])
        with self.assertRaisesRegex(ModelError, "unconfirmed"):
            await general_answer(model, normalize_settings({})["general"], [{"role": "system", "content": "Brief"}],
                                 tools_factory=lambda *_: tools)
        self.assertEqual(len(model.requests), 1)
        tools.call.assert_awaited_once()

    async def test_error_results_reach_the_model_and_are_visible_in_debug(self):
        tools = FakeTools({"isError": True, "content": [{"type": "text", "text": "No search results"}]})
        events = []
        answer, _, model, _ = await self.run_loop([
            {"tool_calls": [tool_call()]}, {"content": "I could not find that information."},
        ], tools=tools, trace=lambda *event: events.append(copy.deepcopy(event)))
        self.assertIn("could not", answer)
        result = next(output for stage, _, output in events if stage == "mcp" and output is not None)
        self.assertEqual(result["status"], "error")
        self.assertIn("duration_ms", result)
        self.assertTrue(json.loads(model.requests[1][1][-1]["content"])["isError"])

    async def test_tool_events_include_names_timing_and_match_start_end_order(self):
        events = []
        await self.run_loop([{ "tool_calls": [tool_call()]}, {"content": "330 metres."}],
                            trace=lambda *event: events.append(copy.deepcopy(event)))
        self.assertEqual([stage for stage, _, output in events if output is None], ["mcp_connect", "general", "mcp", "general"])
        self.assertEqual([stage for stage, _, output in events if output is not None], ["mcp_connect", "general", "mcp", "general"])
        request = next(arguments for stage, arguments, output in events if stage == "mcp" and output is None)
        self.assertEqual(request["server"], "web")
        self.assertEqual(request["tool"], "search")

    async def test_discovery_failure_is_visible_in_the_native_debug_events(self):
        tools = FakeTools()
        events = []
        with patch.object(FakeTools, "__aenter__", new=AsyncMock(side_effect=ModelError("MCP server connection failed."))):
            with self.assertRaisesRegex(ModelError, "connection failed"):
                await self.run_loop([], tools=tools, trace=lambda *event: events.append(event))
            self.assertEqual(events[-1][0], "mcp_connect")
            self.assertEqual(events[-1][2]["status"], "error")
            self.assertIn("duration_ms", events[-1][2])


if __name__ == "__main__":
    unittest.main()
