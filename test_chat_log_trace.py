import importlib.util
import json
import sys
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from custom_components.glow_assistant.debug import failure_details, model_failure
from custom_components.glow_assistant.history import conversation_history


@dataclass
class ToolInput:
    tool_name: str
    tool_args: dict
    external: bool = False
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class AssistantContent:
    agent_id: str
    tool_calls: list
    content: str | None = None
    role: str = "assistant"


@dataclass
class ToolResultContent:
    agent_id: str
    tool_call_id: str
    tool_name: str
    tool_result: dict
    role: str = "tool_result"


class ChatLogTraceTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent / "custom_components/glow_assistant/chat_log_trace.py"
        spec = importlib.util.spec_from_file_location("custom_components.glow_assistant._chat_log_trace_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "homeassistant.components.conversation.chat_log": SimpleNamespace(
                AssistantContent=AssistantContent, ToolResultContent=ToolResultContent,
            ),
            "homeassistant.helpers.llm": SimpleNamespace(ToolInput=ToolInput),
        }):
            spec.loader.exec_module(module)
        self.messages = []
        self.chat_log = SimpleNamespace(
            async_add_assistant_content_without_tools=Mock(side_effect=self.messages.append),
        )
        self.trace = module.ChatLogTrace(self.chat_log, "conversation.glow")

    def test_matching_external_call_and_result_are_added_to_native_chat_log(self):
        arguments = {"model": "decider", "candidate_counts": {"target": 4}}
        result = {"duration_ms": 35, "decisions": {"target": {"choice": "area:office", "confidence": 0.9}}}
        self.trace("target", arguments)
        self.trace("target", {}, result)
        call_message, result_message = self.messages
        call = call_message.tool_calls[0]
        self.assertTrue(call.external)
        self.assertIsNone(call_message.content)
        self.assertEqual(call.tool_args, arguments)
        self.assertEqual(call.tool_name, "glow_assistant_select_target")
        self.assertEqual(result_message.tool_call_id, call.id)
        self.assertEqual(result_message.tool_name, call.tool_name)
        self.assertEqual(result_message.tool_result, result)
        self.assertEqual(result_message.agent_id, "conversation.glow")
        self.assertEqual(json.loads(json.dumps(asdict(result_message)))["tool_result"], result)
        self.assertFalse(self.trace.calls)

    def test_repeated_stages_use_unique_call_ids(self):
        for _ in range(2):
            self.trace("area", {})
            self.trace("area", {}, {"status": "completed"})
        ids = [message.tool_calls[0].id for message in self.messages if isinstance(message, AssistantContent)]
        self.assertEqual(len(set(ids)), 2)

    def test_text_stage_uses_a_native_tool_card_with_word_details(self):
        result = {"duration_ms": 40, "decisions": {"word_1": {"choice": "keep", "confidence": 0.9, "kept": True}}}
        self.trace("text", {"words": ["Add", "Energy", "Drinks"]})
        self.trace("text", {}, result)
        self.assertEqual(self.messages[0].tool_calls[0].tool_name, "glow_assistant_extract_text")
        self.assertEqual(self.messages[1].tool_result, result)

    def test_error_results_close_the_pending_call(self):
        self.trace("answer", {"max_tokens": 256})
        self.trace("answer", {}, {"status": "error", "error": "Token limit reached", "duration_ms": 120})
        self.assertEqual(self.messages[-1].tool_result["status"], "error")
        self.assertFalse(self.trace.calls)

    def test_failure_is_visible_text_with_expandable_details_and_stays_out_of_history(self):
        details = model_failure("answer", "Model reply reached its output token limit.")
        self.trace.failure(details)
        bubble, result = self.messages
        self.assertIn("output token limit", bubble.content)
        self.assertIn("Increase Maximum output tokens under Smart-home answer", bubble.content)
        self.assertTrue(bubble.tool_calls[0].external)
        self.assertEqual(result.tool_result["stage"], "answer")
        self.assertEqual(result.tool_call_id, bubble.tool_calls[0].id)
        self.assertEqual(conversation_history(self.messages, 10), [])
        self.assertTrue(self.trace.failure_reported)
        self.assertFalse(self.trace.calls)

    def test_unexpected_failure_closes_pending_calls_and_retains_command_uncertainty(self):
        self.trace("execution", {"service": "light.turn_off"})
        self.trace.failure(failure_details("execution", "An unexpected error occurred.", "Check the logs."))
        self.assertEqual(self.messages[1].tool_result["status"], "error")
        self.assertIn("outcome is unconfirmed", self.messages[2].content)
        self.assertFalse(self.trace.calls)

    def test_answer_failure_after_execution_shows_that_the_command_was_already_sent(self):
        self.trace("execution", {"service": "light.turn_off"})
        self.trace("execution", {}, {"status": "completed"})
        self.trace.failure(model_failure("answer", "Model returned no spoken answer."))
        self.assertIn("Home Assistant accepted the device command", self.messages[2].content)
        self.assertIn("Check the device before repeating", self.messages[2].content)
        self.assertEqual(self.messages[3].tool_result["device_command"], "completed")


if __name__ == "__main__":
    unittest.main()
