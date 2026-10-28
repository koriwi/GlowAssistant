import importlib.util
import json
import sys
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4


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


@dataclass
class ToolResultContent:
    agent_id: str
    tool_call_id: str
    tool_name: str
    tool_result: dict


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

    def test_error_results_close_the_pending_call(self):
        self.trace("answer", {"max_tokens": 256})
        self.trace("answer", {}, {"status": "error", "error": "Token limit reached", "duration_ms": 120})
        self.assertEqual(self.messages[-1].tool_result["status"], "error")
        self.assertFalse(self.trace.calls)


if __name__ == "__main__":
    unittest.main()
