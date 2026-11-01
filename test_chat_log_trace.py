import importlib.util
import json
import sys
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.debug import model_failure
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

    def test_each_parameter_has_an_independent_native_extraction_card(self):
        for stage, name, value, tool in (("text:item", "item", "nudeln", "glow_assistant_extract_text"),
                                        ("text:description", "description", "weizen", "glow_assistant_extract_text"),
                                        ("numeric:brightness_pct", "brightness_pct", 50, "glow_assistant_extract_number")):
            self.trace(stage, {"parameter": name})
            self.trace(stage, {}, {"parameter": name, "value": value})
            self.assertEqual(self.messages[-2].tool_calls[0].tool_name, tool)
            self.assertEqual(self.messages[-1].tool_result["value"], value)
        self.assertEqual(len({message.tool_calls[0].id for message in self.messages if isinstance(message, AssistantContent)}), 3)
        self.assertFalse(self.trace.calls)

    def test_repeated_stages_use_unique_call_ids(self):
        for _ in range(2):
            self.trace("area", {})
            self.trace("area", {}, {"status": "completed"})
        ids = [message.tool_calls[0].id for message in self.messages if isinstance(message, AssistantContent)]
        self.assertEqual(len(set(ids)), 2)

    def test_text_stage_uses_a_native_tool_card_with_word_details(self):
        result = {"duration_ms": 40, "decisions": {"text_words": {"probabilities": {"word:1:Energy": 0.4, "none": 0.6}, "probability_threshold": 0.02}}}
        self.trace("text", {"words": ["Add", "Energy", "Drinks"]})
        self.trace("text", {}, result)
        self.assertEqual(self.messages[0].tool_calls[0].tool_name, "glow_assistant_extract_text")
        self.assertEqual(self.messages[1].tool_result, result)

    def test_domain_selection_stage_uses_a_native_tool_card_with_counts(self):
        result = {"selected_domain": "sensor", "entities_before": 100, "entities_after": 20}
        self.trace("domain", {"candidate_counts": {"domain": 4}})
        self.trace("domain", {}, result)
        self.assertEqual(self.messages[0].tool_calls[0].tool_name, "glow_assistant_select_domain")
        self.assertEqual(self.messages[1].tool_result, result)

    def test_optional_parameter_card_contains_all_probabilities_and_retained_fields(self):
        result = {"probabilities": {"brightness_pct": 0.3, "hue": 0.2, "none": 0.5},
                  "selected_parameters": ["brightness_pct", "hue"], "probability_threshold": 0.15}
        self.trace("optional_parameters", {"service": "light.turn_on"})
        self.trace("optional_parameters", {}, result)
        self.assertEqual(self.messages[-2].tool_calls[0].tool_name, "glow_assistant_select_optional_parameters")
        self.assertEqual(self.messages[-1].tool_result, result)

    def test_capability_filter_has_native_debug_details_without_a_model_call(self):
        result = {"status": "completed", "model_call": False, "fields_before": 4, "fields_after": 2,
                  "removed_fields": {"target_temp_low": {"unsupported_entity_ids": ["climate.office"]}}}
        self.trace("parameter_capabilities", {"target_entity_ids": ["climate.office"]})
        self.trace("parameter_capabilities", {}, result)
        self.assertEqual(self.messages[-2].tool_calls[0].tool_name, "glow_assistant_filter_parameter_capabilities")
        self.assertEqual(self.messages[-1].tool_result, result)

    def test_target_scope_has_its_own_native_tool_card(self):
        result = {"decisions": {"target_scope": {"choice": "area", "confidence": 0.9}}, "duration_ms": 20}
        self.trace("target_scope", {"candidate_counts": {"target_scope": 3}})
        self.trace("target_scope", {}, result)
        self.assertEqual(self.messages[0].tool_calls[0].tool_name, "glow_assistant_select_target_scope")
        self.assertEqual(self.messages[1].tool_result, result)

    def test_action_area_and_entity_selection_have_native_cards(self):
        for stage, name in (("action_areas", "glow_assistant_select_action_areas"),
                            ("action_entities", "glow_assistant_select_action_entities")):
            self.trace(stage, {"service": "light.turn_off"})
            result = {"selected_names": ["Office"], "probability_threshold": 0.15}
            self.trace(stage, {}, result)
            self.assertEqual(self.messages[-2].tool_calls[0].tool_name, name)
            self.assertEqual(self.messages[-1].tool_result, result)

    def test_yes_no_stages_have_native_debug_cards(self):
        for stage, tool_name in (("query_type", "glow_assistant_select_query_type"),
                                 ("binary_answer", "glow_assistant_answer_yes_no"),
                                 ("state_property", "glow_assistant_select_state_property")):
            with self.subTest(stage=stage):
                result = {"decisions": {stage: {"choice": "unsure", "confidence": 0.3}}, "duration_ms": 20}
                self.trace(stage, {"candidate_counts": {stage: 3}})
                self.trace(stage, {}, result)
                self.assertEqual(self.messages[-2].tool_calls[0].tool_name, tool_name)
                self.assertEqual(self.messages[-1].tool_result, result)
        self.assertFalse(self.trace.calls)

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

    def test_character_count_and_effective_limit_are_visible_in_both_bubble_and_details(self):
        error = ModelError("Model input exceeds its configured character limit.",
                           input_chars=123456, max_input_chars=100000, model="small-answer-model")
        self.trace.failure(model_failure("answer", error))
        bubble, result = self.messages
        self.assertIn("123,456 characters", bubble.content)
        self.assertIn("100,000 characters", bubble.content)
        self.assertIn("Smart-home answer", bubble.content)
        self.assertEqual(result.tool_result["input_chars"], 123456)
        self.assertEqual(result.tool_result["max_input_chars"], 100000)
        self.assertEqual(result.tool_result["model"], "small-answer-model")


if __name__ == "__main__":
    unittest.main()
