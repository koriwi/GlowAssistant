import copy
import unittest
from unittest.mock import AsyncMock

from custom_components.glow_assistant.actions import (
    ActionError,
    action_calls,
    device_outcome,
)
from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.debug import failure_text
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.settings import normalize_settings
from test_integration_engine import SNAPSHOT, ScriptedModels, entity
from test_parameter_values import numeric_answers


class ActionPlanTests(unittest.TestCase):
    def result(self):
        return {
            "service": "light.turn_on", "domain": "light", "parameters": {"brightness_pct": 50, "transition": 2},
            "action_metadata": {"fields": {"brightness_pct": {}, "transition": {}}},
            "parameter_capabilities": {"field_support": {
                "brightness_pct": {"supported_entity_ids": ["light.a"]},
                "transition": {"supported_entity_ids": ["light.a", "light.b"]},
            }},
        }

    def test_groups_targets_by_supported_fields_and_expands_only_explicit_entity_ids(self):
        calls = action_calls(self.result(), [entity("light.a", "A"), entity("light.b", "B"), entity("light.c", "C")])
        self.assertEqual([call["arguments"] for call in calls], [
            {"brightness_pct": 50, "transition": 2, "entity_id": ["light.a"]},
            {"transition": 2, "entity_id": ["light.b"]}, {"entity_id": ["light.c"]},
        ])
        self.assertTrue(all(call["service"] == "light.turn_on" for call in calls))
        self.assertTrue(all(call["status"] == "planned" for call in calls))

    def test_missing_required_and_broad_target_parameters_are_rejected(self):
        for target in ("entity_id", "area_id", "device_id", "floor_id", "label_id"):
            result = self.result()
            result["parameters"][target] = "all"
            with self.subTest(target=target), self.assertRaises(ActionError):
                action_calls(result, [entity("light.a", "A")])
        result = self.result()
        result["action_metadata"]["fields"]["required_value"] = {"required": True}
        with self.assertRaisesRegex(ActionError, "Missing required"):
            action_calls(result, [entity("light.a", "A")])
        result = self.result()
        result["action_metadata"]["fields"]["brightness_pct"]["required"] = True
        with self.assertRaisesRegex(ActionError, "does not support"):
            action_calls(result, [entity("light.b", "B")])
        with self.assertRaisesRegex(ActionError, "domain"):
            action_calls(self.result(), [entity("switch.a", "Wrong")])


class ActionExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_commands_execute_once_and_dry_run_never_invokes_the_executor(self):
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                execute = AsyncMock(return_value={"accepted": True})
                client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}])
                events = []
                result = await AgentEngine(normalize_settings({}), client, dry_run=dry_run, execute=execute,
                                           trace=lambda *event: events.append(event)).process(
                    "Turn off the lights", "en", SNAPSHOT, speaker_area="Office")
                self.assertEqual(result["answer"], "ok")
                self.assertFalse(client.chat_requests)
                self.assertEqual(result["execution"]["status"], "planned" if dry_run else "completed")
                if dry_run:
                    execute.assert_not_awaited()
                else:
                    execute.assert_awaited_once_with("light.turn_off", {"entity_id": ["light.desk", "light.ceiling"]})
                output = next(output for stage, _, output in events if stage == "execution" and output)
                self.assertEqual(output["device_command"], "not_sent" if dry_run else "completed")

    async def test_live_mode_without_an_executor_fails_instead_of_pretending_success(self):
        client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}])
        result = await AgentEngine(normalize_settings({}), client).process("Turn off the lights", "en", SNAPSHOT, speaker_area="Office")
        self.assertEqual(result["route"], "error")
        self.assertIn("executor", result["error"])
        self.assertEqual(result["execution"]["calls"][0]["status"], "not_sent")
        self.assertEqual(result["failures"][0]["device_command"], "not_sent")

    async def test_partial_failure_records_completed_unconfirmed_and_unsent_groups_without_retry(self):
        execute = AsyncMock(side_effect=[None, ActionError("Device failed", command_sent=True)])
        client = ScriptedModels([
            {"service": "light.turn_on:brightness_pct"}, {"target_scope": "implied_area"},
            {"optional_parameters": {"type": "choice", "choice": "brightness_pct", "confidence": 0.9,
                                     "probabilities": {"brightness_pct": 0.5, "transition": 0.5, "none": 0}}},
            numeric_answers(4), numeric_answers(1.6),
        ])
        snapshot = {"entities": [entity("light.a", "A", supported_features=3), entity("light.b", "B", supported_features=1),
                                 entity("light.c", "C", supported_features=0)], "services": {"light.turn_on": {"fields": {
            "brightness_pct": {"filter": {"supported_features": [1]}, "selector": {"number": {"min": 0, "max": 100}}},
            "transition": {"filter": {"supported_features": [2]}, "selector": {"number": {"min": 0, "max": 10}}},
        }}}}
        result = await AgentEngine(normalize_settings({}), client, execute=execute).process("Turn on with brightness and transition", "en", snapshot, speaker_area="Office")
        self.assertEqual(result["route"], "error")
        self.assertEqual(result["execution"]["status"], "partial")
        self.assertEqual([call["status"] for call in result["execution"]["calls"]], ["completed", "unconfirmed", "not_sent"])
        self.assertEqual(execute.await_count, 2)
        self.assertEqual(device_outcome(result["execution"]), "unconfirmed")
        self.assertIn("Check the devices before repeating", failure_text(result["failures"][0]))
        self.assertFalse(client.chat_requests)

    async def test_answer_failure_after_execution_retains_outcome_and_never_retries(self):
        settings = normalize_settings({"answer": {"enable": True}})
        execute = AsyncMock(return_value=None)
        client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}],
                                answers=[ModelError("Model request timed out.")])
        result = await AgentEngine(settings, client, execute=execute).process("Turn off the lights", "en", SNAPSHOT, speaker_area="Office")
        self.assertEqual(result["execution"]["status"], "completed")
        self.assertEqual(result["answer"], settings["answer_error_reply"])
        self.assertIn("answer_error", result)
        self.assertEqual(result["failures"][0]["device_command"], "completed")
        self.assertIn("only the reply failed", failure_text(result["failures"][0]))
        execute.assert_awaited_once()
        facts = client.chat_requests[0][1][1]["content"]
        self.assertIn('"status": "completed"', facts)

    async def test_queries_and_failed_selections_never_execute(self):
        for choices in ([{"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
                         {"query_type": "yes_no"}, {"binary_answer": "yes"}], [{"service": "unsure"}]):
            execute = AsyncMock()
            result = await AgentEngine(normalize_settings({}), ScriptedModels(choices), execute=execute).process("test", "en", SNAPSHOT)
            execute.assert_not_awaited()
            self.assertNotIn("execution", result)

    async def test_executor_mutation_cannot_change_logged_arguments(self):
        async def execute(service, arguments):
            arguments["entity_id"].clear()

        client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}])
        result = await AgentEngine(normalize_settings({}), client, execute=execute).process("Stop", "en", copy.deepcopy(SNAPSHOT), speaker_area="Office")
        self.assertEqual(result["execution"]["calls"][0]["arguments"]["entity_id"], ["light.desk", "light.ceiling"])
