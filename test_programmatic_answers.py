import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

import local as benchmark
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.settings import (
    normalize_settings,
    validate_settings,
)
from custom_components.glow_assistant.state_properties import (
    format_values,
    property_options,
    selected_values,
)
from test_integration_engine import SNAPSHOT, ScriptedModels, entity


def local_settings(enabled=False):
    return normalize_settings({"decider": {"url": "http://local/v1/systemone", "model": "decider"},
                               "answer": {"enable": enabled, "url": "http://local/v1/chat/completions", "model": "answer"}})


class StatePropertyTests(unittest.TestCase):
    def test_properties_include_requested_atomic_values_and_exclude_capability_metadata(self):
        entities = [entity("light.desk", "Desk", brightness=128, hs_color=[120, 80], color_temp_kelvin=3000,
                           supported_features=1, supported_color_modes=["hs"], effect_list=["rainbow"], effect="none")]
        options = property_options(entities)
        self.assertEqual(set(options), {"state", "attribute:brightness", "attribute:color_temp_kelvin", "attribute:effect", "hue", "saturation"})
        self.assertEqual(selected_values(entities, "attribute:brightness")[0]["value"], 50.2)
        self.assertEqual(selected_values(entities, "hue")[0]["value"], 120)
        self.assertEqual(selected_values(entities, "saturation")[0]["unit"], "%")
        self.assertEqual(selected_values(entities, "attribute:color_temp_kelvin")[0]["unit"], "K")

    def test_sensor_measurement_and_climate_target_and_current_temperature_remain_distinct(self):
        sensor = entity("sensor.room", "Room temperature", state="22.5", unit_of_measurement="°C")
        self.assertEqual(format_values(selected_values([sensor], "state"), "de-DE"), "Room temperature: 22.5 °C.")
        climate = entity("climate.room", "Thermostat", state="heat", temperature=24, current_temperature=22, temperature_unit="°C")
        self.assertEqual(selected_values([climate], "attribute:temperature")[0]["value"], 24)
        self.assertEqual(selected_values([climate], "attribute:current_temperature")[0]["value"], 22)

    def test_window_states_and_duplicate_names_include_room_context(self):
        windows = [entity("binary_sensor.left", "Window", state="on", area="office", device_class="window"),
                   entity("binary_sensor.right", "Window", state="off", area="bedroom", device_class="window")]
        rows = selected_values(windows, "state")
        self.assertEqual(format_values(rows, "de"), "Window (Office): offen; Window (Bedroom): geschlossen.")
        self.assertEqual(format_values(rows, "en"), "Window (Office): open; Window (Bedroom): closed.")

    def test_unavailable_unknown_and_missing_attributes_do_not_use_stale_values_or_main_state(self):
        entities = [entity("light.good", "Good", brightness=255), entity("light.missing", "Missing", state="on"),
                    entity("light.stale", "Stale", state="unavailable", brightness=255),
                    entity("light.unknown", "Unknown", state="unknown", brightness=128)]
        rows = selected_values(entities, "attribute:brightness")
        self.assertEqual([row["value"] for row in rows], [100, None, None, None])
        self.assertEqual([row["status"] for row in rows], ["available", "missing", "unavailable", "unknown"])
        self.assertEqual(format_values(rows, "de"), "Good: 100 %; Missing: nicht verfügbar; Stale: nicht verfügbar; Unknown: unbekannt.")


class ProgrammaticAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_complex_query_uses_one_property_choice_for_all_selected_entities_without_chat(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["entities"][0]["attributes"]["brightness"] = 255
        snapshot["entities"][1]["attributes"]["brightness"] = 127.5
        client = ScriptedModels([{"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
                                 {"query_type": "complex"}, {"targets": ["light.desk", "light.ceiling"]},
                                 {"state_property": "attribute:brightness"}])
        events = []
        history = [{"role": "user", "content": "The office lights"}]
        settings = local_settings()
        settings["history_enabled"] = True
        result = await AgentEngine(settings, client, trace=lambda *event: events.append(event), history=history, dry_run=True).process(
            "Wie hell sind sie?", "de", snapshot)
        self.assertEqual(result["answer"], "Desk lamp: 100 %; Ceiling lamp: 50 %.")
        self.assertEqual(result["answer_source"], "programmatic")
        self.assertEqual(result["state_property"], "attribute:brightness")
        self.assertEqual([row["entity_id"] for row in result["state_values"]], ["light.desk", "light.ceiling"])
        self.assertEqual(len(client.requests), 6)
        self.assertEqual(set(client.requests[-1][2]), {"state_property"})
        self.assertEqual(client.requests[-1][1]["history"], history)
        self.assertEqual(client.requests[-1][1]["language"], "de")
        self.assertNotIn("entities", client.requests[-1][1])
        self.assertFalse(client.chat_requests)
        output = next(output for stage, _, output in events if stage == "answer" and output)
        self.assertFalse(output["model_call"])
        self.assertEqual(output["mode"], "programmatic")

    async def test_yes_no_shortcut_returns_fixed_pipeline_language_reply_without_property_or_chat_call(self):
        for language, answer, expected in (("de-DE", "yes", "Ja."), ("de", "no", "Nein."), ("en", "yes", "Yes.")):
            with self.subTest(language=language, answer=answer):
                client = ScriptedModels([{"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
                                         {"query_type": "yes_no"}, {"binary_answer": answer}])
                result = await AgentEngine(local_settings(), client, dry_run=True).process("Is the office light on?", language, SNAPSHOT)
                self.assertEqual(result["answer"], expected)
                self.assertEqual(len(client.requests), 5)
                self.assertNotIn("state_property", result["decisions"])
                self.assertNotIn("target_selection", result)
                self.assertFalse(client.chat_requests)

    async def test_command_defaults_to_ok_without_chat_or_device_execution(self):
        client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}])
        result = await AgentEngine(local_settings(), client, dry_run=True).process("Turn off the lights", "en", SNAPSHOT, speaker_area="Office")
        self.assertEqual(result["answer"], "ok")
        self.assertEqual(result["answer_source"], "programmatic")
        self.assertEqual(len(client.requests), 2)
        self.assertFalse(client.chat_requests)
        self.assertEqual(result["execution"]["status"], "planned")

    async def test_enabled_answer_llm_receives_a_command_plan_without_claiming_execution(self):
        client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}], answers=["I would turn off the lights."])
        result = await AgentEngine(local_settings(True), client, dry_run=True).process("Turn off the lights", "en", SNAPSHOT, speaker_area="Office")
        self.assertEqual(result["answer_source"], "llm")
        self.assertEqual(result["answer"], "I would turn off the lights.")
        self.assertEqual(len(client.chat_requests), 1)
        messages = client.chat_requests[0][1]
        self.assertIn("No device command was sent", messages[0]["content"])
        self.assertEqual(json.loads(messages[1]["content"])["execution"]["status"], "planned")
        self.assertEqual(result["execution"]["status"], "planned")

    async def test_unsure_property_stops_instead_of_answering_with_an_unrelated_value(self):
        client = ScriptedModels([{"service": "query_state"}, {"area": "area:office"}, {"domain": "sensor"},
                                 {"query_type": "complex"}, {"targets": ["sensor.office"]}, {"state_property": "unsure"}])
        result = await AgentEngine(local_settings(), client, dry_run=True).process("What is its unsupported value?", "en", SNAPSHOT)
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["failures"][0]["stage"], "state_property")
        self.assertNotIn("state_values", result)
        self.assertFalse(client.chat_requests)

    async def test_benchmark_requires_flag_even_if_ha_answer_generation_is_enabled(self):
        choices = [{"service": "query_state"}, {"area": "area:office"}, {"domain": "sensor"},
                   {"query_type": "complex"}, {"targets": ["sensor.office"]}, {"state_property": "state"}]
        client = ScriptedModels(choices)
        args = benchmark.parser().parse_args(["-q", "Wie warm ist es im Büro?"])
        exported = {"entry_id": "one", "settings": local_settings(True), "snapshot": copy.deepcopy(SNAPSHOT)}
        with (patch("local.fetch_snapshot", new=AsyncMock(return_value=exported)),
              patch("local.aiohttp.ClientSession", return_value=AsyncMock()), patch("local.ModelClient", return_value=client)):
            report = await benchmark.run(args, {})
        self.assertEqual(report["result"]["answer"], "Room temperature: 22 °C.")
        self.assertFalse(client.chat_requests)
        self.assertIsNone(benchmark.answer_input(report))
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.print_report(report, show_answer_input=True)
        self.assertIn("programmatic", output.getvalue())
        self.assertIn("state_property", output.getvalue())
        self.assertIn("Answer model was not called.", output.getvalue())


class AnswerConfigurationTests(unittest.TestCase):
    def test_answer_model_is_disabled_by_default_and_needs_no_endpoint_or_key(self):
        settings = normalize_settings({"decider": {"url": "http://local/v1/systemone", "model": "decider"},
                                       "answer": {"url": "", "model": "", "api_key": ""}})
        saved = validate_settings(settings)
        self.assertFalse(saved["answer"]["enable"])
        saved["answer"]["enable"] = True
        with self.assertRaisesRegex(ValueError, "answer:"):
            validate_settings(saved)
        for value in ("false", 0, None):
            settings["answer"]["enable"] = value
            with self.subTest(value=value), self.assertRaises(TypeError):
                validate_settings(settings)

    def test_cli_flag_controls_answer_mode_without_changing_general_enable(self):
        settings = local_settings(True)
        args = benchmark.parser().parse_args(["-q", "test"])
        self.assertFalse(benchmark.model_settings(settings, args, {})["answer"]["enable"])
        args = benchmark.parser().parse_args(["-q", "test", "--llm-answer"])
        configured = benchmark.model_settings(settings, args, {})
        self.assertTrue(configured["answer"]["enable"])
        self.assertFalse(configured["general"]["enable"])


if __name__ == "__main__":
    unittest.main()
