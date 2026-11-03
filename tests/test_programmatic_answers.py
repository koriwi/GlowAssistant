import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

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
from scripts import local as benchmark
from tests.test_integration_engine import SNAPSHOT, ScriptedModels, entity


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
        self.assertEqual(format_values(selected_values([sensor], "state"), "de-DE"), "[Office]: Room temperature: 22.5 °C.")
        climate = entity("climate.room", "Thermostat", state="heat", temperature=24, current_temperature=22, temperature_unit="°C")
        self.assertEqual(selected_values([climate], "attribute:temperature")[0]["value"], 24)
        self.assertEqual(selected_values([climate], "attribute:current_temperature")[0]["value"], 22)
        rows = selected_values([climate], "attribute:temperature") + selected_values([climate], "attribute:current_temperature")
        self.assertEqual(format_values(rows, "de"), "[Office]: Thermostat: Solltemperatur: 24 °C, Gemessene Temperatur: 22 °C.")
        self.assertEqual(format_values(rows, "en"), "[Office]: Thermostat: Target temperature: 24 °C, Measured temperature: 22 °C.")

    def test_window_states_and_duplicate_names_include_room_context(self):
        windows = [entity("binary_sensor.left", "Window", state="on", area="office", device_class="window"),
                   entity("binary_sensor.right", "Window", state="off", area="bedroom", device_class="window")]
        rows = selected_values(windows, "state")
        self.assertEqual(format_values(rows, "de"), "[Office]: Window: offen. [Bedroom]: Window: geschlossen.")
        self.assertEqual(format_values(rows, "en"), "[Office]: Window: open. [Bedroom]: Window: closed.")

    def test_unavailable_unknown_and_missing_attributes_do_not_use_stale_values_or_main_state(self):
        entities = [entity("light.good", "Good", brightness=255), entity("light.missing", "Missing", state="on"),
                    entity("light.stale", "Stale", state="unavailable", brightness=255),
                    entity("light.unknown", "Unknown", state="unknown", brightness=128)]
        rows = selected_values(entities, "attribute:brightness")
        self.assertEqual([row["value"] for row in rows], [100, None, None, None])
        self.assertEqual([row["status"] for row in rows], ["available", "missing", "unavailable", "unknown"])
        self.assertEqual(format_values(rows, "de"), "[Office]: Good: 100 %; Missing: nicht verfügbar; Stale: nicht verfügbar; Unknown: unbekannt.")

    def test_entities_are_grouped_by_area_even_when_interleaved_and_unassigned_are_retained(self):
        rows = selected_values([
            entity("light.kitchen", "Ceiling", area="kitchen"),
            entity("light.office", "Desk", area="office", state="off"),
            entity("light.unassigned", "Portable", area=None),
            entity("light.kitchen_counter", "Counter", area="kitchen", state="off"),
            entity("light.unassigned_second", "Spare", area=None, state="off"),
        ], "state")
        self.assertEqual(format_values(rows, "en"),
                         "[Kitchen]: Ceiling: on; Counter: off. [Office]: Desk: off. [No area]: Portable: on; Spare: off.")
        self.assertEqual(format_values(rows, "de"),
                         "[Kitchen]: Ceiling: an; Counter: aus. [Office]: Desk: aus. [Ohne Bereich]: Portable: an; Spare: aus.")


class ProgrammaticAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_properties_share_one_choice_call_and_render_per_entity_and_area(self):
        snapshot = {"entities": [
            entity("light.office", "Lampe", area="office", brightness=128, color_temp_kelvin=3000),
            entity("light.kitchen", "Lampe", area="kitchen", brightness=255),
        ], "services": {}}
        probabilities = {"state": 0.02, "attribute:brightness": 0.35, "attribute:color_temp_kelvin": 0.25, "none": 0.38}
        client = ScriptedModels([
            {"service": "query_state"}, {"area": "all_areas"}, {"domain": "light"},
            {"query_type": "complex"}, {"targets": "all"},
            {"state_property": {"type": "choice", "choice": "none", "confidence": 0, "probabilities": probabilities}},
        ])
        events = []
        execute = AsyncMock()
        result = await AgentEngine(local_settings(), client, execute=execute, trace=lambda *event: events.append(event)).process(
            "Wie hell sind die Lampen und welche Farbtemperatur haben sie?", "de", snapshot)
        self.assertEqual(result["state_properties"], ["attribute:brightness", "attribute:color_temp_kelvin"])
        self.assertEqual(result["property_selection"]["probability_threshold"], 0.5)
        self.assertAlmostEqual(result["property_selection"]["effective_probability_threshold"], 0.5 / 3)
        self.assertEqual(result["answer"], "[Office]: Lampe: Helligkeit: 50.2 %, Farbtemperatur: 3000 K. "
                         "[Kitchen]: Lampe: Helligkeit: 100 %, Farbtemperatur: nicht verfügbar.")
        self.assertEqual(len(result["state_values"]), 4)
        missing = next(row for row in result["state_values"] if row["entity_id"] == "light.kitchen" and row["property"] == "attribute:color_temp_kelvin")
        self.assertEqual(missing["status"], "missing")
        self.assertIsNone(missing["value"])
        self.assertEqual(len(client.requests), 6)
        self.assertEqual({entity["entity_id"] for entity in client.requests[-1][1]["entities"]}, {"light.office", "light.kitchen"})
        self.assertTrue(all(entity["state"] == "on" for entity in client.requests[-1][1]["entities"]))
        self.assertIn("Choosing multiple properties is allowed", client.requests[-1][2]["state_property"]["instructions"])
        self.assertEqual(client.requests[-1][2]["state_property"]["criteria"]["none"], "No supplied state property matches the request.")
        output = next(output for stage, _, output in events if stage == "state_property" and output)
        self.assertEqual(output["selected_properties"], result["state_properties"])
        self.assertEqual(output["decisions"]["state_property"]["confidence"], 0)
        self.assertNotIn("minimum_confidence", output["decisions"]["state_property"])
        self.assertFalse(client.chat_requests)
        execute.assert_not_awaited()

    async def test_complex_query_selects_one_requested_property_without_chat(self):
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
        self.assertEqual(result["answer"], "[Office]: Desk lamp: 100 %; Ceiling lamp: 50 %.")
        self.assertEqual(result["answer_source"], "programmatic")
        self.assertEqual(result["state_properties"], ["attribute:brightness"])
        self.assertEqual([row["entity_id"] for row in result["state_values"]], ["light.desk", "light.ceiling"])
        self.assertEqual(len(client.requests), 6)
        self.assertEqual(set(client.requests[-1][2]), {"state_property"})
        self.assertEqual(client.requests[-1][1]["history"], history)
        self.assertEqual(client.requests[-1][1]["language"], "de")
        self.assertEqual([entity["entity_id"] for entity in client.requests[-1][1]["entities"]], ["light.desk", "light.ceiling"])
        self.assertEqual(client.requests[-1][1]["service"], "query_state")
        self.assertEqual(client.requests[-1][1]["entities"][0]["name"], "Desk lamp")
        self.assertNotIn("attributes", client.requests[-1][1]["entities"][0])
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

    async def test_command_defaults_to_okay_without_chat_or_device_execution(self):
        client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}])
        result = await AgentEngine(local_settings(), client, dry_run=True).process("Turn off the lights", "en", SNAPSHOT, speaker_area="Office")
        self.assertEqual(result["answer"], "Okay")
        self.assertEqual(result["answer_source"], "programmatic")
        self.assertEqual(len(client.requests), 2)
        self.assertFalse(client.chat_requests)
        self.assertEqual(result["execution"]["status"], "planned")

    async def test_custom_command_reply_is_used_for_live_and_planned_commands(self):
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                settings = local_settings()
                settings["ok_reply"] = "Alles klar."
                execute = AsyncMock(return_value=None)
                client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}])
                result = await AgentEngine(settings, client, dry_run=dry_run, execute=execute).process(
                    "Turn off the lights", "de", SNAPSHOT, speaker_area="Office")
                self.assertEqual(result["answer"], "Alles klar.")
                self.assertEqual(result["answer_source"], "programmatic")
                self.assertFalse(client.chat_requests)
                self.assertEqual(execute.await_count, 0 if dry_run else 1)

    async def test_custom_yes_no_replies_override_language_defaults_without_changing_the_decision(self):
        for language, answer, expected in (("de", "yes", "Jawohl!"), ("en", "yes", "Jawohl!"), ("de", "no", "Nö.")):
            with self.subTest(language=language, answer=answer):
                settings = local_settings()
                settings.update(yes_reply="Jawohl!", no_reply="Nö.")
                client = ScriptedModels([{"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
                                         {"query_type": "yes_no"}, {"binary_answer": answer}])
                result = await AgentEngine(settings, client, dry_run=True).process("Is the office light on?", language, SNAPSHOT)
                self.assertEqual(result["answer"], expected)
                self.assertEqual(result["state_answer"], answer)
                self.assertFalse(client.chat_requests)

    async def test_enabled_yes_no_answer_model_keeps_generating_the_reply(self):
        settings = local_settings(True)
        settings.update(yes_reply="Custom yes", no_reply="Custom no")
        client = ScriptedModels([{"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
                                 {"query_type": "yes_no"}, {"binary_answer": "yes"}], answers=["Ja."])
        result = await AgentEngine(settings, client, dry_run=True).process("Ist das Licht an?", "de", SNAPSHOT)
        self.assertEqual(result["answer"], "Ja.")
        self.assertEqual(result["answer_source"], "llm")
        self.assertEqual(json.loads(client.chat_requests[0][1][1]["content"]), {"state_answer": "yes"})

    async def test_enabled_answer_llm_receives_a_command_plan_without_claiming_execution(self):
        client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}], answers=["I would turn off the lights."])
        settings = local_settings(True)
        settings["ok_reply"] = "Custom acknowledgment"
        result = await AgentEngine(settings, client, dry_run=True).process("Turn off the lights", "en", SNAPSHOT, speaker_area="Office")
        self.assertEqual(result["answer_source"], "llm")
        self.assertEqual(result["answer"], "I would turn off the lights.")
        self.assertEqual(len(client.chat_requests), 1)
        messages = client.chat_requests[0][1]
        self.assertIn("No device command was sent", messages[0]["content"])
        self.assertEqual(json.loads(messages[1]["content"])["execution"]["status"], "planned")
        self.assertEqual(result["execution"]["status"], "planned")

    async def test_unsure_property_stops_instead_of_answering_with_an_unrelated_value(self):
        client = ScriptedModels([{"service": "query_state"}, {"area": "area:office"}, {"domain": "sensor"},
                                 {"query_type": "complex"}, {"targets": ["sensor.office"]}, {"state_property": "none"},
                                 {"query_fallback": "none"}])
        result = await AgentEngine(local_settings(), client, dry_run=True).process("What is its unsupported value?", "en", SNAPSHOT)
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["query_fallback"]["trigger"]["stage"], "state_property")
        self.assertEqual(result["failures"][0]["stage"], "query_fallback")
        self.assertNotIn("state_values", result)
        self.assertFalse(client.chat_requests)

    async def test_benchmark_requires_flag_even_if_ha_answer_generation_is_enabled(self):
        choices = [{"service": "query_state"}, {"area": "area:office"}, {"domain": "sensor"},
                   {"query_type": "complex"}, {"targets": ["sensor.office"]}, {"state_property": "state"}]
        client = ScriptedModels(choices)
        args = benchmark.parser().parse_args(["-q", "Wie warm ist es im Büro?"])
        exported = {"entry_id": "one", "settings": local_settings(True), "snapshot": copy.deepcopy(SNAPSHOT)}
        with (patch("scripts.local.fetch_snapshot", new=AsyncMock(return_value=exported)),
              patch("scripts.local.aiohttp.ClientSession", return_value=AsyncMock()), patch("scripts.local.ModelClient", return_value=client)):
            report = await benchmark.run(args, {})
        self.assertEqual(report["result"]["answer"], "[Office]: Room temperature: 22 °C.")
        self.assertFalse(client.chat_requests)
        self.assertIsNone(benchmark.answer_input(report))
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.print_report(report, show_answer_input=True)
        self.assertIn("programmatic", output.getvalue())
        self.assertIn("state_property", output.getvalue())
        self.assertIn("Answer model was not called.", output.getvalue())


class AnswerConfigurationTests(unittest.TestCase):
    def test_property_threshold_reuses_settings_and_legacy_flag_without_global_confidence_override(self):
        from custom_components.glow_assistant.state_properties import (
            INSTRUCTIONS,
            LEGACY_INSTRUCTIONS,
            LEGACY_MULTI_INSTRUCTIONS,
        )

        settings = local_settings()
        settings["decider"].update(state_property_confidence=0.25, state_property_prompt=LEGACY_INSTRUCTIONS)
        args = benchmark.parser().parse_args(["-q", "test", "--confidence", "0.9"])
        configured = benchmark.model_settings(settings, args, {})["decider"]
        self.assertEqual(configured["state_property_confidence"], 0.25)
        self.assertEqual(configured["state_property_prompt"], INSTRUCTIONS)
        self.assertEqual(normalize_settings({"decider": {"state_property_prompt": LEGACY_MULTI_INSTRUCTIONS}})["decider"]["state_property_prompt"], INSTRUCTIONS)
        self.assertEqual(configured["service_confidence"], 0.9)
        self.assertEqual(normalize_settings({"decider": {"state_property_confidence": 0}})["decider"]["state_property_confidence"], 0)
        for flag in ("--state-property-threshold", "--state-property-confidence"):
            with self.subTest(flag=flag):
                args = benchmark.parser().parse_args(["-q", "test", "--confidence", "0.9", flag, "0.6"])
                self.assertEqual(benchmark.model_settings(settings, args, {})["decider"]["state_property_confidence"], 0.6)
        custom = normalize_settings({"decider": {"state_property_prompt": "My property instructions"}})
        self.assertEqual(custom["decider"]["state_property_prompt"], "My property instructions")

    def test_reply_defaults_validation_and_exported_ha_overrides(self):
        settings = local_settings()
        self.assertEqual(settings["ok_reply"], "Okay")
        self.assertEqual(settings["yes_reply"], "")
        self.assertEqual(settings["no_reply"], "")
        settings.update(ok_reply="Erledigt.", yes_reply="Jawohl.", no_reply="Nö.")
        args = benchmark.parser().parse_args(["-q", "test"])
        configured = benchmark.model_settings(settings, args, {})
        for field in ("ok_reply", "yes_reply", "no_reply"):
            self.assertEqual(configured[field], settings[field])
        for value in ("", "   ", None, True):
            invalid = local_settings()
            invalid["ok_reply"] = value
            with self.subTest(ok_reply=value), self.assertRaises(ValueError):
                validate_settings(invalid)
        for field in ("yes_reply", "no_reply"):
            for value in (None, True, 1):
                invalid = local_settings()
                invalid[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_settings(invalid)
        settings.update(yes_reply="   ", no_reply="")
        self.assertEqual(validate_settings(settings)["yes_reply"], "")

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
