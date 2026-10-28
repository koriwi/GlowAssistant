import copy
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.engine import (
    ActionError,
    AgentEngine,
    action_options,
    read_choice,
)
from custom_components.glow_assistant.settings import (
    DEFAULTS,
    normalize_settings,
    validate_settings,
)


def entity(entity_id, name, area="office", state="on", **attributes):
    return {
        "entity_id": entity_id, "name": name, "aliases": [], "area_id": area,
        "area_name": area.title() if area else None, "area_aliases": [],
        "domain": entity_id.split(".")[0], "state": state, "attributes": attributes,
    }


SNAPSHOT = {
    "entities": [entity("light.desk", "Desk lamp"), entity("light.ceiling", "Ceiling lamp"),
                 entity("light.bed", "Bed lamp", "bedroom"),
                 entity("sensor.office", "Room temperature", state="22", unit_of_measurement="°C")],
    "services": {
        "light.turn_off": {"target": {}, "fields": {}},
        "light.turn_on": {"target": {}, "fields": {"brightness_pct": {}}},
    },
}


class ScriptedModels:
    def __init__(self, choices, answers=None, confidence=0.9):
        self.choices = list(choices)
        self.requests = []
        self.answers = list(answers or ["Erledigt."])
        self.chat_requests = []
        self.confidence = confidence

    async def decide(self, config, state, questions):
        self.requests.append(copy.deepcopy((config, state, questions)))
        choices = self.choices.pop(0)
        answers = {}
        for name, question in questions.items():
            if question["type"] == "score":
                score = choices[name]
                probabilities = {str(index): 0 for index in range(len(question["criteria"]))}
                lower = int(score)
                probabilities[str(lower)] = 1 - (score - lower)
                if score != lower:
                    probabilities[str(lower + 1)] = score - lower
                answers[name] = {"type": "score", "score": score, "confidence": self.confidence, "probabilities": probabilities}
            else:
                choice = choices.get(name, "unsure")
                probabilities = dict.fromkeys(question["criteria"], 0)
                if choice in probabilities:
                    probabilities[choice] = 1
                answers[name] = {"type": "choice", "choice": choice, "confidence": self.confidence, "probabilities": probabilities}
        return {"answers": answers, "usage": {"input_tokens": 5}}

    async def chat(self, config, messages):
        self.chat_requests.append(copy.deepcopy((config, messages)))
        reply = self.answers.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, {"completion_tokens": 5}


class IntegrationEngineTests(unittest.IsolatedAsyncioTestCase):
    async def run_agent(self, choices, snapshot=SNAPSHOT, settings=None, answers=None, executor=None):
        models = ScriptedModels(choices, answers)
        execute = executor or AsyncMock()
        agent = AgentEngine(settings or normalize_settings({}), models, execute)
        result = await agent.process("test request", "de", snapshot)
        return result, models, execute

    async def test_direct_action_uses_its_domain_and_filtered_room_targets(self):
        result, models, execute = await self.run_agent([
            {"service": "light.turn_off", "domain": "unsure"},
            {"area": "area:office"}, {"target": "area:office"},
        ])
        execute.assert_awaited_once_with("light.turn_off", {"entity_id": ["light.desk", "light.ceiling"]})
        self.assertEqual(result["domain"], "light")
        self.assertEqual(result["answer"], "Erledigt.")
        self.assertEqual(models.requests[0][1], {"prompt": "test request"})
        self.assertEqual(models.requests[1][1], {"prompt": "test request"})
        criteria = models.requests[2][2]["target"]["criteria"]
        self.assertEqual(set(criteria), {"entity:light.desk", "entity:light.ceiling", "area:office", "unsure"})
        self.assertNotIn("attributes", json.loads(criteria["entity:light.desk"]))
        facts = json.loads(models.chat_requests[0][1][1]["content"])
        self.assertEqual(facts["execution"]["status"], "completed")
        self.assertNotIn("decisions", facts)

    async def test_query_uses_cached_state_and_does_not_execute(self):
        result, models, execute = await self.run_agent([
            {"service": "query_state", "domain": "sensor"},
            {"area": "area:office"}, {"target": "entity:sensor.office"},
        ], answers=["Es sind 22 Grad."])
        execute.assert_not_awaited()
        self.assertEqual(result["state_info"][0]["state"], "22")
        facts = json.loads(models.chat_requests[0][1][1]["content"])
        self.assertEqual(facts["state_info"][0]["attributes"]["unit_of_measurement"], "°C")
        self.assertEqual(result["answer"], "Es sind 22 Grad.")

    async def test_unavailable_query_omits_stale_attributes(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["entities"][-1]["state"] = "unavailable"
        result, _, _ = await self.run_agent([
            {"service": "query_state", "domain": "sensor"},
            {"area": "no_area"}, {"target": "entity:sensor.office"},
        ], snapshot=snapshot)
        self.assertEqual(result["state_info"][0]["attributes"], {})

    async def test_unsure_at_each_stage_bypasses_chat_and_execution(self):
        cases = [
            [{"service": "unsure"}],
            [{"service": "query_state", "domain": "unsure"}],
            [{"service": "light.turn_off"}, {"area": "unsure"}],
            [{"service": "light.turn_off"}, {"area": "no_area"}, {"target": "unsure"}],
        ]
        for choices in cases:
            with self.subTest(choices=choices):
                result, models, execute = await self.run_agent(choices)
                self.assertEqual(result["answer"], "ich bin mir nicht sicher")
                self.assertEqual(result["route"], "unsure")
                self.assertFalse(models.chat_requests)
                execute.assert_not_awaited()

    async def test_disabled_general_returns_fixed_unsure(self):
        result, models, execute = await self.run_agent([{"service": "not-smarthome-related"}])
        self.assertEqual(result["answer"], "ich bin mir nicht sicher")
        self.assertFalse(models.chat_requests)
        execute.assert_not_awaited()

    async def test_enabled_general_uses_its_own_model_and_original_request(self):
        settings = normalize_settings({"general": {"enable": True, "model": "smart-model"}})
        result, models, execute = await self.run_agent(
            [{"service": "not-smarthome-related"}], settings=settings, answers=["330 Meter."],
        )
        self.assertEqual(result["route"], "not-smarthome-related")
        self.assertEqual(len(models.requests), 1)
        self.assertEqual(models.chat_requests[0][0]["model"], "smart-model")
        self.assertEqual(models.chat_requests[0][1][1]["content"], "test request")
        self.assertEqual(result["answer"], "330 Meter.")
        execute.assert_not_awaited()

    async def test_confidence_threshold_stops_action(self):
        settings = normalize_settings({"decider": {"service_confidence": 0.95}})
        result, models, execute = await self.run_agent([{"service": "light.turn_off"}], settings=settings)
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(len(models.requests), 1)
        execute.assert_not_awaited()

    async def test_no_matching_domain_in_area_stops_selection(self):
        result, models, execute = await self.run_agent([
            {"service": "query_state", "domain": "sensor"}, {"area": "area:bedroom"},
        ])
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(len(models.requests), 2)
        execute.assert_not_awaited()

    async def test_answer_failure_after_action_preserves_outcome_and_never_retries(self):
        result, models, execute = await self.run_agent([
            {"service": "light.turn_off"}, {"area": "no_area"}, {"target": "entity:light.desk"},
        ], answers=[ModelError("No answer")])
        execute.assert_awaited_once()
        self.assertEqual(result["execution"]["status"], "completed")
        self.assertEqual(result["answer"], DEFAULTS["answer_error_reply"])
        self.assertTrue(result["answer_error"])
        self.assertEqual(len(models.chat_requests), 1)

    async def test_action_failure_is_reported_as_unconfirmed(self):
        execute = AsyncMock(side_effect=ActionError("partial failure"))
        result, models, _ = await self.run_agent([
            {"service": "light.turn_off"}, {"area": "no_area"}, {"target": "entity:light.desk"},
        ], executor=execute)
        execute.assert_awaited_once()
        self.assertEqual(result["execution"]["status"], "error")
        facts = json.loads(models.chat_requests[0][1][1]["content"])
        self.assertIn("unconfirmed", facts["execution"]["error"])

    async def run_numeric(self, currents=(21,), score=4.02, mode="relative", parameter="temperature", settings=None):
        climate = parameter == "temperature"
        name = "climate.set_temperature" if climate else "light.turn_on"
        field = "temperature" if climate else "brightness_pct"
        snapshot = {
            "entities": [entity(f"{'climate' if climate else 'light'}.device_{index}", f"Device {index}",
                                temperature=current, brightness=current, min_temp=5, max_temp=45, target_temp_step=1)
                         for index, current in enumerate(currents)],
            "services": {name: {"target": {}, "fields": {field: {"required": climate}}}},
        }
        return await self.run_agent([
            {"service": f"{name}:{field}"}, {"area": "area:office"}, {"target": "area:office"},
            {"numeric_mode": mode, "numeric_score": score},
        ], snapshot=snapshot, settings=settings)

    async def test_temperature_is_quantized_and_relative_direction_preserved(self):
        result, models, execute = await self.run_numeric()
        execute.assert_awaited_once_with("climate.set_temperature", {"entity_id": ["climate.device_0"], "temperature": 22})
        self.assertEqual(result["route"], "smarthome")
        state = models.requests[3][1]
        self.assertEqual((state["minimum"], state["maximum"], state["current"], state["baseline"]), (16, 26, 21, "target"))

    async def test_fractional_baseline_does_not_reverse_relative_direction(self):
        _, _, execute = await self.run_numeric(currents=(21.01,), score=4.02)
        self.assertEqual(execute.await_args.args[1]["temperature"], 22)

    async def test_relative_area_with_different_settings_is_unsure(self):
        result, models, execute = await self.run_numeric(currents=(20, 22))
        self.assertEqual(result["route"], "unsure")
        self.assertFalse(models.chat_requests)
        execute.assert_not_awaited()

    async def test_absolute_area_can_set_a_common_temperature(self):
        _, _, execute = await self.run_numeric(currents=(20, 22), score=4.8, mode="absolute")
        execute.assert_awaited_once_with("climate.set_temperature", {
            "entity_id": ["climate.device_0", "climate.device_1"], "temperature": 22,
        })

    async def test_brightness_uses_percent_and_configured_default_change(self):
        _, models, execute = await self.run_numeric(currents=(127.5,), score=3, parameter="brightness")
        self.assertEqual(models.requests[3][1]["current"], 50)
        self.assertEqual(models.requests[3][1]["default_step"], 10)
        execute.assert_awaited_once_with("light.turn_on", {"entity_id": ["light.device_0"], "brightness_pct": 60})

    async def test_numeric_unsure_skips_action_and_answer_model(self):
        result, models, execute = await self.run_numeric(mode="unsure")
        self.assertEqual(result["answer"], "ich bin mir nicht sicher")
        self.assertFalse(models.chat_requests)
        execute.assert_not_awaited()


class IntegrationSettingsTests(unittest.TestCase):
    def local_settings(self):
        return normalize_settings({
            "decider": {"url": "http://server:8321/v1/systemone", "api_key": ""},
            "answer": {"url": "http://server:8322/v1/chat/completions", "model": "small-model"},
        })

    def test_local_models_and_disabled_general_need_no_keys(self):
        settings = validate_settings(self.local_settings())
        self.assertFalse(settings["general"]["enable"])
        self.assertEqual(settings["decider"]["api_key"], "")

    def test_ui_numeric_selectors_produce_integer_token_limits(self):
        settings = self.local_settings()
        settings["answer"]["max_tokens"] = 512.0
        settings["decider"]["max_input_chars"] = 100000.0
        saved = validate_settings(settings)
        self.assertIsInstance(saved["answer"]["max_tokens"], int)
        self.assertIsInstance(saved["decider"]["max_input_chars"], int)

    def test_prompt_reset_and_customization_survive_normalization(self):
        settings = normalize_settings({"decider": {"target_prompt": "My rules", "area_prompt": " "}})
        self.assertEqual(settings["decider"]["target_prompt"], "My rules")
        self.assertEqual(settings["decider"]["area_prompt"], DEFAULTS["decider"]["area_prompt"])
        self.assertNotEqual(id(settings["decider"]), id(DEFAULTS["decider"]))

    def test_invalid_ranges_and_models_are_rejected(self):
        for group, field, value in (
            ("decider", "temperature_min", 27), ("decider", "temperature_levels", 11),
            ("decider", "temperature_levels", 2.5), ("decider", "temperature_step", 0),
            ("decider", "service_confidence", 1.1), ("decider", "url", "localhost:8321"),
            ("answer", "model", ""), ("answer", "max_tokens", 0),
            ("answer", "max_input_chars", float("inf")), ("answer", "temperature", True),
        ):
            settings = self.local_settings()
            settings[group][field] = value
            with self.subTest(group=group, field=field), self.assertRaises(ValueError):
                validate_settings(settings)
        settings = self.local_settings()
        settings["general"]["enable"] = True
        with self.assertRaisesRegex(ValueError, "general"):
            validate_settings(settings)

    def test_cloud_keys_are_required_per_model_without_cross_provider_fallback(self):
        settings = self.local_settings()
        settings["answer"]["url"] = "https://openrouter.ai/api/v1/chat/completions"
        settings["decider"]["api_key"] = "decider-only-key"
        with self.assertRaisesRegex(ValueError, "answer"):
            validate_settings(settings)

    def test_action_catalog_omits_unsupported_required_values(self):
        options = action_options({
            "light.turn_on": {"fields": {}}, "climate.set_temperature": {"fields": {}},
            "climate.set_hvac_mode": {"fields": {"hvac_mode": {"required": True}}},
            "todo.get_items": {"fields": {}, "response": {"optional": False}},
        })
        self.assertIn("light.turn_on", options)
        self.assertIn("light.turn_on:brightness_pct", options)
        self.assertIn("climate.set_temperature:temperature", options)
        self.assertNotIn("climate.set_temperature", options)
        self.assertNotIn("climate.set_hvac_mode", options)
        self.assertNotIn("todo.get_items", options)

    def test_legacy_other_and_malformed_responses_are_errors(self):
        response = {"answers": {"target": {"type": "choice", "choice": "other", "confidence": 1, "probabilities": {"unsure": 1}}}}
        with self.assertRaises(ModelError):
            read_choice(response, "target", {}, 0)

    def test_translation_fields_match_settings_blocks(self):
        root = Path(__file__).parent / "custom_components/glow_assistant"
        strings = json.loads((root / "strings.json").read_text())
        self.assertEqual(strings, json.loads((root / "translations/en.json").read_text()))
        for group in ("decider", "answer", "general"):
            labels = strings["config"]["step"]["user"]["sections"][group]["data"]
            self.assertEqual(set(labels), set(DEFAULTS[group]) | {"test_connection"})
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertTrue(manifest["config_flow"])
        self.assertEqual(manifest["domain"], "glow_assistant")


if __name__ == "__main__":
    unittest.main()
