import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.run_logs import successful
from custom_components.glow_assistant.settings import (
    normalize_settings,
    validate_settings,
)
from scripts import local
from tests.test_integration_engine import ScriptedModels, entity

SNAPSHOT = {
    "entities": [
        entity("lawn_mower.robot", "Mower", area="garden", state="docked"),
        entity("sensor.letztes_mal_gemaht", "Last mowed", area=None, state="2026-11-02T10:00:00Z", device_class="timestamp"),
        entity("light.office", "Office light", state="on"),
    ],
    "services": {},
}
WRONG_DOMAIN = [
    {"service": "query_state"}, {"area": "area:garden"}, {"domain": "lawn_mower"},
    {"query_type": "complex"}, {"targets": []},
]


def fallback_answer(probabilities, confidence=0.9):
    scores = dict.fromkeys([f"entity:{item['entity_id']}" for item in SNAPSHOT["entities"]] + ["none"], 0)
    scores.update(probabilities)
    return {"query_fallback": {"type": "choice", "choice": max(scores, key=scores.get),
                                "probabilities": scores, "confidence": confidence}}


class QueryFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrong_domain_and_area_recover_reporting_sensor_without_more_filters(self):
        for language, query in (("en", "When was the lawn mowed last?"), ("de", "Wann wurde zuletzt gemäht?")):
            with self.subTest(language=language):
                models = ScriptedModels(WRONG_DOMAIN + [
                    {"query_fallback": "entity:sensor.letztes_mal_gemaht"}, {"state_property": "state"},
                ])
                events = []
                execute = AsyncMock()
                settings = normalize_settings({"history_enabled": True})
                history = [{"role": "user", "content": "The mower."}, {"role": "assistant", "content": "Okay"}]
                result = await AgentEngine(settings, models, history=history, execute=execute,
                                           trace=lambda *event: events.append(event)).process(query, language, SNAPSHOT)
                recovery = result["query_fallback"]
                self.assertEqual(result["route"], "smarthome")
                self.assertEqual(result["domain"], "sensor")
                self.assertEqual(result["domain_filter"]["selected_domain"], "lawn_mower")
                self.assertEqual(result["target"]["entity_id"], "sensor.letztes_mal_gemaht")
                self.assertEqual([e["entity_id"] for e in result["state_info"]], ["sensor.letztes_mal_gemaht"])
                self.assertIn("2026-11-02T10:00:00Z", result["answer"])
                self.assertEqual(recovery["status"], "completed")
                self.assertEqual(recovery["trigger"]["stage"], "target")
                self.assertEqual(recovery["entities_before"], 3)
                self.assertAlmostEqual(recovery["effective_probability_threshold"], 0.5 / 3)
                self.assertTrue(successful(result))
                request = models.requests[5]
                self.assertEqual(set(request[2]["query_fallback"]["criteria"]), {
                    "entity:lawn_mower.robot", "entity:sensor.letztes_mal_gemaht", "entity:light.office", "none",
                })
                self.assertEqual(request[1]["history"], history)
                self.assertEqual(request[1]["language"], language)
                self.assertNotIn("area", request[1])
                self.assertNotIn("domain", request[1])
                property_state = models.requests[-1][1]
                self.assertEqual(len(property_state["entities"]), 1)
                selected = property_state["entities"][0]
                self.assertEqual(selected["entity_id"], "sensor.letztes_mal_gemaht")
                self.assertEqual(selected["name"], "Last mowed")
                self.assertEqual(selected["device_class"], "timestamp")
                self.assertEqual(selected["state"], "2026-11-02T10:00:00Z")
                self.assertIsNone(selected["area"])
                self.assertEqual(property_state["domain"], "sensor")
                self.assertEqual(property_state["language"], language)
                self.assertEqual(property_state["history"], history)
                self.assertEqual(len(models.requests), 7)
                self.assertIn("state_property:fallback", result["timings_ms"])
                output = next(output for stage, _, output in events if stage == "query_fallback" and output)
                self.assertEqual(output["trigger"]["stage"], "target")
                decision = output["decisions"]["query_fallback"]
                self.assertEqual(decision["minimum_confidence"], 0.5)
                self.assertTrue(decision["confidence_gate_passed"])
                self.assertIn("probability_threshold", decision)
                report = {"result": result, "context": {"entities": 3, "actions": 0},
                          "trace": [{"stage": stage, "input": args, "output": output or {}} for stage, args, output in events],
                          "timings_ms": {"snapshot": 0, "engine": 0, "total": 0}}
                with redirect_stdout(io.StringIO()) as printed:
                    local.print_report(report)
                self.assertIn("completed; triggered by target", printed.getvalue())
                self.assertIn("selected 1/3", printed.getvalue())
                self.assertFalse(models.chat_requests)
                execute.assert_not_awaited()

    async def test_low_confidence_none_and_empty_selection_reject_without_retry_or_answer(self):
        cases = [
            fallback_answer({"entity:sensor.letztes_mal_gemaht": 1}, confidence=0.799),
            fallback_answer({"none": 0.6, "entity:sensor.letztes_mal_gemaht": 0.4}),
            fallback_answer({"none": 1}),
        ]
        for response in cases:
            with self.subTest(response=response):
                models = ScriptedModels(WRONG_DOMAIN + [response])
                execute = AsyncMock()
                result = await AgentEngine(normalize_settings({"decider": {"confidence": 0.8}}), models, execute=execute).process("Last mowed?", "en", SNAPSHOT)
                self.assertEqual(result["route"], "unsure")
                self.assertEqual(result["query_fallback"]["selected_entity_ids"], [])
                self.assertEqual(result["failures"][0]["stage"], "query_fallback")
                self.assertEqual(len(models.requests), 6)
                self.assertNotIn("state_info", result)
                self.assertFalse(models.chat_requests)
                self.assertFalse(successful(result))
                execute.assert_not_awaited()

    async def test_fallback_accepts_multiple_domains_and_confidence_equal_to_gate(self):
        models = ScriptedModels([
            {"service": "query_state"}, {"area": "none"},
            fallback_answer({"entity:sensor.letztes_mal_gemaht": 0.5, "entity:light.office": 0.5}, confidence=0.8),
            {"query_type": "complex"}, {"state_property": "state"},
        ])
        result = await AgentEngine(normalize_settings({}), models).process("Those two states?", "en", SNAPSHOT)
        self.assertIsNone(result["domain"])
        self.assertEqual(result["domains"], ["light", "sensor"])
        self.assertEqual(result["query_fallback"]["entities_after"], 2)
        self.assertEqual(len(result["state_info"]), 2)
        self.assertNotIn("domain_filter", result)
        self.assertEqual(set(models.requests[3][2]), {"query_type"})
        self.assertEqual(len(models.requests[3][1]["entities"]), 2)

    async def test_domain_and_answer_type_unsure_can_recover_without_reapplying_filters(self):
        for stage in ("domain", "query_type"):
            with self.subTest(stage=stage):
                choices = [{"service": "query_state"}, {"area": "area:garden"}]
                if stage == "query_type":
                    choices.append({"domain": "lawn_mower"})
                choices.extend([{stage: "unsure"}, {"query_fallback": "entity:sensor.letztes_mal_gemaht"},
                                {"query_type": "complex"}, {"state_property": "state"}])
                models = ScriptedModels(choices)
                result = await AgentEngine(normalize_settings({}), models).process("Last mowed?", "en", SNAPSHOT)
                self.assertEqual(result["query_fallback"]["trigger"]["stage"], stage)
                self.assertEqual(result["query_fallback"]["status"], "completed")
                self.assertEqual(result["domain"], "sensor")
                self.assertEqual(sum("domain" in request[2] for request in models.requests), 1)
                self.assertIn("query_type:fallback", result["timings_ms"])

    async def test_model_errors_in_normal_query_do_not_trigger_fallback(self):
        models = ScriptedModels([{"service": "query_state"}])
        original = models.decide

        async def decide(config, state, questions):
            if "area" in questions:
                raise ModelError("Model request timed out.")
            return await original(config, state, questions)

        models.decide = decide
        engine = AgentEngine(normalize_settings({}), models)
        with self.assertRaises(ModelError):
            await engine.process("Last mowed?", "en", SNAPSHOT)
        self.assertNotIn("query_fallback", engine.result)

    async def test_yes_no_unsure_rechecks_only_recovered_entities(self):
        models = ScriptedModels([
            {"service": "query_state"}, {"area": "all_areas"}, {"domain": "lawn_mower"},
            {"query_type": "yes_no"}, {"binary_answer": "unsure"},
            {"query_fallback": "entity:light.office"}, {"binary_answer": "yes"},
        ])
        result = await AgentEngine(normalize_settings({}), models).process("Is the office light on?", "en", SNAPSHOT)
        self.assertEqual(result["answer"], "Yes.")
        self.assertEqual(result["query_fallback"]["trigger"]["stage"], "binary_answer")
        self.assertEqual([e["entity_id"] for e in models.requests[-1][1]["entities"]], ["light.office"])
        self.assertEqual(models.requests[-1][1]["domain"], "light")
        self.assertIn("binary_answer", result["decisions"])
        self.assertIn("binary_answer:fallback", result["decisions"])
        self.assertEqual(len(models.requests), 7)

    async def test_downstream_unsure_after_recovery_does_not_start_another_fallback(self):
        models = ScriptedModels([
            {"service": "query_state"}, {"area": "area:garden"}, {"domain": "lawn_mower"},
            {"query_type": "complex"}, {"targets": ["lawn_mower.robot"]}, {"state_property": "none"},
            {"query_fallback": "entity:sensor.letztes_mal_gemaht"}, {"state_property": "none"},
        ])
        result = await AgentEngine(normalize_settings({}), models).process("Last mowed?", "en", SNAPSHOT)
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["query_fallback"]["trigger"]["stage"], "state_property")
        self.assertEqual(result["query_fallback"]["status"], "failed")
        self.assertEqual(result["failures"][0]["stage"], "state_property")
        self.assertEqual(sum("query_fallback" in request[2] for request in models.requests), 1)

    async def test_malformed_fallback_is_an_error_without_retry(self):
        malformed = fallback_answer({"entity:sensor.letztes_mal_gemaht": 1})
        malformed["query_fallback"]["probabilities"].pop("none")
        models = ScriptedModels(WRONG_DOMAIN + [malformed])
        engine = AgentEngine(normalize_settings({}), models)
        with self.assertRaises(ModelError):
            await engine.process("Last mowed?", "en", SNAPSHOT)
        self.assertEqual(engine.result["failures"][0]["stage"], "query_fallback")
        self.assertEqual(len(models.requests), 6)

    async def test_llm_answer_uses_only_recovered_states_without_old_scope_or_scores(self):
        settings = normalize_settings({"answer": {"enable": True}})
        models = ScriptedModels(WRONG_DOMAIN + [{"query_fallback": "entity:sensor.letztes_mal_gemaht"}])
        result = await AgentEngine(settings, models).process("Last mowed?", "en", SNAPSHOT)
        facts = json.loads(models.chat_requests[0][1][-1]["content"])
        self.assertEqual([e["entity_id"] for e in facts["entities"]], ["sensor.letztes_mal_gemaht"])
        for key in ("selected_areas", "all_areas", "query_fallback", "domain_filter", "target_selection"):
            self.assertNotIn(key, facts)
        self.assertEqual(result["query_fallback"]["status"], "completed")


class QueryFallbackSettingsTests(unittest.TestCase):
    def test_defaults_cli_overrides_validation_and_prompt_reset(self):
        defaults = normalize_settings({})
        self.assertEqual(defaults["decider"]["confidence"], 0.5)
        self.assertEqual(defaults["decider"]["probability_threshold"], 0.5)
        args = local.parser().parse_args(["-q", "test", "--confidence", "0.9",
                                         "--threshold", "0.6"])
        defaults["decider"].update(url="http://local/v1/systemone", model="test")
        settings = local.model_settings(defaults, args, {})
        self.assertEqual(settings["decider"]["confidence"], 0.9)
        self.assertEqual(settings["decider"]["probability_threshold"], 0.6)
        for field in ("confidence", "probability_threshold"):
            invalid = copy.deepcopy(defaults)
            invalid["decider"][field] = 1.1
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_settings(invalid)
        self.assertEqual(normalize_settings({"decider": {"query_fallback_prompt": " "}})["decider"]["query_fallback_prompt"],
                         defaults["decider"]["query_fallback_prompt"])
        self.assertEqual(normalize_settings({"decider": {"query_fallback_prompt": "Custom"}})["decider"]["query_fallback_prompt"], "Custom")
