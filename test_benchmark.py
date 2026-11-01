import argparse
import copy
import io
import json
import unittest
from contextlib import asynccontextmanager, redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import benchmark
from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.settings import normalize_settings
from test_integration_engine import SNAPSHOT, ScriptedModels


def settings():
    return normalize_settings({
        "decider": {"url": "http://decider:8321/v1/systemone", "model": "decider"},
        "answer": {"url": "http://answer:8322/v1/chat/completions", "model": "answer"},
    })


class BenchmarkSettingsTests(unittest.TestCase):
    def test_environment_connections_and_specific_thresholds_override_exported_settings(self):
        args = benchmark.parser().parse_args(["--query", "test", "--confidence", "0.3", "--area-confidence", "0.7"])
        exported = settings()
        exported["decider"]["query_target_prompt"] = "Custom rules from HA"
        result = benchmark.model_settings(exported, args, {
            "DECIDER_URL": "http://localhost:8321/v1/systemone", "DECIDER_MODEL": "local-model",
            "DECIDER_API_KEY": "local-key", "ANSWER_API_KEY": "answer-key",
        })
        self.assertEqual(result["decider"]["url"], "http://localhost:8321/v1/systemone")
        self.assertEqual(result["decider"]["model"], "local-model")
        self.assertEqual(result["decider"]["api_key"], "local-key")
        self.assertEqual(result["answer"]["api_key"], "answer-key")
        self.assertEqual(result["decider"]["query_target_prompt"], "Custom rules from HA")
        for field in benchmark.CONFIDENCE_FIELDS:
            self.assertEqual(result["decider"][field], 0.7 if field == "area_confidence" else 0.3)
        self.assertEqual(exported["decider"]["area_confidence"], 0)

    def test_legacy_openrouter_key_is_only_used_for_openrouter_decider(self):
        args = benchmark.parser().parse_args(["--query", "test"])
        local = benchmark.model_settings(settings(), args, {"OPENROUTER_API_KEY": "cloud-key"})
        self.assertEqual(local["decider"]["api_key"], "")
        exported = settings()
        exported["decider"]["url"] = "https://openrouter.ai/api/alpha/decisions"
        cloud = benchmark.model_settings(exported, args, {"OPENROUTER_API_KEY": "cloud-key"})
        self.assertEqual(cloud["decider"]["api_key"], "cloud-key")
        self.assertEqual(cloud["answer"]["api_key"], "")

    def test_previous_domain_filter_confidence_flag_sets_the_new_single_choice_threshold(self):
        args = benchmark.parser().parse_args([
            "--query", "test", "--domain-filter-confidence", "0.2",
            "--query-type-confidence", "0.3", "--binary-answer-confidence", "0.4",
        ])
        result = benchmark.model_settings(settings(), args, {})
        self.assertEqual(result["decider"]["domain_confidence"], 0.2)
        self.assertEqual(result["decider"]["query_type_confidence"], 0.3)
        self.assertEqual(result["decider"]["binary_answer_confidence"], 0.4)

    def test_invalid_confidence_values_are_rejected(self):
        for value in ("-0.1", "1.1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                benchmark.confidence(value)

    def test_parameter_threshold_is_separate_from_global_confidence(self):
        args = benchmark.parser().parse_args(["--query", "test", "--confidence", "0.9", "--parameter-threshold", "0.2"])
        configured = benchmark.model_settings(settings(), args, {})
        self.assertEqual(configured["decider"]["service_confidence"], 0.9)
        self.assertEqual(configured["decider"]["parameter_probability_threshold"], 0.2)
        args = benchmark.parser().parse_args(["--query", "test", "--confidence", "0"])
        self.assertEqual(benchmark.model_settings(settings(), args, {})["decider"]["parameter_probability_threshold"], 0.15)

    def test_action_selection_thresholds_are_independent_of_global_confidence(self):
        args = benchmark.parser().parse_args([
            "--query", "test", "--confidence", "0.9", "--action-area-threshold", "0.1", "--action-entity-threshold", "0.2",
        ])
        configured = benchmark.model_settings(settings(), args, {})["decider"]
        self.assertEqual(configured["service_confidence"], 0.9)
        self.assertEqual(configured["action_area_probability_threshold"], 0.1)
        self.assertEqual(configured["action_entity_probability_threshold"], 0.2)

    def test_text_threshold_is_independent_of_global_confidence_and_old_values(self):
        args = benchmark.parser().parse_args(["-q", "Add milk", "--confidence", "0.9", "--text-threshold", "0.04"])
        exported = settings()
        exported["decider"].update(text_confidence=1, text_keep_uncertain=True)
        decider = benchmark.model_settings(exported, args, {})["decider"]
        self.assertEqual(decider["text_probability_threshold"], 0.04)
        self.assertEqual(decider["service_confidence"], 0.9)
        self.assertNotIn("text_confidence", decider)

    def test_query_target_probability_threshold_is_separate_from_confidence_and_parameter_threshold(self):
        args = benchmark.parser().parse_args(["-q", "test", "--confidence", "0.9", "--query-target-threshold", "0.04"])
        configured = benchmark.model_settings(settings(), args, {})["decider"]
        self.assertEqual(configured["query_target_probability_threshold"], 0.04)
        self.assertEqual(configured["service_confidence"], 0.9)
        self.assertEqual(configured["parameter_probability_threshold"], 0.15)


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def exported(self):
        return {"entry_id": "one", "snapshot": copy.deepcopy(SNAPSHOT), "settings": settings()}

    async def test_benchmark_imports_the_same_engine_and_sends_identical_decider_requests(self):
        self.assertIs(benchmark.AgentEngine, AgentEngine)
        choices = [{"service": "query_state"}, {"area": "all_areas"},
                   {"domain": "light"}, {"query_type": "complex"}, {"targets": ["light.desk"]}]
        local, addon = ScriptedModels(choices), ScriptedModels(choices)
        args = benchmark.parser().parse_args(["--query", "Is the desk lamp on?", "--language", "en"])
        session = AsyncMock()
        with (patch("benchmark.fetch_snapshot", new=AsyncMock(return_value=self.exported())),
              patch("benchmark.aiohttp.ClientSession", return_value=session),
              patch("benchmark.ModelClient", return_value=local)):
            report = await benchmark.run(args, {})

        await AgentEngine(settings(), addon).process(args.query, "en", SNAPSHOT)
        self.assertEqual(local.requests, addon.requests)
        self.assertEqual(set(local.requests[-1][2]), {"query_targets"})
        self.assertEqual(report["mode"], "plan_only")
        self.assertEqual(report["result"]["state_info"][0]["entity_id"], "light.desk")
        self.assertEqual({event["stage"] for event in report["trace"]}, {"service", "area", "domain", "query_type", "target", "answer"})
        self.assertIn("engine", report["timings_ms"])


    async def test_yes_no_benchmark_matches_addon_and_exposes_only_translation_input(self):
        choices = [{"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
                   {"query_type": "yes_no"}, {"binary_answer": "yes"}]
        local, addon = ScriptedModels(choices), ScriptedModels(choices)
        args = benchmark.parser().parse_args(["--query", "Is the office light on?", "--language", "en"])
        with (patch("benchmark.fetch_snapshot", new=AsyncMock(return_value=self.exported())),
              patch("benchmark.aiohttp.ClientSession", return_value=AsyncMock()),
              patch("benchmark.ModelClient", return_value=local)):
            report = await benchmark.run(args, {})

        await AgentEngine(settings(), addon).process(args.query, "en", SNAPSHOT)
        self.assertEqual(local.requests, addon.requests)
        self.assertEqual(local.chat_requests, addon.chat_requests)
        self.assertEqual(report["result"]["state_answer"], "yes")
        self.assertEqual({event["stage"] for event in report["trace"]},
                         {"service", "area", "domain", "query_type", "binary_answer", "answer"})
        self.assertEqual(benchmark.answer_input(report, readable=True)["messages"][1]["content"], {"state_answer": "yes"})


    async def test_text_benchmark_matches_ha_engine_and_prints_extracted_key_value(self):
        from custom_components.glow_assistant.text_values import (
            INSTRUCTIONS,
            text_request,
        )
        from test_integration_engine import entity
        from test_text_values import response_for

        query = "Schreibe Energy Drinks auf die Einkaufsliste"
        args = benchmark.parser().parse_args(["-q", query, "--text-threshold", "0.05"])
        exported = self.exported()
        exported["snapshot"] = {
            "entities": [entity("todo.shopping", "Einkaufsliste", area=None)],
            "services": {"todo.add_item": {"fields": {"item": {"required": True, "selector": {"text": {}}}}}},
        }
        words, _ = text_request(query, "todo.add_item", INSTRUCTIONS)
        probabilities = {key: 0.45 if word["word"] in ("Energy", "Drinks") else 0 for key, word in words.items()}
        probabilities["none"] = 0.1
        choices = [{"service": "todo.add_item"}, {"target_scope": "entities"}, {"action_entities": "entity:todo.shopping"},
                   {"text_words": response_for(probabilities)["answers"]["text_words"]}]
        local, addon = ScriptedModels(choices), ScriptedModels(choices)
        with (patch("benchmark.fetch_snapshot", new=AsyncMock(return_value=exported)),
              patch("benchmark.aiohttp.ClientSession", return_value=AsyncMock()),
              patch("benchmark.ModelClient", return_value=local)):
            report = await benchmark.run(args, {})
        configured = benchmark.model_settings(exported["settings"], args, {})
        await AgentEngine(configured, addon).process(query, "de", exported["snapshot"])
        self.assertEqual(local.requests, addon.requests)
        self.assertEqual(report["result"]["parameters"], {"item": "Energy Drinks"})
        self.assertEqual(report["result"]["parameter_extractions"]["item"]["probability_threshold"], 0.05)
        self.assertEqual(report["result"]["parameter_extractions"]["item"]["real_option_count"], 6)
        self.assertAlmostEqual(report["result"]["parameter_extractions"]["item"]["effective_probability_threshold"], 0.05 / 6)
        self.assertFalse(local.chat_requests)
        self.assertNotIn("execution", report["result"])
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.print_report(report)
        self.assertRegex(output.getvalue(), r"Parameter\s+Value")
        self.assertRegex(output.getvalue(), r"item\s+Energy Drinks")
        self.assertRegex(output.getvalue(), r"text:item\s+\d+\s+item=Energy Drinks")

    async def test_action_benchmark_prints_field_metadata_and_stops_after_selection(self):
        args = benchmark.parser().parse_args(["--query", "Set the office lights to 50 percent"])
        exported = self.exported()
        exported["snapshot"]["services"]["light.turn_on"] = {
            "description": "Turn on a light", "target": {"entity": {"domain": "light"}},
            "fields": {"brightness_pct": {"required": False, "description": "Brightness percentage", "example": 50,
                                          "selector": {"number": {"min": 0, "max": 100, "step": 1}}},
                       "hue": {"selector": {"number": {"min": 0, "max": 360}}}},
        }
        models = ScriptedModels([{"service": "light.turn_on:brightness_pct"}, {"target_scope": "areas"}, {"action_areas": "area:office"}, {"optional_parameters": "brightness_pct"}, {
            "numeric_mode": {"type": "choice", "choice": "absolute", "confidence": 0.9,
                             "probabilities": {"absolute": 1, "relative": 0, "unsure": 0}},
            "numeric_score": {"type": "score", "score": 4, "confidence": 0.9,
                              "probabilities": {str(index): 1 if index == 4 else 0 for index in range(9)}},
        }])
        with (patch("benchmark.fetch_snapshot", new=AsyncMock(return_value=exported)),
              patch("benchmark.aiohttp.ClientSession", return_value=AsyncMock()),
              patch("benchmark.ModelClient", return_value=models)):
            report = await benchmark.run(args, {"DECIDER_API_KEY": "private-key"})
        self.assertEqual(report["result"]["service"], "light.turn_on")
        self.assertEqual(report["result"]["action_metadata"]["fields"],
                         exported["snapshot"]["services"]["light.turn_on"]["fields"])
        self.assertNotIn("execution", report["result"])
        self.assertEqual(len(models.requests), 5)
        self.assertFalse(models.chat_requests)
        self.assertEqual({event["stage"] for event in report["trace"]}, {"service", "target_scope", "action_areas", "parameter_capabilities", "optional_parameters", "numeric:brightness_pct"})
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.print_report(report, show_answer_input=True)
        printed = output.getvalue()
        self.assertRegex(printed, r"target scope\s+Explicit area\(s\); no specific entities")
        self.assertRegex(printed, r"speaker area\s+unknown")
        self.assertNotIn('"selector"', printed)
        self.assertIn("Selected optional parameters: brightness_pct", printed)
        self.assertIn("Probability cutoff: 0.15 / max(1, 2) = 0.075", printed)
        self.assertRegex(printed, r"brightness_pct\s+1\.0000\s+2/2\s+keep")
        self.assertRegex(printed, r"hue\s+0\.0000\s+2/2\s+drop: p < 0\.075")
        self.assertIn("hue", report["result"]["parameter_selection"]["probabilities"])
        self.assertIn("Answer model was not called.", printed)
        self.assertRegex(printed, r"area filter\s+Office")
        self.assertRegex(printed, r"target\s+area; Office \(office\)")
        self.assertNotIn("private-key", printed)
        self.assertNotIn("private-key", json.dumps(report))
        self.assertIsNone(benchmark.answer_input(report))

    async def test_speaker_area_flag_reaches_scope_decision_and_concise_console_output(self):
        args = benchmark.parser().parse_args(["-q", "Mach das Licht aus", "--speaker-area", "Küche"])
        models = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}])
        exported = self.exported()
        exported["snapshot"]["entities"].append({**SNAPSHOT["entities"][0], "entity_id": "light.kitchen",
                                                 "name": "Kitchen light", "area_id": "kitchen", "area_name": "Küche"})
        with (patch("benchmark.fetch_snapshot", new=AsyncMock(return_value=exported)),
              patch("benchmark.aiohttp.ClientSession", return_value=AsyncMock()),
              patch("benchmark.ModelClient", return_value=models)):
            report = await benchmark.run(args, {})
        self.assertEqual(models.requests[-1][1]["speaker_area"], "Küche")
        self.assertEqual(report["result"]["target_scope"], "implied_area")
        self.assertIn("action_metadata", report["result"])
        self.assertEqual(report["result"]["target"]["area_id"], "kitchen")
        self.assertEqual(len(models.requests), 2)
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.print_report(report)
        self.assertRegex(output.getvalue(), r"speaker area\s+Küche")
        self.assertRegex(output.getvalue(), r"target scope\s+Implied area; no specific entities")
        self.assertEqual(report["result"]["parameter_selection"]["selected_parameters"], [])

    async def test_query_answer_input_remains_readable_without_mutating_the_api_messages(self):
        args = benchmark.parser().parse_args(["--query", "Which lights are on?"])
        models = ScriptedModels([{"service": "query_state"}, {"area": "all_areas"}, {"domain": "light"},
                                 {"query_type": "complex"}, {"targets": ["light.desk"]}])
        with (patch("benchmark.fetch_snapshot", new=AsyncMock(return_value=self.exported())),
              patch("benchmark.aiohttp.ClientSession", return_value=AsyncMock()),
              patch("benchmark.ModelClient", return_value=models)):
            report = await benchmark.run(args, {})
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.print_report(report, show_answer_input=True)
        request = json.loads(output.getvalue().split("Answer model input:\n", 1)[1])
        self.assertEqual(request["messages"][0], models.chat_requests[0][1][0])
        self.assertEqual(request["messages"][1]["content"], json.loads(models.chat_requests[0][1][1]["content"]))
        self.assertEqual(request["messages"][1]["content"]["entities"][0]["entity_id"], "light.desk")
        self.assertIsInstance(models.chat_requests[0][1][1]["content"], str)
        self.assertEqual(benchmark.answer_input(report)["messages"], models.chat_requests[0][1])
        self.assertEqual(set(request), {"model", "messages", "max_tokens", "temperature"})

    async def test_model_failure_keeps_stage_diagnostics(self):
        args = benchmark.parser().parse_args(["--query", "test"])
        client = SimpleNamespace(decide=AsyncMock(side_effect=ModelError("Model request timed out.")))
        with (patch("benchmark.fetch_snapshot", new=AsyncMock(return_value=self.exported())),
              patch("benchmark.aiohttp.ClientSession", return_value=AsyncMock()),
              patch("benchmark.ModelClient", return_value=client)):
            report = await benchmark.run(args, {})
        self.assertEqual(report["result"]["error"], "Model request timed out.")
        self.assertTrue(any(event["stage"] == "failure" for event in report["trace"]))
        self.assertIsNone(benchmark.answer_input(report))

    async def test_snapshot_request_uses_administrator_token_without_redirects(self):
        response = SimpleNamespace(status=200, json=AsyncMock(return_value=self.exported()))

        @asynccontextmanager
        async def request(*args, **kwargs):
            yield response

        session = SimpleNamespace(get=Mock(side_effect=request))
        exported = await benchmark.fetch_snapshot(session, {"HA_URL": "http://ha:8123/", "HA_TOKEN": "ha-secret", "GLOW_ENTRY_ID": "one"})
        self.assertEqual(exported["entry_id"], "one")
        self.assertEqual(session.get.call_args.args, ("http://ha:8123/api/glow_assistant/snapshot",))
        self.assertEqual(session.get.call_args.kwargs["headers"], {"Authorization": "Bearer ha-secret"})
        self.assertEqual(session.get.call_args.kwargs["params"], {"entry_id": "one"})
        self.assertFalse(session.get.call_args.kwargs["allow_redirects"])

    async def test_snapshot_errors_never_include_token_or_echo_server_response(self):
        for status in (302, 401, 403, 404, 500):
            @asynccontextmanager
            async def request(*args, **kwargs):
                yield SimpleNamespace(status=status)

            with self.subTest(status=status), self.assertRaises(ModelError) as error:
                await benchmark.fetch_snapshot(SimpleNamespace(get=request), {"HA_URL": "http://ha:8123", "HA_TOKEN": "private-token"})
            self.assertNotIn("private-token", str(error.exception))


if __name__ == "__main__":
    unittest.main()
