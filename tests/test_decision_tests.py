import copy
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.settings import normalize_settings
from scripts import decision_tests as runner
from tests.test_integration_engine import SNAPSHOT, ScriptedModels, entity


class DecisionScoringTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = json.loads((runner.DATA / "snapshot.json").read_text())
        self.cases = runner.load_cases(runner.DATA / "cases.jsonl", self.snapshot)

    def test_text_distance_normalization_and_missing_values(self):
        self.assertEqual(runner.levenshtein("kitten", "sitting"), 3)
        self.assertEqual(runner.value_score("Rote Zwiebeln", "  rote   ZWIEBELN ")["score"], 1)
        self.assertAlmostEqual(runner.value_score("kitten", "sitting")["score"], 4 / 7)
        for actual in (None, "", 42):
            self.assertEqual(runner.value_score("onions", actual)["score"], 0)

    def test_number_score_keeps_absolute_error_and_clamps(self):
        score = runner.value_score(22, 20, scale=10)
        self.assertEqual(score["absolute_error"], 2)
        self.assertEqual(score["score"], 0.8)
        for actual in (None, "22", float("nan"), 200):
            self.assertEqual(runner.value_score(22, actual, 10)["score"], 0)
        with self.assertRaises(ValueError):
            runner.value_score(22, 22, 0)

    def test_bad_extraction_scores_do_not_fail_good_decisions(self):
        case = next(case for case in self.cases if case["id"] == "todo.add.en")
        result = {"route": "smarthome", "service": "todo.add_item", "domain": "todo", "target_scope": "entities",
                  "target": {"type": "entity", "entity_id": "todo.einkaufsliste"}, "required_parameters": ["item"],
                  "parameters": {"item": "completely wrong text"}}
        evaluated = runner.evaluate(case, result, self.snapshot)
        self.assertTrue(evaluated["passed"])
        self.assertLess(evaluated["scores"]["item"]["score"], 0.5)
        result.update(route="unsure", failures=[{"stage": "text:item"}])
        result.pop("parameters")
        evaluated = runner.evaluate(case, result, self.snapshot)
        self.assertTrue(evaluated["passed"])
        self.assertTrue(evaluated["actual"]["extraction_only_stop"])
        self.assertEqual(evaluated["scores"]["item"]["score"], 0)
        self.assertEqual(result["route"], "unsure")

    def test_wrong_targets_enum_or_technical_error_fail(self):
        case = {"expect": {"route": "smarthome", "entity_ids": ["todo.einkaufsliste"],
                           "select_values": {"status": ["needs_action"]}}}
        result = {"route": "smarthome", "target": {"entity_ids": ["todo.aufgaben"]},
                  "parameter_extractions": {"status": {"type": "select", "value": ["completed"]}}}
        evaluated = runner.evaluate(case, result, self.snapshot)
        self.assertFalse(evaluated["passed"])
        self.assertFalse(evaluated["checks"]["entity_ids"]["passed"])
        self.assertFalse(evaluated["checks"]["select_values"]["passed"])
        result["target"]["entity_ids"] = ["todo.einkaufsliste"]
        result["parameter_extractions"]["status"]["value"] = ["needs_action"]
        self.assertTrue(runner.evaluate(case, result, self.snapshot)["passed"])
        self.assertFalse(runner.evaluate(case, result, self.snapshot, error="HTTP 503")["passed"])

    def test_area_expansion_and_alternative_parameter_encodings(self):
        case = next(case for case in self.cases if case["id"] == "lights.brighter.en")
        result = {"route": "smarthome", "service": "light.turn_on", "domain": "light",
                  "target_scope": "entities_explicit_area", "selected_areas": [{"area_id": "office"}],
                  "target": {"entity_id": "light.buro_lichtschalter"},
                  "parameter_selection": {"selected_parameters": ["brightness_step_pct"]},
                  "parameters": {"brightness_step_pct": 8}}
        evaluated = runner.evaluate(case, result, self.snapshot)
        self.assertTrue(evaluated["passed"])
        self.assertEqual(evaluated["scores"]["brightness_step_pct"]["score"], 0.98)
        result.update(domain="light", target={"type": "area", "area_id": "office"})
        self.assertEqual(runner.actual_decisions(result, self.snapshot)["entity_ids"],
                         ["light.buro_lichtschalter", "light.buro_schreibtisch"])

    def test_missing_numeric_value_scores_zero_without_failing_decisions(self):
        case = {"expect": {"route": "smarthome", "score_values": {"temperature": 22}}}
        result = {"route": "unsure", "service": "climate.set_temperature", "failures": [{"stage": "numeric:temperature"}]}
        evaluated = runner.evaluate(case, result, self.snapshot)
        self.assertTrue(evaluated["passed"])
        self.assertEqual(evaluated["scores"]["temperature"]["score"], 0)
        self.assertEqual(evaluated["scores"]["temperature"]["scale"], 10)
        result["failures"].append({"stage": "service"})
        self.assertFalse(runner.evaluate(case, result, self.snapshot)["passed"])

    def test_corpus_has_paired_languages_and_distinct_translations(self):
        self.assertEqual({case["language"] for case in self.cases}, {"de", "en"})
        cases = {case["id"]: case for case in self.cases}
        for case in self.cases:
            other = case["id"].rsplit(".", 1)[0] + (".en" if case["language"] == "de" else ".de")
            self.assertIn(other, cases)
        self.assertEqual(cases["todo.add.de"]["expect"]["score_values"]["item"], "rote Zwiebeln")
        self.assertEqual(cases["todo.add.en"]["expect"]["score_values"]["item"], "red onions")
        self.assertTrue(any(case.get("speaker_area") for case in self.cases))
        self.assertTrue(any(case.get("history") for case in self.cases))

    def test_disabled_references_are_skipped_but_invalid_active_json_is_rejected(self):
        case = self.cases[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text('# Disabled reference\n  # ' + json.dumps(case) + '\n\n' + json.dumps(case) + '\n')
            self.assertEqual(runner.load_cases(path, self.snapshot), [case])
            path.write_text(path.read_text() + 'invalid active JSON\n')
            with self.assertRaises(json.JSONDecodeError):
                runner.load_cases(path, self.snapshot)

    def test_active_queries_are_not_verbatim_default_prompt_examples(self):
        def normalize(value):
            return " " + " ".join(re.findall(r"\w+|%", value.casefold())) + " "

        prompts = {key: normalize(value) for key, value in normalize_settings({})["decider"].items() if key.endswith("_prompt")}
        for case in self.cases:
            for name, prompt in prompts.items():
                with self.subTest(case=case["id"], prompt=name):
                    self.assertNotIn(normalize(case["query"]), prompt)


class DecisionOnlyEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_action_keeps_all_decisions_but_never_executes_or_formats_reply(self):
        client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}])
        config = normalize_settings({"answer": {"enable": True}})
        execute = AsyncMock()
        with patch("custom_components.glow_assistant.engine.format_action_response", side_effect=AssertionError("formatted reply")):
            result = await AgentEngine(config, client, execute=execute, decisions_only=True).process(
                "Turn off the lights", "en", SNAPSHOT, speaker_area="office")
        self.assertEqual(result["route"], "smarthome")
        self.assertEqual(result["target"]["area_id"], "office")
        self.assertNotIn("answer", result)
        self.assertNotIn("execution", result)
        self.assertFalse(client.chat_requests)
        execute.assert_not_awaited()

    async def test_complex_query_selects_properties_even_with_answer_model_enabled(self):
        client = ScriptedModels([{"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
                                 {"query_type": "complex"}, {"targets": ["light.desk"]}, {"state_property": "state"}])
        with patch("custom_components.glow_assistant.engine.format_values", side_effect=AssertionError("formatted reply")):
            result = await AgentEngine(normalize_settings({"answer": {"enable": True}}), client, decisions_only=True).process(
                "Which office lights are on?", "en", SNAPSHOT)
        self.assertEqual(result["state_properties"], ["state"])
        self.assertEqual(result["state_values"][0]["entity_id"], "light.desk")
        self.assertNotIn("answer", result)
        self.assertFalse(client.chat_requests)

    async def test_binary_query_keeps_yes_no_decision_without_reply(self):
        client = ScriptedModels([{"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
                                 {"query_type": "yes_no"}, {"binary_answer": "yes"}])
        result = await AgentEngine(normalize_settings({}), client, decisions_only=True).process("Is the office light on?", "en", SNAPSHOT)
        self.assertEqual(result["state_answer"], "yes")
        self.assertNotIn("answer", result)
        self.assertFalse(client.chat_requests)

    async def test_general_is_classified_without_chat_or_mcp_regardless_of_enable_setting(self):
        for enabled in (True, False):
            client = ScriptedModels([{"service": "not-smarthome-related"}])
            config = normalize_settings({"general": {"enable": enabled}})
            config["general"]["mcp_servers"] = [{"enable": True}]
            with patch("custom_components.glow_assistant.engine.general_answer", new_callable=AsyncMock) as general:
                result = await AgentEngine(config, client, decisions_only=True).process("Tell me a joke", "en", SNAPSHOT)
            self.assertEqual(result["route"], "not-smarthome-related")
            self.assertNotIn("answer", result)
            self.assertFalse(client.chat_requests)
            general.assert_not_awaited()

    async def test_uncertain_and_api_errors_never_leave_answer_text(self):
        client = ScriptedModels([{"service": "light.turn_off"}], confidence=0)
        result = await AgentEngine(normalize_settings({}), client, decisions_only=True).process("test", "en", SNAPSHOT)
        self.assertEqual(result["route"], "unsure")
        self.assertNotIn("answer", result)
        client.decide = AsyncMock(side_effect=ModelError("HTTP 503"))
        engine = AgentEngine(normalize_settings({}), client, decisions_only=True)
        with self.assertRaises(ModelError):
            await engine.process("test", "en", SNAPSHOT)
        self.assertNotIn("answer", engine.result)

    async def test_run_case_uses_shared_text_extraction_and_passes_history(self):
        case = {"id": "test", "category": "todos", "query": "Add onions", "language": "en",
                "history": [{"role": "user", "content": "Use my shopping list"}],
                "expect": {"route": "smarthome", "parameters": ["item"], "score_values": {"item": "onions"}}}
        snapshot = {"entities": [entity("todo.shopping", "Shopping list", area=None)], "services": {
            "todo.add_item": {"fields": {"item": {"required": True, "selector": {"text": {}}}}}}}
        client = ScriptedModels([{"service": "todo.add_item"}, {"target_scope": "entities"},
                                 {"action_entities": "entity:todo.shopping"}, {"text_words": "word:1:onions"}])
        with patch("scripts.decision_tests.ModelClient", return_value=client):
            result = await runner.run_case(case, snapshot, normalize_settings({}), None)
        self.assertTrue(result["passed"])
        self.assertEqual(result["scores"]["item"]["score"], 1)
        self.assertNotIn("answer", result["result"])
        self.assertTrue(result["trace"])
        for request in client.requests:
            self.assertEqual(request[1]["history"], case["history"])
        self.assertFalse(client.chat_requests)


class DecisionRunnerCLITests(unittest.TestCase):
    def test_list_needs_no_connection_or_credentials(self):
        output = io.StringIO()
        with (redirect_stdout(output), patch("scripts.decision_tests.ModelClient", side_effect=AssertionError("model call")),
              patch("scripts.decision_tests.load_dotenv", side_effect=AssertionError("loaded secrets"))):
            self.assertEqual(runner.main(["--list", "--category", "shutters", "--limit", "2"]), 0)
        self.assertIn("shutters.percent.de", output.getvalue())
        self.assertNotIn("lights.room_on", output.getvalue())

    def test_json_report_redacts_secrets_and_exit_status_tracks_decisions(self):
        result = {"id": "test", "passed": True, "error": None, "scores": {}, "duration_ms": 12,
                  "model_calls": [{"response": "secret-decider-key", "request": {"state": "test"}}]}
        with tempfile.TemporaryDirectory() as directory:
            for passed in (True, False):
                result["passed"] = passed
                path = Path(directory) / f"{passed}.json"
                with (redirect_stdout(io.StringIO()), patch("scripts.decision_tests.load_dotenv"),
                      patch.dict("os.environ", {"DECIDER_URL": "http://decider:8321/v1/systemone", "DECIDER_MODEL": "test-model",
                                                "DECIDER_API_KEY": "secret-decider-key"}, clear=True),
                      patch("scripts.decision_tests.run", new=AsyncMock(return_value=[copy.deepcopy(result)]))):
                    code = runner.main(["--case", "general.knowledge.en", "--confidence", "0", "--query-target-threshold", "0.15", "--output", str(path)])
                self.assertEqual(code, 0 if passed else 1)
                report = json.loads(path.read_text())
                self.assertEqual(report["model"], "test-model")
                self.assertEqual(report["summary"]["passed"], int(passed))
                self.assertNotIn("secret-decider-key", path.read_text())
                self.assertEqual(len(report["cases_sha256"]), 64)
                self.assertFalse(report["settings"]["answer"]["enable"])
                self.assertEqual(report["settings"]["decider"]["query_target_probability_threshold"], 0.15)
                self.assertEqual(report["settings"]["decider"]["service_confidence"], 0)
                self.assertEqual(report["settings"]["decider"]["action_entity_probability_threshold"], 0.8)
