import copy
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock

from custom_components.glow_assistant.action_targets import (
    decode_selection,
    noul_request,
    selection_question,
)
from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.debug import decision_details
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.optional_parameters import decode_parameters
from custom_components.glow_assistant.select_values import decode_select
from custom_components.glow_assistant.settings import (
    normalize_settings,
    validate_settings,
)
from custom_components.glow_assistant.text_values import decode_text, text_request
from scripts import local
from tests.test_integration_engine import SNAPSHOT, ScriptedModels, entity


def noul_answers(name, values):
    return {"answers": {f"{name}:{key}": {"type": "noul", "noul": value} for key, value in values.items()}}


class NoulSelectionTests(unittest.TestCase):
    def test_setting_defaults_and_invalid_values(self):
        settings = normalize_settings({"decider": {"url": "http://localhost:8321/v1/systemone"}})
        self.assertFalse(settings["decider"]["use_noul"])
        self.assertEqual(settings["decider"]["confidence"], 0.5)
        for value in (0, 1, "true", None):
            invalid = copy.deepcopy(settings)
            invalid["decider"]["use_noul"] = value
            with self.subTest(value=value), self.assertRaises(TypeError):
                validate_settings(invalid)
        for value in (-0.1, 1.1, True, float("nan")):
            invalid = copy.deepcopy(settings)
            invalid["decider"]["confidence"] = value
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                validate_settings(invalid)

    def test_only_one_confidence_setting_remains_and_relative_thresholds_are_preserved(self):
        removed = {name + "_confidence": 0.99 for name in (
            "service", "domain", "query_type", "binary_answer", "query_fallback", "target_scope", "action_area", "numeric", "select", "noul",
        )}
        removed.update(dict.fromkeys(("area_confidence", "state_property_confidence", "query_fallback_probability_threshold",
                                     "action_area_probability_threshold", "action_entity_probability_threshold",
                                     "parameter_probability_threshold", "select_probability_threshold"), 0.99))
        config = normalize_settings({"decider": {**removed, "probability_threshold": 0.2, "query_target_probability_threshold": 0.3,
                                                "text_probability_threshold": 0.4}})["decider"]
        self.assertTrue(set(removed).isdisjoint(config))
        self.assertEqual(config["confidence"], 0.5)
        self.assertEqual(config["probability_threshold"], 0.2)
        self.assertEqual(config["query_target_probability_threshold"], 0.3)
        self.assertEqual(config["text_probability_threshold"], 0.4)

    def test_exactly_four_decision_numbers_and_their_defaults(self):
        settings = normalize_settings({"decider": {"url": "http://localhost:8321/v1/systemone"}})
        defaults = {key: value for key, value in settings["decider"].items()
                    if key == "confidence" or key.endswith("probability_threshold")}
        self.assertEqual(defaults, {"confidence": 0.5, "probability_threshold": 0.5,
                                    "query_target_probability_threshold": 0.15, "text_probability_threshold": 0.4})
        for key in defaults:
            for invalid_value in (-0.1, 1.1, True, float("nan")):
                invalid = copy.deepcopy(settings)
                invalid["decider"][key] = invalid_value
                with self.subTest(key=key, value=invalid_value), self.assertRaises(ValueError):
                    validate_settings(invalid)

    def test_requests_batch_independent_candidates_without_fallback_options(self):
        options = {"entity:light.desk": {"name": "Desk lamp"}, "entity:light.ceiling": {"name": "Ceiling lamp"}}
        question = selection_question(options, "Select the requested lights.")
        original = copy.deepcopy(question)
        state, questions = noul_request({"prompt": "Turn both lights on."}, "action_entities", question)
        self.assertEqual(state["candidates"], options)
        self.assertEqual(state["selection_rules"], original["instructions"])
        self.assertEqual(set(questions), {f"action_entities:{key}" for key in options})
        self.assertTrue(all(question["type"] == "noul" and "criteria" not in question for question in questions.values()))
        self.assertEqual(question, original)

    def test_yes_probability_uses_an_absolute_inclusive_confidence_cutoff(self):
        for count in (2, 20):
            probabilities = {"yes": 0.8, "no": 0.79, **{f"irrelevant:{index}": 0.01 for index in range(count)}}
            options = {key: {} for key in probabilities}
            for relative_threshold in (0, 0.1, 1):
                selection = decode_selection(noul_answers("query_targets", probabilities), "query_targets", options,
                                             relative_threshold, noul_confidence=0.8)
                self.assertEqual(selection["selected_choices"], ["yes"])
                self.assertEqual(selection["minimum_confidence"], 0.8)
                self.assertEqual(selection["selection_mode"], "independent_noul")
                self.assertIsNone(selection["effective_probability_threshold"])

    def test_all_candidates_can_match_or_none_can_match(self):
        for probabilities, expected in (({"a": 0.9, "b": 0.8}, ["a", "b"]), ({"a": 0.1, "b": 0.2}, [])):
            selection = decode_selection(noul_answers("area", probabilities), "area", probabilities, noul_confidence=0.5)
            self.assertEqual(selection["selected_choices"], expected)

    def test_incomplete_extra_and_malformed_answers_are_rejected(self):
        response = noul_answers("query_targets", {"a": 0.9, "b": 0.1})
        invalid = [{"answers": {}}, noul_answers("query_targets", {"a": 0.9}),
                   noul_answers("query_targets", {"a": 0.9, "b": 0.1, "extra": 0.8})]
        for value in (True, "0.9", None, -0.1, 1.1, float("nan"), float("inf")):
            item = copy.deepcopy(response)
            item["answers"]["query_targets:a"]["noul"] = value
            invalid.append(item)
        item = copy.deepcopy(response)
        item["answers"]["query_targets:a"]["type"] = "choice"
        invalid.append(item)
        for item in invalid:
            with self.subTest(response=item), self.assertRaises(ModelError):
                decode_selection(item, "query_targets", {"a": {}, "b": {}}, noul_confidence=0.5)

    def test_parameter_text_and_multi_select_extractors_use_noul_results(self):
        result = decode_parameters(noul_answers("optional_parameters", {"brightness": 0.8, "hue": 0.2}),
                                   {"brightness": {}, "hue": {}}, 0.01, noul_confidence=0.7)
        self.assertEqual(result["selected_parameters"], ["brightness"])
        words, _ = text_request("add red onions please", "todo.add_item", "Extract the item.")
        values = {key: 0.9 if value["word"] in ("red", "onions") else 0.1 for key, value in words.items()}
        result = decode_text(noul_answers("text_words", values), words, 0.01, noul_confidence=0.7)
        self.assertEqual(result["value"], "red onions")
        options = {"value:open": {"value": "open"}, "value:closed": {"value": "closed"}}
        result = decode_select(noul_answers("select_values", {"value:open": 0.9, "value:closed": 0.1}),
                               options, True, 0.01, 0.99, noul_confidence=0.7)
        self.assertEqual(result["value"], ["open"])

    def test_debug_reports_the_actual_noul_values_and_confidence_cutoff(self):
        _, questions = noul_request({}, "action_entities", selection_question({"desk": {}, "ceiling": {}}, "Private task"))
        response = noul_answers("action_entities", {"desk": 0.8, "ceiling": 0.2})
        details = decision_details(questions, response["answers"], {"confidence": 0.7})
        self.assertTrue(details["action_entities:desk"]["kept"])
        self.assertFalse(details["action_entities:ceiling"]["kept"])
        self.assertEqual(details["action_entities:desk"]["probability"], 0.8)
        self.assertEqual(details["action_entities:desk"]["minimum_confidence"], 0.7)
        self.assertNotIn("Private task", str(details))


class NoulModels(ScriptedModels):
    async def decide(self, config, state, questions):
        if next(iter(questions.values()))["type"] != "noul":
            return await super().decide(config, state, questions)
        self.requests.append(copy.deepcopy((config, state, questions)))
        batch = self.choices.pop(0)
        if batch is None:
            return {"answers": {}}
        answers = {}
        for name in questions:
            stage, candidate = name.split(":", 1)
            answers[name] = {"type": "noul", "noul": batch[stage].get(candidate, 0.1)}
        return {"answers": answers, "usage": {"input_tokens": 7}}


class NoulEngineTests(unittest.IsolatedAsyncioTestCase):
    async def run_engine(self, choices, snapshot=SNAPSHOT, settings=None, history=None, speaker_area=None):
        config = normalize_settings({"history_enabled": bool(history), "decider": {
            "use_noul": True, "confidence": 0.7, **(settings or {}),
        }})
        models, events, execute = NoulModels(choices), [], AsyncMock()
        engine = AgentEngine(config, models, history=history, decisions_only=True, execute=execute,
                             trace=lambda stage, arguments, output=None: events.append((stage, arguments, output)))
        result = await engine.process("test request", "en", snapshot, speaker_area=speaker_area)
        execute.assert_not_awaited()
        self.assertFalse(models.chat_requests)
        return result, models, events

    async def test_query_uses_noul_for_areas_entities_and_properties_with_history(self):
        data = copy.deepcopy(SNAPSHOT)
        data["entities"][0]["attributes"]["brightness"] = 128
        history = [{"role": "user", "content": "The office and bedroom."}, {"role": "assistant", "content": "Okay"}]
        result, models, events = await self.run_engine([
            {"service": "query_state"}, {"area": {"area:office": 0.9, "area:bedroom": 0.8, "all_areas": 0.95}},
            {"domain": "light"}, {"query_type": "complex"},
            {"query_targets": {"entity:light.desk": 0.8, "entity:light.bed": 0.7}},
            {"state_property": {"state": 0.9, "attribute:brightness": 0.8}},
        ], snapshot=data, history=history, settings={"query_target_probability_threshold": 0, "query_target_prompt": "PRIVATE RULES"})
        self.assertEqual(result["route"], "smarthome")
        self.assertEqual(result["target"]["entity_ids"], ["light.desk", "light.bed"])
        self.assertEqual(result["state_properties"], ["state", "attribute:brightness"])
        self.assertFalse(result["all_areas"])
        self.assertEqual(result["area_selection"]["selected_area_ids"], ["office", "bedroom"])
        self.assertEqual(len(models.requests), 6)
        for _, state, _ in models.requests:
            self.assertEqual(state["history"], history)
            self.assertEqual(state["language"], "en")
        self.assertEqual([next(iter(questions.values()))["type"] for _, _, questions in models.requests],
                         ["choice", "noul", "choice", "choice", "noul", "noul"])
        self.assertNotIn("PRIVATE RULES", str(events))
        target = next(output for stage, _, output in events if stage == "target" and output is not None)
        self.assertEqual(target["minimum_confidence"], 0.7)
        self.assertEqual(target["decisions"]["query_targets:entity:light.desk"]["probability"], 0.8)
        for _, _, output in events:
            for decision in (output or {}).get("decisions", {}).values():
                self.assertEqual(decision["minimum_confidence"], 0.7)

    async def test_noul_and_single_choices_share_the_same_configured_confidence(self):
        result, models, events = await self.run_engine([{"service": "query_state"}], settings={"confidence": 0.95})
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(len(models.requests), 1)
        output = next(output for stage, _, output in events if stage == "service" and output is not None)
        self.assertEqual(output["decisions"]["service"]["minimum_confidence"], 0.95)

    async def test_action_target_paths_keep_single_area_choices_and_speaker_shortcuts(self):
        paths = [
            ("areas", [{"action_areas": {"area:office": 0.8, "area:bedroom": 0.9}}], None, ["choice", "choice", "noul"]),
            ("entities_explicit_area", [{"action_areas": "area:office"}, {"action_entities": {"entity:light.desk": 0.8}}],
             None, ["choice", "choice", "choice", "noul"]),
            ("entities_implied_area", [{"action_entities": {"entity:light.desk": 0.8}}], "office", ["choice", "choice", "noul"]),
            ("implied_area", [], "office", ["choice", "choice"]),
        ]
        for scope, steps, speaker, types in paths:
            with self.subTest(scope=scope):
                result, models, _ = await self.run_engine([{"service": "light.turn_off"}, {"target_scope": scope}, *steps], speaker_area=speaker)
                self.assertEqual(result["route"], "smarthome")
                self.assertEqual([next(iter(questions.values()))["type"] for _, _, questions in models.requests], types)

    async def test_optional_fields_and_text_words_use_shared_confidence(self):
        data = {"entities": [entity("todo.shopping", "Shopping list", area=None), entity("todo.other", "Other list", area=None)],
                "services": {"todo.add_item": {"target": {}, "fields": {
                    "item": {"required": True, "selector": {"text": {}}}, "description": {"selector": {"text": {}}},
                }}}}
        words = {"word:0:test": 0.8, "word:1:request": 0.1}
        result, models, _ = await self.run_engine([
            {"service": "todo.add_item"}, {"target_scope": "entities"},
            {"action_entities": {"entity:todo.shopping": 0.8}}, {"optional_parameters": {"description": 0.7}},
            {"text_words": words}, {"text_words": {"word:1:request": 0.9}},
        ], snapshot=data, settings={"text_probability_threshold": 0, "probability_threshold": 1})
        self.assertEqual(result["route"], "smarthome")
        self.assertEqual(result["parameters"], {"item": "test", "description": "request"})
        self.assertEqual(result["parameter_selection"]["selected_parameters"], ["description"])
        self.assertTrue(all(question["type"] == "noul" for _, _, questions in models.requests[2:] for question in questions.values()))

    async def test_local_output_shows_noul_confidence_and_selection_mode(self):
        data = {"entities": [entity("todo.shopping", "Shopping list", area=None)], "services": {
            "todo.get_items": {"target": {}, "fields": {"status": {"selector": {
                "select": {"multiple": True, "options": ["needs_action", "completed"]},
            }}}}}}
        result, _, events = await self.run_engine([
            {"service": "todo.get_items"}, {"target_scope": "entities"},
            {"action_entities": {"entity:todo.shopping": 0.8}},
            {"optional_parameters": {"status": 0.9}}, {"select_values": {"value:needs_action": 0.9}},
        ], snapshot=data)
        report = {"result": result, "context": {"entities": 1, "actions": 1},
                  "timings_ms": {"snapshot": 1, "engine": 5, "total": 6},
                  "trace": [{"stage": stage, "output": output} for stage, _, output in events if output is not None]}
        output = io.StringIO()
        with redirect_stdout(output):
            local.print_report(report)
        self.assertIn("Noul confidence: keep p(yes) >= 0.7", output.getvalue())
        self.assertNotIn("Probability cutoff:", output.getvalue())

    async def test_yes_no_shortcut_only_changes_area_multi_selection(self):
        result, models, _ = await self.run_engine([
            {"service": "query_state"}, {"area": {"area:office": 0.9}},
            {"domain": "light"}, {"query_type": "yes_no"}, {"binary_answer": "yes"},
        ])
        self.assertEqual(result["state_answer"], "yes")
        self.assertNotIn("target_selection", result)
        self.assertEqual([next(iter(questions.values()))["type"] for _, _, questions in models.requests],
                         ["choice", "noul", "choice", "choice", "choice"])

    async def test_select_values_use_noul_only_for_fields_allowing_multiple_values(self):
        for multiple in (True, False):
            data = {"entities": [entity("todo.shopping", "Shopping list", area=None)], "services": {
                "todo.get_items": {"target": {}, "fields": {"status": {"required": True, "selector": {
                    "select": {"multiple": multiple, "options": ["needs_action", "completed"]},
                }}}}}}
            choice = {"select_values": {"value:needs_action": 0.8, "value:completed": 0.9}} if multiple else {"select": "value:completed"}
            result, models, _ = await self.run_engine([
                {"service": "todo.get_items"}, {"target_scope": "entities"},
                {"action_entities": {"entity:todo.shopping": 0.9}}, choice,
            ], snapshot=data)
            self.assertEqual(result["route"], "smarthome")
            self.assertEqual(result["parameters"]["status"], ["needs_action", "completed"] if multiple else "completed")
            self.assertEqual(next(iter(models.requests[-1][2].values()))["type"], "noul" if multiple else "choice")

    async def test_fallback_uses_shared_noul_confidence_without_choice_gates(self):
        data = copy.deepcopy(SNAPSHOT)
        data["entities"].append(entity("sensor.last_mowed", "Last mowed", area=None, state="2026-11-02T10:00:00Z"))
        result, models, _ = await self.run_engine([
            {"service": "query_state"}, {"area": {"area:office": 0.9}}, {"domain": "unsure"},
            {"query_fallback": {"entity:sensor.last_mowed": 0.8}}, {"query_type": "complex"}, {"state_property": {"state": 0.9}},
        ], snapshot=data, settings={"probability_threshold": 1})
        self.assertEqual(result["route"], "smarthome")
        self.assertEqual(result["query_fallback"]["status"], "completed")
        self.assertEqual(result["target"]["entity_id"], "sensor.last_mowed")
        self.assertEqual(result["query_fallback"]["minimum_confidence"], 0.7)
        self.assertEqual(len(models.requests), 6)

    async def test_empty_or_malformed_noul_targets_stop_without_execution(self):
        for response in ({"action_entities": {}}, None):
            with self.subTest(response=response):
                if response is None:
                    with self.assertRaisesRegex(ModelError, "Malformed"):
                        await self.run_engine([
                            {"service": "light.turn_off"}, {"target_scope": "entities_implied_area"}, response,
                        ], speaker_area="office")
                    continue
                result, models, _ = await self.run_engine([
                    {"service": "light.turn_off"}, {"target_scope": "entities_implied_area"}, response,
                ], speaker_area="office")
                self.assertEqual(len(models.requests), 3)
                self.assertIsNone(result["target"])
                self.assertEqual(result["route"], "unsure")

    async def test_numeric_mode_and_score_are_not_converted_to_noul(self):
        data = {"entities": [entity("media_player.speaker", "Speaker", volume_level=0.2)], "services": {
            "media_player.volume_set": {"target": {}, "fields": {"volume_level": {"required": True, "selector": {
                "number": {"min": 0, "max": 1, "step": 0.01},
            }}}}}}
        result, models, _ = await self.run_engine([
            {"service": "media_player.volume_set"}, {"target_scope": "entities"},
            {"action_entities": {"entity:media_player.speaker": 0.8}},
            {"numeric_mode": "absolute", "numeric_score": {"type": "score", "score": 4, "confidence": 0.9,
                "probabilities": {str(index): float(index == 4) for index in range(9)}}},
        ], snapshot=data)
        self.assertEqual(result["route"], "smarthome")
        self.assertEqual(result["parameters"]["volume_level"], 0.5)
        self.assertEqual({name: question["type"] for name, question in models.requests[-1][2].items()},
                         {"numeric_mode": "choice", "numeric_score": "score"})
