import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.settings import normalize_settings
from scripts import local
from tests.test_integration_engine import ScriptedModels, entity


def snapshot():
    entities = [
        entity("light.kitchen", "Deckenlampe", area="kitchen"),
        entity("light.office", "Deckenlampe", area="office"),
        entity("light.desk", "Desk lamp", area="office", state="off"),
        entity("light.bedroom", "Bedroom lamp", area="bedroom"),
        entity("light.unassigned", "Unassigned lamp", area=None),
        entity("sensor.kitchen", "Kitchen temperature", area="kitchen", state="22"),
        entity("sensor.outside", "Outside temperature", area="outside", state="10"),
    ]
    for item in entities:
        if item["area_id"] in ("kitchen", "office"):
            item["area_name"] = "Küche" if item["area_id"] == "kitchen" else "Büro"
            item["area_aliases"] = ["kueche"] if item["area_id"] == "kitchen" else ["buero"]
    return {"entities": entities, "services": {}}


def area_choice(probabilities):
    scores = dict.fromkeys(("area:kitchen", "area:office", "area:bedroom", "area:outside", "all_areas", "none"), 0)
    scores.update(probabilities)
    return {"area": {"type": "choice", "choice": max(scores, key=scores.get), "confidence": 0,
                     "probabilities": scores}}


class QueryAreaTests(unittest.IsolatedAsyncioTestCase):
    async def test_speaker_context_reaches_query_area_selection_without_overriding_explicit_scope(self):
        data = {"entities": [
            entity("binary_sensor.kitchen", "Window", area="kitchen", state="off", device_class="window"),
            entity("binary_sensor.office", "Window", area="office", state="on", device_class="window"),
        ], "services": {}}
        scenarios = [
            ("de", "Sind hier Fenster offen?", "area:kitchen", "no", {"binary_sensor.kitchen"}),
            ("en", "Are any windows open in here?", "area:kitchen", "no", {"binary_sensor.kitchen"}),
            ("de", "Sind im Büro Fenster offen?", "area:office", "yes", {"binary_sensor.office"}),
            ("en", "Are any office windows open?", "area:office", "yes", {"binary_sensor.office"}),
            ("de", "Sind im ganzen Haus Fenster offen?", "all_areas", "yes", {"binary_sensor.kitchen", "binary_sensor.office"}),
            ("en", "Are any windows open in the whole house?", "all_areas", "yes", {"binary_sensor.kitchen", "binary_sensor.office"}),
        ]
        for language, query, area, answer, expected_ids in scenarios:
            with self.subTest(query=query):
                client = ScriptedModels([
                    {"service": "query_state"}, {"area": area}, {"domain": "binary_sensor"},
                    {"query_type": "yes_no"}, {"binary_answer": answer},
                ])
                result = await AgentEngine(normalize_settings({}), client, decisions_only=True).process(
                    query, language, data, speaker_area="kitchen")
                self.assertEqual(client.requests[1][1]["speaker_area"], "kitchen")
                self.assertEqual(result["speaker_area"], "kitchen")
                self.assertEqual(result["state_answer"], answer)
                self.assertEqual({row["entity_id"] for row in client.requests[-1][1]["entities"]}, expected_ids)

    async def test_query_fallback_receives_speaker_context_for_local_references(self):
        client = ScriptedModels([
            {"service": "query_state"}, {"area": "none"}, {"query_fallback": "entity:light.kitchen"},
            {"query_type": "yes_no"}, {"binary_answer": "yes"},
        ])
        result = await AgentEngine(normalize_settings({}), client, decisions_only=True).process(
            "Is the light on in here?", "en", snapshot(), speaker_area="kitchen")
        self.assertEqual(client.requests[2][1]["speaker_area"], "kitchen")
        self.assertEqual(result["query_fallback"]["status"], "completed")
        self.assertEqual(result["target"]["entity_id"], "light.kitchen")
        self.assertNotIn("answer", result)

    async def test_named_areas_form_a_union_before_domain_and_target_selection(self):
        for language, prompt in (("en", "Which lights are on in the kitchen and office?"),
                                 ("de", "Welche Lichter sind in Küche und Büro an?")):
            with self.subTest(language=language):
                client = ScriptedModels([
                    {"service": "query_state"},
                    area_choice({"area:kitchen": 0.35, "area:office": 0.35, "all_areas": 0.2,
                                 "area:bedroom": 0.03, "area:outside": 0.02, "none": 0.05}),
                    {"domain": "light"}, {"query_type": "complex"},
                    {"targets": ["light.kitchen", "light.office"]}, {"state_property": "state"},
                ])
                execute = AsyncMock()
                events = []
                history = [{"role": "user", "content": "The kitchen and office."}, {"role": "assistant", "content": "Okay"}]
                result = await AgentEngine(normalize_settings({"history_enabled": True}), client, history=history,
                                           execute=execute, trace=lambda *event: events.append(event)).process(
                    prompt, language, snapshot())
                selection = result["area_selection"]
                self.assertFalse(result["all_areas"])
                self.assertEqual(selection["selected_area_ids"], ["kitchen", "office"])
                self.assertIn("all_areas", selection["selected_choices"])
                self.assertEqual(selection["effective_choices"], ["area:kitchen", "area:office"])
                self.assertEqual(selection["entities_before"], 7)
                self.assertEqual(selection["entities_after"], 4)
                self.assertEqual(selection["real_option_count"], 5)
                self.assertEqual(selection["effective_probability_threshold"], 0.16)
                self.assertEqual(result["domain_filter"]["entities_before"], 4)
                self.assertEqual(result["domain_filter"]["entities_after"], 3)
                candidates = client.requests[4][2]["query_targets"]["criteria"]
                self.assertEqual(set(candidates), {"entity:light.kitchen", "entity:light.office", "entity:light.desk", "none"})
                self.assertEqual({state["entity_id"] for state in result["state_info"]}, {"light.kitchen", "light.office"})
                state = "an" if language == "de" else "on"
                self.assertEqual(result["answer"], f"[Küche]: Deckenlampe: {state}. [Büro]: Deckenlampe: {state}.")
                self.assertEqual(client.requests[1][1]["history"], history)
                self.assertEqual(client.requests[1][1]["language"], language)
                self.assertNotIn("entities", client.requests[1][1])
                output = next(output for stage, _, output in events if stage == "area" and output)
                self.assertEqual(output["decisions"]["area"]["probability_threshold"], 0.8)
                self.assertNotIn("minimum_confidence", output["decisions"]["area"])
                self.assertEqual(len(client.requests), 6)
                self.assertFalse(client.chat_requests)
                execute.assert_not_awaited()
                report = {"result": result, "mode": "live", "context": {"entities": 7, "actions": 0},
                          "trace": [{"stage": stage, "input": arguments, "output": output or {}}
                                    for stage, arguments, output in events],
                          "timings_ms": {"snapshot": 0, "engine": 0, "total": 0}}
                printed = io.StringIO()
                with redirect_stdout(printed):
                    local.print_report(report)
                self.assertRegex(printed.getvalue(), r"area filter\s+Küche, Büro")
                self.assertIn("selected: Küche, Büro", printed.getvalue())

    async def test_yes_no_queries_receive_only_entities_from_the_selected_rooms(self):
        client = ScriptedModels([
            {"service": "query_state"}, area_choice({"area:kitchen": 0.5, "area:office": 0.5}),
            {"domain": "light"}, {"query_type": "yes_no"}, {"binary_answer": "yes"},
        ])
        result = await AgentEngine(normalize_settings({}), client).process("Are any kitchen or office lights on?", "en", snapshot())
        states = client.requests[-1][1]["entities"]
        self.assertEqual({state["entity_id"] for state in states}, {"light.kitchen", "light.office", "light.desk"})
        self.assertEqual({area["area_id"] for area in client.requests[-1][1]["area"]["areas"]}, {"kitchen", "office"})
        self.assertEqual(result["answer"], "Yes.")
        self.assertNotIn("target_selection", result)
        self.assertEqual(len(client.requests), 5)

    async def test_all_areas_fallback_includes_unassigned_entities(self):
        client = ScriptedModels([
            {"service": "query_state"}, area_choice({"all_areas": 0.8, "area:kitchen": 0.04, "area:office": 0.04,
                                                   "area:bedroom": 0.04, "area:outside": 0.04, "none": 0.04}),
            {"domain": "light"}, {"query_type": "yes_no"}, {"binary_answer": "yes"},
        ])
        result = await AgentEngine(normalize_settings({}), client).process("Are any lights on in the whole house?", "en", snapshot())
        self.assertTrue(result["all_areas"])
        self.assertEqual(result["selected_areas"], [])
        self.assertEqual(result["area_selection"]["effective_choices"], ["all_areas"])
        self.assertEqual(result["area_selection"]["entities_after"], 7)
        self.assertIn("light.unassigned", {state["entity_id"] for state in client.requests[-1][1]["entities"]})

    async def test_no_retained_area_tries_fallback_before_stopping_without_device_calls(self):
        client = ScriptedModels([{"service": "query_state"}, area_choice({"none": 1}), {"query_fallback": "none"}])
        execute = AsyncMock()
        result = await AgentEngine(normalize_settings({}), client, execute=execute).process("Unknown room", "en", snapshot())
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["query_fallback"]["trigger"]["stage"], "area")
        self.assertEqual(result["failures"][0]["stage"], "query_fallback")
        self.assertEqual(result["area_selection"]["effective_choices"], [])
        self.assertEqual(len(client.requests), 3)
        self.assertNotIn("domain_filter", result)
        execute.assert_not_awaited()

    async def test_malformed_area_choices_are_rejected_without_broadening(self):
        invalid = area_choice({"area:invented": 1})
        client = ScriptedModels([{"service": "query_state"}, invalid])
        execute = AsyncMock()
        with self.assertRaises(ModelError):
            await AgentEngine(normalize_settings({}), client, execute=execute).process("Unknown room", "en", snapshot())
        self.assertEqual(len(client.requests), 2)
        execute.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
