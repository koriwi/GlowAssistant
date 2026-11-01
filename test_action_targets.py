import copy
import unittest
from unittest.mock import AsyncMock

from custom_components.glow_assistant.action_targets import decode_selection
from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.settings import normalize_settings
from test_integration_engine import SNAPSHOT, ScriptedModels, entity


def answer(probabilities):
    return {"type": "choice", "choice": max(probabilities, key=probabilities.get),
            "confidence": 0, "probabilities": probabilities}


def snapshot():
    return {"entities": [
        entity("light.desk", "Desk lamp", area_aliases=["Büro", "buero"]),
        entity("light.ceiling", "Ceiling lamp", area_aliases=["Büro", "buero"]),
        entity("light.kitchen", "Kitchen light", area="kitchen"),
        entity("light.unassigned", "Unassigned light", area=None),
        entity("sensor.office", "Office temperature", state="22"),
        entity("todo.shopping", "Shopping list", area="kitchen"),
        entity("todo.work", "Work list"), entity("todo.unassigned", "Other list", area=None),
    ], "services": {**SNAPSHOT["services"], "todo.add_item": {"target": {}, "fields": {"item": {"required": True}}}}}


class ProbabilitySelectionTests(unittest.TestCase):
    def test_multiple_selection_ignores_confidence_and_none_winner(self):
        data = {"answers": {"action_areas": answer({"office": 0.3, "kitchen": 0.2, "none": 0.5})}}
        selected = decode_selection(data, "action_areas", {"office": {}, "kitchen": {}}, 0.15)
        self.assertEqual(selected["selected_choices"], ["office", "kitchen"])
        self.assertEqual(selected["selection_mode"], "above_probability_threshold")
        self.assertEqual(decode_selection(data, "action_areas", {"office": {}, "kitchen": {}}, 0.3)["selected_choices"], [])

    def test_single_selection_uses_only_unique_highest_probability(self):
        options = {"office": {}, "kitchen": {}}
        for probabilities, expected in (
            ({"office": 0.45, "kitchen": 0.4, "none": 0.15}, ["office"]),
            ({"office": 0.2, "kitchen": 0.3, "none": 0.5}, []),
            ({"office": 0.5, "kitchen": 0.5, "none": 0}, []),
        ):
            result = decode_selection({"answers": {"action_areas": answer(probabilities)}}, "action_areas", options)
            self.assertEqual(result["selected_choices"], expected)
            self.assertEqual(result["selection_mode"], "highest_probability")

    def test_invalid_choices_or_incomplete_probabilities_are_rejected(self):
        valid = {"answers": {"action_entities": answer({"desk": 0.8, "none": 0.2})}}
        for change in ({"choice": "other"}, {"type": "score"}, {"confidence": True},
                       {"probabilities": {"desk": 1}}, {"probabilities": {"desk": float("nan"), "none": 0}}):
            data = copy.deepcopy(valid)
            data["answers"]["action_entities"].update(change)
            with self.subTest(change=change), self.assertRaises(ModelError):
                decode_selection(data, "action_entities", {"desk": {}}, 0.15)


class ActionRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def run_route(self, scope, later=(), speaker_area=None, service="light.turn_off", data=None, settings=None, history=None):
        client = ScriptedModels([{"service": service}, {"target_scope": scope}, *later])
        execute = AsyncMock()
        events = []
        result = await AgentEngine(settings or normalize_settings({}), client, execute,
                                   trace=lambda *event: events.append(event), history=history).process(
            "test request", "de", data or snapshot(), speaker_area=speaker_area,
        )
        execute.assert_not_awaited()
        self.assertFalse(client.chat_requests)
        self.assertNotIn("execution", result)
        self.assertNotIn("parameter_selection", result)
        return result, client, events

    async def test_explicit_areas_are_final_targets_without_entity_decision(self):
        result, client, _ = await self.run_route("areas", [{"action_areas": answer({
            "area:office": 0.3, "area:kitchen": 0.2, "none": 0.5,
        })}])
        self.assertEqual(result["target"]["type"], "areas")
        self.assertEqual(result["target"]["area_ids"], ["office", "kitchen"])
        self.assertEqual(len(client.requests), 3)
        self.assertNotIn("action_entities", result["decisions"])
        self.assertEqual(result["action_target_filter"]["entities_after_domain"], 4)
        self.assertEqual(result["action_target_filter"]["entities_after_area"], 3)

    async def test_implied_area_uses_context_and_needs_no_other_decisions(self):
        result, client, _ = await self.run_route("implied_area", speaker_area="Kitchen")
        self.assertEqual(result["target"], {"type": "area", "area_id": "kitchen", "name": "Kitchen", "aliases": []})
        self.assertEqual(result["area_source"], "speaker_context")
        self.assertEqual(len(client.requests), 2)
        self.assertNotIn("action_areas", result["decisions"])
        self.assertNotIn("action_entities", result["decisions"])

    async def test_entities_in_one_explicit_area_choose_highest_area_then_only_matching_entities(self):
        result, client, events = await self.run_route("entities_explicit_area", [
            {"action_areas": answer({"area:office": 0.45, "area:kitchen": 0.4, "none": 0.15})},
            {"action_entities": answer({"entity:light.desk": 0.4, "entity:light.ceiling": 0.35, "none": 0.25})},
        ], settings=normalize_settings({"decider": {"action_area_probability_threshold": 0.99}}))
        self.assertEqual([area["area_id"] for area in result["selected_areas"]], ["office"])
        self.assertEqual(result["target"]["entity_ids"], ["light.desk", "light.ceiling"])
        self.assertEqual(set(client.requests[-1][2]["action_entities"]["criteria"]),
                         {"entity:light.desk", "entity:light.ceiling", "none"})
        self.assertEqual(result["action_areas"]["selection_mode"], "highest_probability")
        output = next(output for stage, _, output in events if stage == "action_entities" and output)
        self.assertEqual(output["selected_names"], ["Desk lamp", "Ceiling lamp"])
        self.assertEqual(output["probability_threshold"], 0.15)

    async def test_entities_in_implied_area_skip_area_decision_and_resolve_room_alias(self):
        result, client, _ = await self.run_route("entities_implied_area", [
            {"action_entities": answer({"entity:light.desk": 0.8, "entity:light.ceiling": 0.1, "none": 0.1})},
        ], speaker_area="buero")
        self.assertEqual(result["target"]["entity_id"], "light.desk")
        self.assertEqual(result["selected_areas"][0]["area_id"], "office")
        self.assertEqual(len(client.requests), 3)
        self.assertNotIn("action_areas", result["decisions"])

    async def test_arealess_targets_filter_only_by_domain_including_assigned_and_unassigned_entities(self):
        result, client, _ = await self.run_route("entities", [{"action_entities": answer({
            "entity:todo.shopping": 0.6, "entity:todo.work": 0.1, "entity:todo.unassigned": 0.2, "none": 0.1,
        })}], speaker_area="Office", service="todo.add_item")
        self.assertEqual(result["target"]["entity_ids"], ["todo.shopping", "todo.unassigned"])
        self.assertEqual(result["selected_areas"], [])
        self.assertEqual(result["action_target_filter"]["entities_after_domain"], 3)
        self.assertEqual(result["action_target_filter"]["entities_after_area"], 3)
        self.assertEqual(set(client.requests[-1][2]["action_entities"]["criteria"]),
                         {"entity:todo.shopping", "entity:todo.work", "entity:todo.unassigned", "none"})
        self.assertNotIn("action_areas", result["decisions"])

    async def test_unknown_speaker_area_stops_without_guessing_another_room(self):
        result, client, _ = await self.run_route("implied_area", speaker_area="Unknown room")
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["failures"][0]["stage"], "action_areas")
        self.assertEqual(len(client.requests), 2)
        self.assertIsNone(result["target"])

    async def test_none_or_tied_area_selection_stops_without_entity_call(self):
        for probabilities in ({"area:office": 0.1, "area:kitchen": 0.2, "none": 0.7},
                              {"area:office": 0.5, "area:kitchen": 0.5, "none": 0}):
            result, client, _ = await self.run_route("entities_explicit_area", [{"action_areas": answer(probabilities)}])
            self.assertEqual(result["route"], "unsure")
            self.assertEqual(len(client.requests), 3)
            self.assertIsNone(result["target"])

    async def test_empty_entity_selection_does_not_restore_all_candidates(self):
        result, client, _ = await self.run_route("entities_implied_area", [{"action_entities": answer({
            "entity:light.desk": 0, "entity:light.ceiling": 0, "none": 1,
        })}], speaker_area="Office")
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["failures"][0]["stage"], "action_entities")
        self.assertIsNone(result["target"])
        self.assertEqual(len(client.requests), 3)

    async def test_domain_and_area_mismatch_does_not_fall_back_to_another_area(self):
        data = snapshot()
        data["entities"].append(entity("sensor.garden", "Garden temperature", area="garden"))
        result, client, _ = await self.run_route("areas", [{"action_areas": "area:garden"}], data=data)
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["action_target_filter"]["entities_after_area"], 0)
        self.assertEqual(len(client.requests), 3)

    async def test_language_history_and_custom_instructions_reach_area_and_entity_selection(self):
        history = [{"role": "user", "content": "The office"}, {"role": "assistant", "content": "Okay"}]
        settings = normalize_settings({"history_enabled": True, "decider": {
            "action_area_prompt": "Custom area task", "action_entity_prompt": "Custom entity task",
        }})
        result, client, events = await self.run_route("entities_explicit_area", [
            {"action_areas": "area:office"}, {"action_entities": "entity:light.desk"},
        ], settings=settings, history=history)
        for _, state, _ in client.requests:
            self.assertEqual(state["language"], "de")
            self.assertEqual(state["history"], history)
        self.assertIn("Custom area task", client.requests[-2][2]["action_areas"]["instructions"])
        self.assertIn("Custom entity task", client.requests[-1][2]["action_entities"]["instructions"])
        self.assertEqual(result["target"]["entity_id"], "light.desk")
        self.assertNotIn("Custom area task", str(events))
        self.assertNotIn("Custom entity task", str(events))
