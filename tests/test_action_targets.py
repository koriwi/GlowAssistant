import copy
import unittest
from unittest.mock import AsyncMock, patch

from custom_components.glow_assistant.action_targets import decode_selection
from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.settings import normalize_settings
from tests.test_integration_engine import SNAPSHOT, ScriptedModels, entity


def answer(probabilities, confidence=0):
    return {"type": "choice", "choice": max(probabilities, key=probabilities.get),
            "confidence": confidence, "probabilities": probabilities}


def snapshot():
    data = {"entities": [
        entity("light.desk", "Desk lamp", area_aliases=["Büro", "buero"]),
        entity("light.ceiling", "Ceiling lamp", area_aliases=["Büro", "buero"]),
        entity("light.kitchen", "Kitchen light", area="kitchen"),
        entity("light.unassigned", "Unassigned light", area=None),
        entity("sensor.office", "Office temperature", state="22"),
        entity("todo.shopping", "Shopping list", area="kitchen"),
        entity("todo.work", "Work list"), entity("todo.unassigned", "Other list", area=None),
    ], "services": {**copy.deepcopy(SNAPSHOT["services"]), "todo.add_item": {"target": {}, "fields": {"item": {"required": True}}}}}
    for item in data["entities"]:
        if item["area_id"] == "office":
            item["area_aliases"] = ["Büro", "buero"]
    return data


def climate_snapshot():
    data = snapshot()
    data["entities"].extend([
        entity("climate.office", "Office thermostat", temperature=20, supported_features=401),
        entity("climate.kitchen", "Kitchen thermostat", area="kitchen", temperature=21, supported_features=2),
    ])
    data["services"]["climate.set_temperature"] = {"target": {}, "fields": {
        "temperature": {"filter": {"supported_features": [1]},
                        "selector": {"number": {"min": 0, "max": 250, "step": 0.1}}},
        "temperature_range": {"fields": {
            "target_temp_low": {"filter": {"supported_features": [2]}, "selector": {"number": {"min": 0, "max": 250}}},
            "target_temp_high": {"filter": {"supported_features": [2]}, "selector": {"number": {"min": 0, "max": 250}}},
        }},
        "hvac_mode": {"selector": {"state": {}}},
    }}
    return data


class ProbabilitySelectionTests(unittest.TestCase):
    def test_multiple_selection_ignores_confidence_and_none_winner(self):
        data = {"answers": {"action_areas": answer({"office": 0.3, "kitchen": 0.2, "none": 0.5})}}
        selected = decode_selection(data, "action_areas", {"office": {}, "kitchen": {}}, 0.15)
        self.assertEqual(selected["selected_choices"], ["office", "kitchen"])
        self.assertEqual(selected["selection_mode"], "at_or_above_probability_threshold")
        self.assertEqual(decode_selection(data, "action_areas", {"office": {}, "kitchen": {}}, 1)["selected_choices"], [])

    def test_relative_cutoff_retains_ten_valid_targets_in_a_large_choice(self):
        options = {f"entity:{index}": {} for index in range(11)}
        probabilities = {key: 0.1 if index < 10 else 0 for index, key in enumerate(options)}
        probabilities["none"] = 0
        for question in ("action_areas", "action_entities", "query_targets", "text_words"):
            with self.subTest(question=question):
                result = decode_selection({"answers": {question: answer(probabilities)}}, question, options, 0.3)
                self.assertEqual(result["selected_choices"], list(options)[:10])
                self.assertEqual(result["real_option_count"], 11)
                self.assertAlmostEqual(result["effective_probability_threshold"], 0.3 / 11)

    def test_two_real_options_halve_the_cutoff_and_include_the_boundary(self):
        options = {"first": {}, "second": {}}
        result = decode_selection({"answers": {"action_entities": answer({"first": 0.75, "second": 0.15, "none": 0.1})}},
                                  "action_entities", options, 0.3)
        self.assertEqual(result["selected_choices"], ["first", "second"])
        self.assertEqual(result["effective_probability_threshold"], 0.15)
        self.assertEqual(result["real_option_count"], 2)
        for options, probabilities, expected in (({}, {"none": 1}, []),
                                                  ({"first": {}}, {"first": 0.3, "none": 0.7}, ["first"]),
                                                  ({"first": {}}, {"first": 0.29, "none": 0.71}, [])):
            result = decode_selection({"answers": {"action_entities": answer(probabilities)}}, "action_entities", options, 0.3)
            self.assertEqual(result["selected_choices"], expected)
            self.assertEqual(result["effective_probability_threshold"], 0.3)

    def test_threshold_one_retains_uniform_real_options_for_every_multi_select_stage(self):
        for count in (1, 2, 3, 10):
            options = {f"option:{index}": {} for index in range(count)}
            probabilities = {key: 1 / count for key in options} | {"none": 0}
            for question in ("action_areas", "action_entities", "query_targets", "text_words"):
                with self.subTest(count=count, question=question):
                    result = decode_selection({"answers": {question: answer(probabilities)}}, question, options, 1)
                    self.assertEqual(result["selected_choices"], list(options))
                    self.assertEqual(result["effective_probability_threshold"], 1 / count)

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
    async def run_route(self, scope, later=(), speaker_area=None, service="light.turn_off", data=None, settings=None, history=None, selected_parameters=(), prompt="test request"):
        client = ScriptedModels([{"service": service}, {"target_scope": scope}, *later])

        events = []
        with patch.object(AgentEngine, "_extract_parameters", new_callable=AsyncMock), \
                patch.object(AgentEngine, "_execute_action", new_callable=AsyncMock):
            result = await AgentEngine(settings or normalize_settings({}), client,
                                       trace=lambda *event: events.append(event), history=history, dry_run=True).process(
                prompt, "de", data or snapshot(), speaker_area=speaker_area,
            )

        self.assertFalse(client.chat_requests)
        self.assertNotIn("execution", result)
        if result["route"] == "unsure":
            self.assertNotIn("parameter_selection", result)
        else:
            self.assertEqual(result["parameter_selection"]["selected_parameters"], list(selected_parameters))
        return result, client, events

    async def test_explicit_areas_are_final_targets_without_entity_decision(self):
        result, client, _ = await self.run_route("areas", [{"action_areas": answer({
            "area:office": 0.45, "area:kitchen": 0.4, "none": 0.15,
        })}], settings=normalize_settings({"decider": {"action_area_confidence": 1}}))
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
            {"action_areas": answer({"area:office": 0.45, "area:kitchen": 0.4, "none": 0.15}, confidence=0.9)},
            {"action_entities": answer({"entity:light.desk": 0.45, "entity:light.ceiling": 0.4, "none": 0.15})},
        ], settings=normalize_settings({"decider": {"action_area_probability_threshold": 0.99}}))
        self.assertEqual([area["area_id"] for area in result["selected_areas"]], ["office"])
        self.assertEqual(result["target"]["entity_ids"], ["light.desk", "light.ceiling"])
        self.assertEqual(set(next(request for request in client.requests if "action_entities" in request[2])[2]["action_entities"]["criteria"]),
                         {"entity:light.desk", "entity:light.ceiling", "none"})
        self.assertEqual(result["action_areas"]["selection_mode"], "highest_probability")
        output = next(output for stage, _, output in events if stage == "action_entities" and output)
        self.assertEqual(output["selected_names"], ["Desk lamp", "Ceiling lamp"])
        self.assertEqual(output["probability_threshold"], 0.8)

    async def test_single_area_confidence_gates_entity_selection_and_execution_at_the_boundary(self):
        for confidence, accepted in ((0.79, False), (0.8, True), (0.95, True)):
            with self.subTest(confidence=confidence):
                client = ScriptedModels([
                    {"service": "light.turn_off"}, {"target_scope": "entities_explicit_area"},
                    {"action_areas": answer({"area:office": 0.96, "area:kitchen": 0.02, "none": 0.02}, confidence=confidence)},
                    {"action_entities": "entity:light.desk"},
                ])
                execute = AsyncMock(return_value=None)
                events = []
                result = await AgentEngine(normalize_settings({}), client, execute=execute,
                                           trace=lambda *event: events.append(event)).process(
                    "Turn off the desk lamp in the office", "en", snapshot())
                output = next(output for stage, _, output in events if stage == "action_areas" and output)
                decision = output["decisions"]["action_areas"]
                self.assertEqual(output["confidence"], confidence)
                self.assertEqual(output["minimum_confidence"], 0.8)
                self.assertEqual(decision["minimum_confidence"], 0.8)
                self.assertIsNone(decision["effective_probability_threshold"])
                self.assertEqual(decision["selection_basis"], "unique_highest_probability_and_minimum_confidence")
                self.assertEqual(decision["effective_choice"], "area:office" if accepted else "none")
                if accepted:
                    self.assertEqual(result["route"], "smarthome")
                    execute.assert_awaited_once_with("light.turn_off", {"entity_id": ["light.desk"]})
                else:
                    self.assertEqual(result["route"], "unsure")
                    self.assertEqual(len(client.requests), 3)
                    self.assertIsNone(result["target"])
                    self.assertEqual(result["failures"][0]["stage"], "action_areas")
                    self.assertIn("0.79", result["failures"][0]["reason"])
                    self.assertNotIn("execution", result)
                    execute.assert_not_awaited()

    async def test_entities_in_implied_area_skip_area_decision_and_resolve_room_alias(self):
        result, client, _ = await self.run_route("entities_implied_area", [
            {"action_entities": answer({"entity:light.desk": 0.85, "entity:light.ceiling": 0.05, "none": 0.1})},
        ], speaker_area="buero")
        self.assertEqual(result["target"]["entity_id"], "light.desk")
        self.assertEqual(result["selected_areas"][0]["area_id"], "office")
        self.assertEqual(len(client.requests), 3)
        self.assertNotIn("action_areas", result["decisions"])

    async def test_arealess_targets_filter_only_by_domain_including_assigned_and_unassigned_entities(self):
        result, client, _ = await self.run_route("entities", [{"action_entities": answer({
            "entity:todo.shopping": 0.55, "entity:todo.work": 0.04, "entity:todo.unassigned": 0.31, "none": 0.1,
        })}, {"text_words": answer({"word:0:test": 0, "word:1:request": 0.9, "none": 0.1})}], speaker_area="Office", service="todo.add_item")
        self.assertEqual(result["target"]["entity_ids"], ["todo.shopping", "todo.unassigned"])
        self.assertEqual(result["selected_areas"], [])
        self.assertEqual(result["action_target_filter"]["entities_after_domain"], 3)
        self.assertEqual(result["action_target_filter"]["entities_after_area"], 3)
        self.assertEqual(set(next(request for request in client.requests if "action_entities" in request[2])[2]["action_entities"]["criteria"]),
                         {"entity:todo.shopping", "entity:todo.work", "entity:todo.unassigned", "none"})
        self.assertNotIn("action_areas", result["decisions"])
        self.assertEqual(result["required_parameters"], ["item"])
        self.assertNotIn("optional_parameters", result["decisions"])

    async def test_parameter_detection_runs_after_resolving_target_for_relative_climate_request(self):
        data = climate_snapshot()
        result, client, events = await self.run_route("implied_area", [{"optional_parameters": answer({
            "temperature": 0.89, "hvac_mode": 0.01, "none": 0.1,
        })}], speaker_area="Office", service="climate.set_temperature:temperature", data=data,
            selected_parameters=["temperature"], prompt="mach ein bisschen waermer")
        self.assertEqual([set(questions) for _, _, questions in client.requests],
                         [{"service"}, {"target_scope"}, {"optional_parameters"}])
        self.assertEqual(client.requests[-1][1]["target"], result["target"])
        self.assertEqual(result["target"]["area_id"], "office")
        self.assertNotIn("entities", client.requests[-1][1])
        self.assertNotIn("current", client.requests[-1][1])
        self.assertEqual(result["required_parameters"], [])
        self.assertNotIn("numeric", result["decisions"])
        self.assertEqual(set(client.requests[-1][2]["optional_parameters"]["criteria"]), {"temperature", "hvac_mode", "none"})
        self.assertEqual(client.requests[-1][1]["capability_filter_policy"], "any_resolved_target")
        self.assertEqual(result["parameter_capabilities"]["target_entity_ids"], ["climate.office"])
        self.assertEqual(set(result["parameter_capabilities"]["removed_fields"]), {"target_temp_low", "target_temp_high"})
        capabilities = next(output for stage, _, output in events if stage == "parameter_capabilities" and output)
        self.assertFalse(capabilities["model_call"])
        self.assertEqual(capabilities["fields_after"], 2)
        output = next(output for stage, _, output in events if stage == "optional_parameters" and output)
        self.assertEqual(output["selected_parameters"], ["temperature"])

    async def test_range_target_excludes_single_temperature_before_parameter_detection(self):
        result, client, _ = await self.run_route("implied_area", [{"optional_parameters": answer({
            "target_temp_low": 0.4, "target_temp_high": 0.39, "hvac_mode": 0.01, "none": 0.2,
        })}], speaker_area="Kitchen", service="climate.set_temperature:temperature", data=climate_snapshot(),
            selected_parameters=["target_temp_low", "target_temp_high"])
        self.assertNotIn("temperature", client.requests[-1][2]["optional_parameters"]["criteria"])
        self.assertEqual(result["parameter_capabilities"]["removed_fields"]["temperature"]["unsupported_entity_ids"],
                         ["climate.kitchen"])

    async def test_multi_area_targets_retain_partially_supported_fields_with_per_field_members(self):
        result, client, _ = await self.run_route("areas", [
            {"action_areas": answer({"area:office": 0.5, "area:kitchen": 0.4, "none": 0.1})},
            {"optional_parameters": "none"},
        ], service="climate.set_temperature:temperature", data=climate_snapshot())
        self.assertEqual(set(client.requests[-1][2]["optional_parameters"]["criteria"]),
                         {"temperature", "target_temp_low", "target_temp_high", "hvac_mode", "none"})
        capabilities = result["parameter_capabilities"]
        self.assertEqual(capabilities["target_entity_ids"], ["climate.office", "climate.kitchen"])
        self.assertEqual(capabilities["field_support"]["temperature"]["supported_entity_ids"], ["climate.office"])
        self.assertEqual(capabilities["field_support"]["target_temp_low"]["supported_entity_ids"], ["climate.kitchen"])

    async def test_specific_entity_capabilities_ignore_unselected_candidates_in_same_area(self):
        data = climate_snapshot()
        data["entities"].append(entity("climate.office_range", "Other office thermostat", supported_features=2))
        result, client, _ = await self.run_route("entities_implied_area", [
            {"action_entities": answer({"entity:climate.office": 0.9, "entity:climate.office_range": 0.05, "none": 0.05})},
            {"optional_parameters": "temperature"},
        ], speaker_area="Office", service="climate.set_temperature:temperature", data=data, selected_parameters=["temperature"])
        self.assertEqual(result["parameter_capabilities"]["target_entity_ids"], ["climate.office"])
        self.assertIn("temperature", client.requests[-1][2]["optional_parameters"]["criteria"])

    async def test_all_optional_fields_filtered_out_skip_the_parameter_model_call(self):
        data = snapshot()
        data["services"]["light.turn_on"]["fields"] = {"brightness_pct": {"filter": {"supported_features": [1]}}}
        result, client, events = await self.run_route("implied_area", speaker_area="Office", service="light.turn_on", data=data)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(result["parameter_capabilities"]["fields_after"], 0)
        self.assertEqual(result["parameter_capabilities"]["removed_fields"]["brightness_pct"]["unsupported_entity_ids"],
                         ["light.desk", "light.ceiling"])
        self.assertNotIn("optional_parameters", result["decisions"])
        self.assertTrue(any(stage == "parameter_capabilities" for stage, _, _ in events))

    async def test_room_brightness_step_survives_on_off_only_members(self):
        data = snapshot()
        data["entities"][0]["attributes"]["supported_color_modes"] = ["brightness"]
        data["entities"][1]["attributes"]["supported_color_modes"] = ["onoff"]
        data["entities"][2]["attributes"]["supported_color_modes"] = ["rgb"]
        data["entities"].extend(entity(f"light.binary_{index}", f"Binary light {index}", supported_color_modes=["onoff"])
                                for index in range(6))
        brightness_filter = {"attribute": {"supported_color_modes": ["brightness", "color_temp", "hs", "xy", "rgb", "rgbw", "rgbww"]}}
        data["services"]["light.turn_on"]["fields"] = {
            "brightness_pct": {"filter": brightness_filter, "selector": {"number": {"min": 0, "max": 100}}},
            "brightness_step_pct": {"filter": brightness_filter, "selector": {"number": {"min": -100, "max": 100}}},
            "rgb_color": {"filter": {"attribute": {"supported_color_modes": ["rgb", "hs"]}}},
            "transition": {"filter": {"supported_features": [32]}},
        }
        result, client, _ = await self.run_route("implied_area", [{"optional_parameters": answer({
            "brightness_pct": 0.05, "brightness_step_pct": 0.9, "none": 0.05,
        })}], speaker_area="Office", service="light.turn_on:brightness_pct", data=data,
            selected_parameters=["brightness_step_pct"], prompt="mach das licht ein bisschen heller")
        self.assertEqual(result["target"]["area_id"], "office")
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(set(client.requests[-1][2]["optional_parameters"]["criteria"]),
                         {"brightness_pct", "brightness_step_pct", "none"})
        capabilities = result["parameter_capabilities"]
        self.assertEqual(len(capabilities["target_entity_ids"]), 8)
        self.assertEqual(capabilities["field_support"]["brightness_step_pct"]["supported_entity_ids"], ["light.desk"])
        self.assertEqual(len(capabilities["field_support"]["brightness_step_pct"]["unsupported_entity_ids"]), 7)
        self.assertEqual(set(capabilities["removed_fields"]), {"rgb_color", "transition"})

    async def test_unknown_speaker_area_stops_without_guessing_another_room(self):
        result, client, _ = await self.run_route("implied_area", speaker_area="Unknown room")
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["failures"][0]["stage"], "action_areas")
        self.assertEqual(len(client.requests), 2)
        self.assertIsNone(result["target"])

    async def test_none_or_tied_area_selection_stops_without_entity_call(self):
        for probabilities in ({"area:office": 0.1, "area:kitchen": 0.2, "none": 0.7},
                              {"area:office": 0.5, "area:kitchen": 0.5, "none": 0}):
            result, client, _ = await self.run_route("entities_explicit_area", [{"action_areas": answer(probabilities, confidence=0.95)}])
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
