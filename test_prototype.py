import asyncio
import copy
import io
import json
import sys
import unittest
from contextlib import asynccontextmanager, redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from prototype import area_request, current_numeric_values, decision_request, execute_mcp, parse_entities, print_result, query_state_info, run, selection_result, service_request, target_arguments


CONTEXT = """Live Context: An overview of the areas and the devices in this smart home:
- names: Desk lamp, Work light
  domain: light
  state: 'off'
  areas: Office, Study
- names: Desk lamp
  domain: light
  state: 'on'
  areas: Bedroom
"""
TARGET_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "area": {"type": "string"},
    "domain": {"type": "array", "items": {"type": "string"}},
}}
INSPECTION = {
    "context": json.dumps({"success": True, "result": CONTEXT}),
    "tools": [
        {"name": "assist__HassTurnOn", "description": "Turn on a device", "inputSchema": TARGET_SCHEMA},
        {"name": "assist__HassTurnOff", "description": "Turn off a device", "inputSchema": TARGET_SCHEMA},
    ],
}


def response(choice="entity_001"):
    probabilities = dict.fromkeys(("entity_001", "entity_002", "area_001", "area_002", "other"), 0.05)
    if choice in probabilities:
        probabilities[choice] = 0.8
    return {
        "answers": {
            "target": {
                "type": "choice",
                "choice": choice,
                "confidence": 0.65,
                "probabilities": probabilities,
            }
        },
        "usage": {"cost": 0.00002},
    }


def service_response(tool="assist__HassTurnOff", domain="light"):
    tool_scores = dict.fromkeys(("assist__HassTurnOn", "assist__HassTurnOff", "query_state", "other"), 0.1)
    tool_scores[tool] = 0.7
    domain_scores = {"light": 0.1, "other": 0.1}
    domain_scores[domain] = 0.9
    return {
        "answers": {
            "service": {"type": "choice", "choice": tool, "confidence": 0.65, "probabilities": tool_scores},
            "domain": {"type": "choice", "choice": domain, "confidence": 0.75, "probabilities": domain_scores},
        },
        "usage": {"cost": 0.00003},
    }


def area_response(inspection, choice):
    options, _ = area_request("test", parse_entities(inspection))
    scores = dict.fromkeys([*options, "other"], 0)
    scores[choice] = 1
    return {"answers": {"area": {
        "type": "choice", "choice": choice, "confidence": 1, "probabilities": scores,
    }}, "usage": {"cost": 0.000005}}


def numeric_case(parameter="temperature", currents=(20.5,), mode="relative", score=4):
    climate = parameter == "temperature"
    domain = "climate" if climate else "light"
    tool_name = "assist__HassClimateSetTemperature" if climate else "assist__HassLightSet"
    properties = copy.deepcopy(TARGET_SCHEMA["properties"])
    if climate:
        properties.pop("domain")
        properties["temperature"] = {"type": "number"}
    else:
        properties.update(brightness={"type": "integer", "minimum": 0, "maximum": 100},
                          color={"type": "string"}, temperature={"type": "integer"})
    entities = [
        {"names": f"Device {index}", "domain": domain, "state": "heat" if climate else "on",
         "areas": "Office, Study", "attributes": {"temperature": current, "current_temperature": 24.5}
         if climate else {"brightness": current}}
        for index, current in enumerate(currents, 1)
    ]
    inspection = {
        "context": json.dumps(entities),
        "tools": [{"name": tool_name, "description": f"Set {parameter}",
                   "inputSchema": {"type": "object", "properties": properties}}],
    }
    first = service_response()
    first["answers"]["service"].update(choice=tool_name,
        probabilities={tool_name: 0.9, "query_state": 0.05, "other": 0.05})
    first["answers"]["domain"].update(choice=domain, probabilities={domain: 0.9, "other": 0.1})
    target = response()
    target["answers"]["target"]["probabilities"] = {
        **{f"entity_{index:03}": 0 for index in range(1, len(entities) + 1)},
        "area_001": 0.1, "other": 0.1,
    }
    target["answers"]["target"]["probabilities"]["entity_001"] = 0.8
    levels = 9 if climate else 6
    scores = {str(index): 0 for index in range(levels)}
    lower = int(score)
    scores[str(lower)] = 1 - (score - lower)
    if score != lower:
        scores[str(lower + 1)] = score - lower
    modes = {"absolute": 0.1, "relative": 0.1, "other": 0.1}
    modes[mode] = 0.8
    numeric = {"answers": {
        "numeric_score": {"type": "score", "score": score, "confidence": 0.8, "probabilities": scores},
        "numeric_mode": {"type": "choice", "choice": mode, "confidence": 0.8, "probabilities": modes},
    }, "usage": {"cost": 0.00004}}
    return inspection, [first, target, numeric]


class PrototypeTests(unittest.TestCase):
    def run_cli(self, responses, inspection=INSPECTION, json_output=True,
                prompt="turn off the lights in the office", executor=None, exit_code=0, area_choice="no_area",
                answer_settings=None):
        inspector = AsyncMock(return_value=inspection)
        responses = list(responses)
        if len(responses) > 1:
            responses.insert(1, area_response(inspection, area_choice))
        decider = AsyncMock(side_effect=responses)
        async def completed(session, name, arguments):
            return {"status": "completed", "arguments": arguments, "response": {"isError": False}}
        if executor is None:
            executor = AsyncMock(side_effect=completed)
        session = SimpleNamespace()
        @asynccontextmanager
        async def connection(url, token):
            yield session
        output = io.StringIO()
        argv = ["prototype.py", prompt]
        if json_output:
            argv.append("--json")
        with (
            patch.object(sys, "argv", argv),
            patch("prototype.load_dotenv"),
            patch("prototype.answer_config", return_value=answer_settings),
            patch("prototype.setting", return_value="test-credential"),
            patch("prototype.mcp_session", new=connection),
            patch("prototype.inspect_mcp", new=inspector),
            patch("prototype.call_decider", new=decider),
            patch("prototype.execute_mcp", new=executor),
            redirect_stdout(output),
        ):
            self.assertEqual(asyncio.run(run()), exit_code)
        return output.getvalue(), inspector, decider

    def test_answer_generation_is_disabled_without_configuration(self):
        with patch("prototype.generate_answer", new_callable=AsyncMock) as generator:
            output, _, _ = self.run_cli([service_response(), response()])
        generator.assert_not_awaited()
        self.assertNotIn("answer", json.loads(output))

    def test_query_answer_receives_selected_cached_state(self):
        with patch("prototype.generate_answer", new_callable=AsyncMock,
                   return_value=("The desk lamp is off.", {"completion_tokens": 7})) as generator:
            output, _, _ = self.run_cli(
                [service_response("query_state"), response()], answer_settings={"model": "small"},
                prompt="Is the desk lamp off?",
            )
        result = json.loads(output)
        self.assertEqual(result["answer"], "The desk lamp is off.")
        self.assertEqual(result["usage"]["answer"], {"completion_tokens": 7})
        self.assertIn("answer", result["timings_ms"])
        self.assertEqual(generator.await_args.args[0]["state_info"][0]["state"], "off")
        self.assertNotIn("execution", result)

    def test_answer_generation_runs_after_execution(self):
        executor = AsyncMock(return_value={"status": "completed", "arguments": {"name": "Desk lamp"}})
        async def reply(result, config):
            executor.assert_awaited_once()
            self.assertEqual(result["execution"]["status"], "completed")
            return "The lamp has been turned off.", {}
        with patch("prototype.generate_answer", new=reply):
            output, _, _ = self.run_cli(
                [service_response(), response()], answer_settings={"model": "small"},
                executor=executor, json_output=False,
            )
        self.assertIn("answer: The lamp has been turned off.", output)

    def test_answer_failure_preserves_execution_and_does_not_repeat_action(self):
        executor = AsyncMock(return_value={"status": "completed", "arguments": {"name": "Desk lamp"}})
        with patch("prototype.generate_answer", new_callable=AsyncMock,
                   side_effect=ValueError("Answer endpoint HTTP 503")) as generator:
            output, _, _ = self.run_cli(
                [service_response(), response()], answer_settings={"model": "small"},
                executor=executor, exit_code=1,
            )
        result = json.loads(output)
        executor.assert_awaited_once()
        generator.assert_awaited_once()
        self.assertEqual(result["execution"]["status"], "completed")
        self.assertEqual(result["answer_error"], "Answer endpoint HTTP 503")
        self.assertNotIn("answer", result)

    def test_tool_and_resource_context_preserve_aliases_and_state(self):
        entities = parse_entities(INSPECTION)
        self.assertEqual(entities, parse_entities({"context": CONTEXT}))
        self.assertEqual(entities[0]["names"], "Desk lamp, Work light")
        self.assertEqual(entities[0]["areas"], "Office, Study")
        self.assertEqual(entities[0]["state"], "off")

    def test_failed_empty_and_invalid_contexts_are_errors(self):
        for inspection in (
            {"context_error": "GetLiveContext failed"},
            {},
            {"context": json.dumps({"success": False, "result": CONTEXT})},
            {"context": "[]"},
            {"context": "- names: Lamp\n"},
        ):
            with self.subTest(inspection=inspection), self.assertRaises(ValueError):
                parse_entities(inspection)

    def test_duplicate_entries_remain_separate_choices(self):
        entity = {"names": "Same name", "domain": "switch"}
        options, payload = decision_request("turn it on", [entity, entity])
        self.assertEqual(list(options), ["entity_001", "entity_002"])
        self.assertEqual(len(payload["questions"]["target"]["criteria"]), 3)

    def test_area_choices_group_aliases_and_available_domains(self):
        entities = parse_entities(INSPECTION) + [
            {"names": "Floor lamp", "domain": "light", "areas": "Office, Study"},
            {"names": "Outlet", "domain": "switch", "areas": " Office, Study, Workroom "},
            {"names": "Unassigned", "domain": "sensor"},
        ]
        options, _ = decision_request("turn off the lights in the study", entities)
        areas = [target for target in options.values() if target["type"] == "area"]
        self.assertEqual(len(areas), 2)
        self.assertEqual(areas[0], {
            "type": "area", "name": "Office", "aliases": ["Study", "Workroom"],
            "domains": ["light", "switch"],
        })
        self.assertEqual(len([target for target in options.values() if target["type"] == "entity"]), 5)

    def test_area_selection_maps_to_room_instead_of_one_lamp(self):
        options, _ = decision_request("turn off the lights in the study", parse_entities(INSPECTION))
        result = selection_result(response("area_001"), options)
        self.assertEqual(result["target"], {
            "type": "area", "name": "Office", "aliases": ["Study"], "domains": ["light"],
        })
        self.assertEqual(result["probability"], 0.8)

    def test_selection_maps_back_to_metadata_and_preserves_scores(self):
        entities = parse_entities(INSPECTION)
        options, payload = decision_request("turn on the office desk lamp", entities)
        result = selection_result(response(), options)
        self.assertEqual(payload["state"]["prompt"], "turn on the office desk lamp")
        self.assertEqual(result["target"], {**entities[0], "type": "entity"})
        self.assertEqual(result["probability"], 0.8)
        self.assertEqual(result["confidence"], 0.65)

    def test_service_choices_come_from_advertised_tools_and_context_domains(self):
        entities = parse_entities(INSPECTION)
        tools, domains, payload = service_request("turn off the lights", INSPECTION, entities)
        self.assertEqual(set(tools), {"assist__HassTurnOn", "assist__HassTurnOff", "query_state"})
        self.assertEqual(domains, {"light": "light"})
        self.assertEqual(payload["state"], {"prompt": "turn off the lights"})
        self.assertEqual(set(payload["questions"]["service"]["criteria"]), set(tools) | {"other"})
        services, _, _ = service_request("is the lamp on?", {"tools": []}, entities)
        self.assertEqual(set(services), {"query_state"})

    def test_area_request_contains_all_names_and_aliases_without_entity_metadata(self):
        entities = parse_entities(INSPECTION) + [
            {"names": "Outlet", "domain": "switch", "areas": "Garage", "state": "on"},
            {"names": "Temperature", "domain": "sensor", "areas": " Office, Study, Workroom "},
            {"names": "Unassigned", "domain": "light"},
        ]
        options, payload = area_request("turn off lights in the study", entities)
        areas = [value for key, value in options.items() if key != "no_area"]
        self.assertEqual(areas, [
            {"name": "Office", "aliases": ["Study", "Workroom"]},
            {"name": "Bedroom", "aliases": []},
            {"name": "Garage", "aliases": []},
        ])
        self.assertEqual(payload["state"], {"prompt": "turn off lights in the study"})
        self.assertEqual(set(payload["questions"]["area"]["criteria"]),
                         {"area_001", "area_002", "area_003", "no_area", "other"})

    def test_target_request_filters_both_domain_and_canonical_area(self):
        entities = parse_entities(INSPECTION) + [
            {"names": "Outlet", "domain": "switch", "areas": "Office, Study"},
            {"names": "Unassigned light", "domain": "light"},
        ]
        options, payload = decision_request("turn off lights in the study", entities,
                                            {"tool": "assist__HassTurnOff", "domain": "light"}, "Office")
        self.assertEqual(set(options), {"entity_001", "area_001"})
        self.assertEqual(options["entity_001"]["names"], "Desk lamp, Work light")
        self.assertEqual(options["area_001"]["name"], "Office")
        self.assertEqual(payload["state"]["area"], "Office")

    def test_area_selection_filters_next_call_and_preserves_canonical_execution_target(self):
        target = response("area_001")
        target["answers"]["target"]["probabilities"] = {"entity_001": 0.1, "area_001": 0.8, "other": 0.1}
        executor = AsyncMock(return_value={"status": "completed"})
        output, inspector, decider = self.run_cli(
            [service_response(), target], executor=executor, area_choice="area_001",
            prompt="turn off all lights in the study",
        )
        result = json.loads(output)
        self.assertEqual(result["area"], "Office")
        self.assertEqual(result["decisions"]["area"]["area"]["aliases"], ["Study"])
        self.assertEqual(result["decisions"]["target"]["candidate_count"], 2)
        self.assertEqual(decider.await_count, 3)
        target_payload, _ = decider.await_args_list[2].args
        self.assertEqual(set(target_payload["questions"]["target"]["criteria"]),
                         {"entity_001", "area_001", "other"})
        executor.assert_awaited_once_with(inspector.await_args.args[0], "assist__HassTurnOff", {
            "domain": ["light"], "area": "Office",
        })
        inspector.assert_awaited_once()

    def test_no_area_selection_keeps_all_domain_candidates(self):
        output, _, decider = self.run_cli([service_response(), response()],
                                         area_choice="no_area", prompt="turn off the work light")
        result = json.loads(output)
        self.assertIsNone(result["area"])
        self.assertEqual(result["decisions"]["area"]["choice"], "no_area")
        self.assertEqual(result["decisions"]["target"]["candidate_count"], 4)
        self.assertEqual(decider.await_count, 3)

    def test_unknown_or_multiple_area_selection_stops_before_target_or_execution(self):
        for service in ("assist__HassTurnOff", "query_state"):
            for prompt in ("turn off lights in an unknown room", "lights in office and bedroom"):
                with self.subTest(service=service, prompt=prompt):
                    executor = AsyncMock()
                    output, inspector, decider = self.run_cli(
                        [service_response(service), response()], area_choice="other", prompt=prompt, executor=executor,
                    )
                    result = json.loads(output)
                    self.assertIsNone(result["target"])
                    self.assertIsNone(result["decisions"]["target"])
                    self.assertEqual(result["decisions"]["area"]["choice"], "other")
                    self.assertEqual(decider.await_count, 2)
                    executor.assert_not_awaited()
                    inspector.assert_awaited_once()

    def test_known_area_without_selected_domain_does_not_fall_back_to_other_rooms(self):
        inspection = copy.deepcopy(INSPECTION)
        entities = parse_entities(inspection) + [
            {"names": "Garage outlet", "domain": "switch", "areas": "Garage"},
        ]
        inspection["context"] = json.dumps(entities)
        first = service_response()
        first["answers"]["domain"]["probabilities"]["switch"] = 0
        executor = AsyncMock()
        output, _, decider = self.run_cli([first, response()], inspection=inspection,
                                         area_choice="area_003", executor=executor)
        result = json.loads(output)
        self.assertEqual(result["area"], "Garage")
        self.assertIn("No light entities in selected area Garage", result["execution"]["reason"])
        self.assertEqual(decider.await_count, 2)
        executor.assert_not_awaited()

    def test_area_selection_handles_context_with_no_area_metadata(self):
        options, payload = area_request("turn on the lamp", [{"names": "Lamp", "domain": "light"}])
        self.assertEqual(set(options), {"no_area"})
        self.assertEqual(set(payload["questions"]["area"]["criteria"]), {"no_area", "other"})

    def test_selected_domain_filters_entities_and_areas(self):
        entities = parse_entities(INSPECTION) + [
            {"names": "Outlet", "domain": "switch", "areas": "Office, Study"},
            {"names": "Garage outlet", "domain": "switch", "areas": "Garage"},
        ]
        service = {"tool": "assist__HassTurnOff", "domain": "light"}
        options, payload = decision_request("turn off the office lights", entities, service)
        self.assertEqual(set(options), {"entity_001", "entity_002", "area_001", "area_002"})
        self.assertTrue(all(target.get("domain") == "light" for target in options.values()
                            if target["type"] == "entity"))
        areas = [target for target in options.values() if target["type"] == "area"]
        self.assertEqual([area["name"] for area in areas], ["Office", "Bedroom"])
        self.assertTrue(all(area["domains"] == ["light"] for area in areas))
        self.assertEqual(payload["state"]["service"], service)

    def test_other_returns_no_target(self):
        options, _ = decision_request("open a nonexistent garage", parse_entities(INSPECTION))
        data = response("other")
        result = selection_result(data, options)
        self.assertEqual(result["choice"], "other")
        self.assertIsNone(result["target"])
        self.assertEqual(result["probability"], 0.8)

    def test_invalid_decisions_are_errors_instead_of_other(self):
        options, _ = decision_request("lamp", parse_entities(INSPECTION))
        changes = (
            {"choice": "invented_entity"},
            {"choice": ["entity_001"]},
            {"type": "score"},
            {"confidence": True},
            {"confidence": float("nan")},
            {"probabilities": {"entity_001": 1.0}},
            {"probabilities": {**response()["answers"]["target"]["probabilities"], "entity_001": 1.1}},
        )
        for change in changes:
            data = copy.deepcopy(response())
            data["answers"]["target"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                selection_result(data, options)
        for data in ({}, {"error": {"message": "Provider failed"}}, []):
            with self.subTest(data=data), self.assertRaises(ValueError):
                selection_result(data, options)

    def test_cli_reuses_one_mcp_snapshot_for_three_sequential_requests(self):
        output, inspector, decider = self.run_cli([service_response(), response()])
        result = json.loads(output)
        self.assertEqual(result["service"], "assist__HassTurnOff")
        self.assertNotIn("state_info", result)
        self.assertEqual(result["domain"], "light")
        self.assertEqual(result["target"]["type"], "entity")
        self.assertEqual(result["target"]["areas"], "Office, Study")
        self.assertEqual(result["decisions"]["target"]["candidate_count"], 4)
        self.assertEqual(result["usage"], {"service": {"cost": 0.00003}, "area": {"cost": 0.000005}, "target": {"cost": 0.00002}})
        inspector.assert_awaited_once()
        self.assertEqual(decider.await_count, 3)
        service_payload, _ = decider.await_args_list[0].args
        self.assertEqual(set(service_payload["questions"]), {"service", "domain"})
        area_payload, _ = decider.await_args_list[1].args
        self.assertEqual(set(area_payload["questions"]), {"area"})
        payload, _ = decider.await_args_list[2].args
        self.assertEqual(payload["state"]["service"], {"tool": "assist__HassTurnOff", "domain": "light"})
        self.assertEqual(set(payload["questions"]["target"]["criteria"]),
                         {"entity_001", "entity_002", "area_001", "area_002", "other"})

    def test_cli_summary_displays_tool_domain_and_area(self):
        output, _, _ = self.run_cli([service_response(), response("area_001")], json_output=False)
        self.assertEqual(output, 'service: assist__HassTurnOff\ndomain: light\narea: not specified\ntarget type: area\ntarget: Office\nexecution: completed\narguments: {"domain": ["light"], "area": "Office"}\n')

    def test_entity_execution_uses_alias_area_and_domain_once(self):
        executor = AsyncMock(return_value={"status": "completed"})
        output, inspector, _ = self.run_cli([service_response(), response()], executor=executor)
        self.assertEqual(json.loads(output)["execution"]["status"], "completed")
        executor.assert_awaited_once_with(inspector.await_args.args[0], "assist__HassTurnOff", {
            "domain": ["light"], "name": "Desk lamp", "area": "Office",
        })
        inspector.assert_awaited_once()

    def test_area_execution_keeps_domain_filter_and_omits_entity_name(self):
        executor = AsyncMock(return_value={"status": "completed"})
        _, inspector, _ = self.run_cli([service_response(), response("area_001")], executor=executor)
        executor.assert_awaited_once_with(inspector.await_args.args[0], "assist__HassTurnOff", {
            "domain": ["light"], "area": "Office",
        })

    def test_ambiguous_decisions_and_query_state_never_execute(self):
        for first, second in (
            (service_response("other"), None),
            (service_response(domain="other"), None),
            (service_response(), response("other")),
            (service_response("query_state"), response()),
        ):
            executor = AsyncMock()
            responses = [first] + ([second] if second is not None else [])
            self.run_cli(responses, executor=executor)
            executor.assert_not_awaited()

    def test_tools_needing_extra_values_are_skipped(self):
        inspection = copy.deepcopy(INSPECTION)
        inspection["tools"][1]["inputSchema"]["properties"]["brightness"] = {"type": "integer"}
        executor = AsyncMock()
        output, _, _ = self.run_cli([service_response(), response()], inspection=inspection, executor=executor)
        result = json.loads(output)
        self.assertEqual(result["execution"]["status"], "skipped")
        self.assertIn("brightness", result["execution"]["reason"])
        executor.assert_not_awaited()

    def test_temperature_extraction_uses_constants_and_existing_target_before_execution(self):
        inspection, responses = numeric_case()
        executor = AsyncMock(return_value={"status": "completed"})
        output, inspector, decider = self.run_cli(
            responses, inspection=inspection, executor=executor, prompt="make the office warmer",
        )
        result = json.loads(output)
        self.assertEqual(decider.await_count, 4)
        payload, _ = decider.await_args_list[3].args
        self.assertEqual(payload["state"]["current"], 20.5)
        self.assertEqual((payload["state"]["minimum"], payload["state"]["maximum"]), (16, 26))
        self.assertEqual(len(payload["questions"]["numeric_score"]["criteria"]), 9)
        self.assertEqual(result["decisions"]["numeric"]["value"], 21)
        self.assertEqual(result["usage"]["numeric"], {"cost": 0.00004})
        executor.assert_awaited_once_with(inspector.await_args.args[0], "assist__HassClimateSetTemperature", {
            "name": "Device 1", "area": "Office", "temperature": 21,
        })
        inspector.assert_awaited_once()

    def test_absolute_temperature_does_not_need_an_existing_setting(self):
        inspection, responses = numeric_case(currents=(None,), mode="absolute", score=4.8)
        executor = AsyncMock(return_value={"status": "completed"})
        output, _, _ = self.run_cli(responses, inspection=inspection, executor=executor, prompt="set it to 22 degrees")
        self.assertEqual(json.loads(output)["decisions"]["numeric"]["value"], 22)
        self.assertEqual(executor.await_args.args[2]["temperature"], 22)

    def test_temperature_is_quantized_to_whole_degrees_before_mcp_call(self):
        inspection, responses = numeric_case(currents=(20,), score=4.02)
        executor = AsyncMock(return_value={"status": "completed"})
        output, _, _ = self.run_cli(responses, inspection=inspection, executor=executor)
        numeric = json.loads(output)["decisions"]["numeric"]
        self.assertAlmostEqual(numeric["estimate"]["value"], 21.025)
        self.assertEqual(numeric["value"], 21)
        temperature = executor.await_args.args[2]["temperature"]
        self.assertEqual(temperature, 21)
        self.assertIsInstance(temperature, int)

    def test_subdegree_relative_changes_preserve_direction(self):
        for score, expected in ((3.6, 21), (3.44, 21), (4.4, 21), (4.56, 21)):
            current = 20 if score < 4 else 22
            with self.subTest(score=score, current=current):
                inspection, responses = numeric_case(currents=(current,), score=score)
                executor = AsyncMock(return_value={"status": "completed"})
                self.run_cli(responses, inspection=inspection, executor=executor)
                self.assertEqual(executor.await_args.args[2]["temperature"], expected)

    def test_relative_temperature_without_inferred_change_is_skipped(self):
        inspection, responses = numeric_case(currents=(21,), score=4)
        executor = AsyncMock()
        output, _, _ = self.run_cli(responses, inspection=inspection, executor=executor)
        executor.assert_not_awaited()
        self.assertIn("No temperature change inferred", json.loads(output)["execution"]["reason"])

    def test_brightness_uses_percent_baseline_and_sends_an_integer(self):
        inspection, responses = numeric_case("brightness", currents=(51,), mode="absolute", score=3.13)
        executor = AsyncMock(return_value={"status": "completed"})
        output, _, decider = self.run_cli(responses, inspection=inspection, executor=executor, prompt="set brightness to 63 percent")
        payload, _ = decider.await_args_list[3].args
        self.assertEqual(payload["state"]["current"], 20)
        arguments = executor.await_args.args[2]
        self.assertEqual(arguments["brightness"], 63)
        self.assertIsInstance(arguments["brightness"], int)
        self.assertNotIn("temperature", arguments)
        self.assertNotIn("color", arguments)
        self.assertEqual(json.loads(output)["execution"]["status"], "completed")

    def test_current_brightness_prefers_explicit_percent_and_handles_off_or_unavailable(self):
        entities = [
            {"state": "on", "attributes": {"brightness_pct": "70", "brightness": "51"}},
            {"state": "off", "attributes": {"brightness": 100}},
            {"state": "unavailable", "attributes": {"brightness": 100}},
        ]
        self.assertEqual(current_numeric_values(entities, "brightness"), [70, 0, None])

    def test_relative_area_temperature_requires_one_common_target_setting(self):
        for currents, should_execute in (((20.5, 20.5), True), ((20.5, 22), False), ((None,), False)):
            with self.subTest(currents=currents):
                inspection, responses = numeric_case(currents=currents)
                responses[1]["answers"]["target"]["choice"] = "area_001"
                executor = AsyncMock(return_value={"status": "completed"})
                output, _, _ = self.run_cli(responses, inspection=inspection, executor=executor)
                if should_execute:
                    self.assertEqual(executor.await_args.args[2], {"area": "Office", "temperature": 21})
                else:
                    executor.assert_not_awaited()
                    self.assertEqual(json.loads(output)["execution"]["status"], "skipped")

    def test_absolute_area_temperature_can_replace_different_settings(self):
        inspection, responses = numeric_case(currents=(20, 23), mode="absolute", score=4.8)
        responses[1]["answers"]["target"]["choice"] = "area_001"
        executor = AsyncMock(return_value={"status": "completed"})
        self.run_cli(responses, inspection=inspection, executor=executor)
        self.assertEqual(executor.await_args.args[2], {"area": "Office", "temperature": 22})

    def test_unsupported_numeric_intent_and_invalid_numeric_responses_do_not_execute(self):
        for kind in ("unsupported", "invalid", "schema_limit"):
            with self.subTest(kind=kind):
                inspection, responses = numeric_case(score=4.8, mode="other" if kind == "unsupported" else "absolute")
                if kind == "invalid":
                    responses[2]["answers"]["numeric_score"]["score"] = float("nan")
                if kind == "schema_limit":
                    inspection["tools"][0]["inputSchema"]["properties"]["temperature"]["maximum"] = 21
                executor = AsyncMock()
                output, _, _ = self.run_cli(responses, inspection=inspection, executor=executor)
                executor.assert_not_awaited()
                self.assertEqual(json.loads(output)["execution"]["status"], "skipped")

    def test_numeric_value_is_visible_in_summary(self):
        inspection, responses = numeric_case()
        output, _, _ = self.run_cli(responses, inspection=inspection, json_output=False)
        self.assertIn("temperature: 21 °C\n", output)

    def test_execution_errors_are_reported_and_not_retried(self):
        for failure in (TimeoutError("timed out"), None):
            executor = AsyncMock(side_effect=failure, return_value={"status": "error", "error": "HA rejected the target"})
            output, _, _ = self.run_cli([service_response(), response()], executor=executor, exit_code=1)
            result = json.loads(output)
            self.assertEqual(result["execution"]["status"], "error")
            executor.assert_awaited_once()

    def test_target_arguments_reject_unsupported_and_nonunique_targets(self):
        entities = parse_entities(INSPECTION)
        tool = INSPECTION["tools"][1]
        target = {**entities[0], "type": "entity"}
        with self.assertRaisesRegex(ValueError, "not unique"):
            target_arguments(tool, target, "light", entities + [entities[0]])
        with self.assertRaisesRegex(ValueError, "does not support area"):
            target_arguments({"inputSchema": {"properties": {"name": {}}}}, {"type": "area", "name": "Office"}, "light", entities)
        with self.assertRaisesRegex(ValueError, "does not support named"):
            target_arguments({"inputSchema": {"properties": {"area": {}}}}, target, "light", entities)
        required = copy.deepcopy(tool)
        required["inputSchema"]["required"] = ["floor"]
        with self.assertRaisesRegex(ValueError, "Missing required arguments: floor"):
            target_arguments(required, target, "light", entities)

    def test_execute_mcp_passes_exact_arguments_and_recognizes_errors(self):
        for data in (
            {"isError": False, "content": [{"type": "text", "text": '{"success":true}'}]},
            {"isError": True, "content": [{"type": "text", "text": "No matching device"}]},
            {"isError": False, "content": [{"type": "text", "text": '{"success":false,"error":"Not found"}'}]},
            {"isError": False, "structuredContent": {"response_type": "error"}, "content": []},
            {"isError": False, "content": [{"type": "text", "text": '{"response_type":"action_done","data":{"failed":[{"name":"Lamp"}]}}'}]},
        ):
            response = SimpleNamespace(isError=data["isError"], model_dump=Mock(return_value=data))
            session = SimpleNamespace(call_tool=AsyncMock(return_value=response))
            arguments = {"area": "Office", "domain": ["light"]}
            with self.subTest(data=data):
                result = asyncio.run(execute_mcp(session, "assist__HassTurnOff", arguments))
            self.assertEqual(result["status"], "completed" if data["content"] and data["content"][0]["text"] == '{"success":true}' else "error")
            self.assertEqual(result["response"], data)
            session.call_tool.assert_awaited_once_with("assist__HassTurnOff", arguments)

    def test_uncertain_service_or_domain_skips_target_request(self):
        for tool, domain in (("other", "light"), ("assist__HassTurnOff", "other")):
            with self.subTest(tool=tool, domain=domain):
                output, inspector, decider = self.run_cli([service_response(tool, domain)])
                result = json.loads(output)
                self.assertIsNone(result["target"])
                self.assertIsNone(result["decisions"]["target"])
                self.assertIsNone(result["usage"]["target"])
                inspector.assert_awaited_once()
                decider.assert_awaited_once()

    def test_state_query_selects_target_without_a_service_or_extra_mcp_fetch(self):
        output, inspector, decider = self.run_cli(
            [service_response("query_state"), response()], prompt="is the office desk lamp on?",
        )
        result = json.loads(output)
        self.assertIsNone(result["service"])
        self.assertEqual(result["decisions"]["service"]["choice"], "query_state")
        self.assertEqual(result["domain"], "light")
        self.assertEqual(result["target"]["state"], "off")
        self.assertEqual(result["target"]["type"], "entity")
        self.assertEqual(result["state_info"][0]["state"], "off")
        inspector.assert_awaited_once()
        self.assertEqual(decider.await_count, 3)
        payload, _ = decider.await_args_list[2].args
        self.assertEqual(payload["state"]["service"], {
            "tool": None, "domain": "light", "operation": "query_state",
        })

    def test_temperature_query_preserves_measurement_and_unit(self):
        inspection = copy.deepcopy(INSPECTION)
        inspection["context"] = """- names: Temperature
  domain: sensor
  state: '21.5'
  areas: Office
  attributes:
    device_class: temperature
    unit_of_measurement: °C
"""
        service = service_response("query_state")
        service["answers"]["domain"].update(choice="sensor", probabilities={"sensor": 0.9, "other": 0.1})
        target = response()
        target["answers"]["target"]["probabilities"] = {
            "entity_001": 0.8, "area_001": 0.1, "other": 0.1,
        }
        output, inspector, _ = self.run_cli(
            [service, target], inspection=inspection, prompt="how warm is the office?",
        )
        result = json.loads(output)
        self.assertIsNone(result["service"])
        self.assertEqual(result["domain"], "sensor")
        self.assertEqual(result["target"]["state"], "21.5")
        self.assertEqual(result["target"]["attributes"]["unit_of_measurement"], "°C")
        self.assertEqual(result["state_info"][0]["state"], "21.5")
        inspector.assert_awaited_once()
        output, _, _ = self.run_cli(
            [service, target], inspection=inspection, prompt="how warm is the office?", json_output=False,
        )
        self.assertIn("state: 21.5 °C\n", output)

    def test_state_query_summary_shows_no_service_and_still_selects_an_area(self):
        output, _, _ = self.run_cli(
            [service_response("query_state"), response("area_001")], json_output=False,
            prompt="are all the office lights off?",
        )
        self.assertEqual(output, "service: none (query state)\ndomain: light\narea: not specified\ntarget type: area\ntarget: Office\nDesk lamp, Work light:\n  state: off\n")

    def test_area_state_info_includes_only_matching_room_and_domain(self):
        entities = parse_entities(INSPECTION) + [
            {"names": "Floor lamp", "domain": "light", "areas": " Office, Study ", "state": "on"},
            {"names": "Outlet", "domain": "switch", "areas": "Office, Study", "state": "on"},
            {"names": "No area", "domain": "light", "state": "on"},
        ]
        target = {"type": "area", "name": "Office", "aliases": ["Study"]}
        info = query_state_info(target, entities, "light")
        self.assertEqual([entity["names"] for entity in info], ["Desk lamp, Work light", "Floor lamp"])
        self.assertEqual([entity["state"] for entity in info], ["off", "on"])

    def test_climate_state_displays_current_and_target_measurements(self):
        target = {
            "type": "entity", "names": "Thermostat", "domain": "climate", "state": "heat",
            "attributes": {"current_temperature": 21.5, "temperature": 23, "temperature_unit": "°C"},
        }
        output = io.StringIO()
        with redirect_stdout(output):
            print_result({"service": None, "domain": "climate", "target": target, "state_info": [target]})
        self.assertIn("state: heat\n", output.getvalue())
        self.assertIn("current_temperature: 21.5 °C\n", output.getvalue())
        self.assertIn("temperature: 23 °C\n", output.getvalue())

    def test_unknown_states_do_not_display_stale_measurements(self):
        for state in (None, "unknown", "unavailable"):
            target = {
                "type": "entity", "names": "Temperature", "domain": "sensor", "state": state,
                "attributes": {"unit_of_measurement": "°C", "temperature": 21.5},
            }
            output = io.StringIO()
            with self.subTest(state=state), redirect_stdout(output):
                print_result({"service": None, "domain": "sensor", "target": target, "state_info": [target]})
            self.assertIn(f"state: {state if state is not None else 'unknown'}\n", output.getvalue())
            self.assertNotIn("21.5", output.getvalue())
            self.assertNotIn("°C", output.getvalue())

    def test_state_query_with_unknown_domain_does_not_guess_a_target(self):
        output, inspector, decider = self.run_cli([service_response("query_state", "other")])
        result = json.loads(output)
        self.assertIsNone(result["service"])
        self.assertIsNone(result["target"])
        self.assertEqual(result["state_info"], [])
        decider.assert_awaited_once()
        inspector.assert_awaited_once()

    def test_tool_domain_restriction_rejects_incompatible_decision(self):
        inspection = copy.deepcopy(INSPECTION)
        inspection["tools"][1]["inputSchema"] = {
            "properties": {"domain": {"type": "array", "items": {"enum": ["switch"]}}},
        }
        with self.assertRaisesRegex(ValueError, "light is not allowed"):
            self.run_cli([service_response()], inspection=inspection)

    def test_unmatched_target_is_visible_in_summary(self):
        output, _, _ = self.run_cli([service_response(), response("other")], json_output=False)
        self.assertIn("target type: other\ntarget: other\n", output)

    def test_inspection_does_not_require_or_call_openrouter(self):
        decider = AsyncMock()
        @asynccontextmanager
        async def connection(url, token):
            yield SimpleNamespace()
        with (
            patch.object(sys, "argv", ["prototype.py", "--inspect"]),
            patch("prototype.load_dotenv"),
            patch("prototype.setting", side_effect=lambda name: {"HA_MCP_URL": "test", "HA_TOKEN": "test"}[name]),
            patch("prototype.mcp_session", new=connection),
            patch("prototype.inspect_mcp", new=AsyncMock(return_value=INSPECTION)),
            patch("prototype.call_decider", new=decider),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(asyncio.run(run()), 0)
        decider.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
