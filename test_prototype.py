import asyncio
import copy
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

from prototype import decision_request, parse_entities, print_result, query_state_info, run, selection_result, service_request


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
INSPECTION = {
    "context": json.dumps({"success": True, "result": CONTEXT}),
    "tools": [
        {"name": "assist__HassTurnOn", "description": "Turn on a device", "inputSchema": {}},
        {"name": "assist__HassTurnOff", "description": "Turn off a device", "inputSchema": {}},
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


class PrototypeTests(unittest.TestCase):
    def run_cli(self, responses, inspection=INSPECTION, json_output=True,
                prompt="turn off the lights in the office"):
        inspector = AsyncMock(return_value=inspection)
        decider = AsyncMock(side_effect=responses)
        output = io.StringIO()
        argv = ["prototype.py", prompt]
        if json_output:
            argv.append("--json")
        with (
            patch.object(sys, "argv", argv),
            patch("prototype.load_dotenv"),
            patch("prototype.setting", return_value="test-credential"),
            patch("prototype.inspect_mcp", new=inspector),
            patch("prototype.call_decider", new=decider),
            redirect_stdout(output),
        ):
            self.assertEqual(asyncio.run(run()), 0)
        return output.getvalue(), inspector, decider

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
        self.assertEqual(payload["state"]["entities"], entities)
        self.assertEqual(set(payload["questions"]["service"]["criteria"]), set(tools) | {"other"})
        services, _, _ = service_request("is the lamp on?", {"tools": []}, entities)
        self.assertEqual(set(services), {"query_state"})

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

    def test_cli_reuses_one_mcp_snapshot_for_two_sequential_requests(self):
        output, inspector, decider = self.run_cli([service_response(), response()])
        result = json.loads(output)
        self.assertEqual(result["service"], "assist__HassTurnOff")
        self.assertNotIn("state_info", result)
        self.assertEqual(result["domain"], "light")
        self.assertEqual(result["target"]["type"], "entity")
        self.assertEqual(result["target"]["areas"], "Office, Study")
        self.assertEqual(result["decisions"]["target"]["candidate_count"], 4)
        self.assertEqual(result["usage"], {"service": {"cost": 0.00003}, "target": {"cost": 0.00002}})
        inspector.assert_awaited_once()
        self.assertEqual(decider.await_count, 2)
        service_payload, _ = decider.await_args_list[0].args
        self.assertEqual(set(service_payload["questions"]), {"service", "domain"})
        payload, _ = decider.await_args_list[1].args
        self.assertEqual(payload["state"]["service"], {"tool": "assist__HassTurnOff", "domain": "light"})
        self.assertEqual(set(payload["questions"]["target"]["criteria"]),
                         {"entity_001", "entity_002", "area_001", "area_002", "other"})

    def test_cli_summary_displays_tool_domain_and_area(self):
        output, _, _ = self.run_cli([service_response(), response("area_001")], json_output=False)
        self.assertEqual(output, "service: assist__HassTurnOff\ndomain: light\ntarget type: area\ntarget: Office\n")

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
        self.assertEqual(decider.await_count, 2)
        payload, _ = decider.await_args_list[1].args
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
        self.assertEqual(output, "service: none (query state)\ndomain: light\ntarget type: area\ntarget: Office\nDesk lamp, Work light:\n  state: off\n")

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
        with (
            patch.object(sys, "argv", ["prototype.py", "--inspect"]),
            patch("prototype.load_dotenv"),
            patch("prototype.setting", side_effect=lambda name: {"HA_MCP_URL": "test", "HA_TOKEN": "test"}[name]),
            patch("prototype.inspect_mcp", new=AsyncMock(return_value=INSPECTION)),
            patch("prototype.call_decider", new=decider),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(asyncio.run(run()), 0)
        decider.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
