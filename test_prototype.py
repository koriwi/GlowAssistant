import asyncio
import copy
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

from prototype import decision_request, parse_entities, run, selection_result


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
INSPECTION = {"context": json.dumps({"success": True, "result": CONTEXT})}


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


class PrototypeTests(unittest.TestCase):
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
        self.assertEqual(result["usage"]["cost"], 0.00002)

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

    def test_cli_connects_context_to_one_decision_request(self):
        decider = AsyncMock(return_value=response())
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["prototype.py", "turn on the office desk lamp"]),
            patch("prototype.load_dotenv"),
            patch("prototype.setting", return_value="test-credential"),
            patch("prototype.inspect_mcp", new=AsyncMock(return_value=INSPECTION)),
            patch("prototype.call_decider", new=decider),
            redirect_stdout(output),
        ):
            self.assertEqual(asyncio.run(run()), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["target"]["type"], "entity")
        self.assertEqual(result["target"]["areas"], "Office, Study")
        self.assertEqual(result["candidate_count"], 4)
        decider.assert_awaited_once()
        payload, _ = decider.await_args.args
        self.assertEqual(set(payload["questions"]["target"]["criteria"]),
                         {"entity_001", "entity_002", "area_001", "area_002", "other"})

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
