import copy
import json
import unittest

import local
from custom_components.glow_assistant.engine import (
    AgentEngine,
    filter_entity_attributes,
)
from custom_components.glow_assistant.settings import (
    normalize_settings,
    validate_settings,
)
from tests.test_integration_engine import ScriptedModels, entity


def snapshot():
    return {"services": {}, "entities": [entity(
        "media_player.kitchen", "Kitchen speaker", area="kitchen", state="playing",
        exact="é" * 1024, oversized="x" * 1025, playlist=["x" * 1024],
        nested={"track": "x" * 1024}, volume_level=0.4, media_title="Song",
    )]}


class EntityAttributeTests(unittest.IsolatedAsyncioTestCase):
    def test_limit_uses_characters_and_whole_json_values_without_mutating_snapshot(self):
        original = snapshot()
        before = copy.deepcopy(original)
        filtered = filter_entity_attributes(original["entities"], 1024)
        attributes = filtered[0]["attributes"]
        self.assertEqual(set(attributes), {"exact", "volume_level", "media_title"})
        self.assertEqual(len(attributes["exact"]), 1024)
        self.assertEqual(filtered[0]["state"], "playing")
        self.assertEqual(original, before)
        self.assertIsNot(filtered[0]["attributes"], original["entities"][0]["attributes"])

    async def test_yes_no_model_receives_filtered_attributes(self):
        original = snapshot()
        models = ScriptedModels([
            {"service": "query_state"}, {"area": "area:kitchen"}, {"domain": "media_player"},
            {"query_type": "yes_no"}, {"binary_answer": "yes"},
        ])
        result = await AgentEngine(normalize_settings({}), models).process("Is music playing in the kitchen?", "en", original)
        candidates = models.requests[-1][1]["entities"]
        self.assertEqual(result["answer"], "Yes.")
        self.assertEqual(set(candidates[0]["attributes"]), {"exact", "volume_level", "media_title"})
        self.assertEqual(len(original["entities"][0]["attributes"]["oversized"]), 1025)

    async def test_answer_model_and_property_options_use_filtered_attributes(self):
        for llm_answer in (False, True):
            with self.subTest(llm_answer=llm_answer):
                choices = [
                    {"service": "query_state"}, {"area": "area:kitchen"}, {"domain": "media_player"},
                    {"query_type": "complex"}, {"targets": ["media_player.kitchen"]},
                ]
                if not llm_answer:
                    choices.append({"state_property": "attribute:media_title"})
                models = ScriptedModels(choices)
                settings = normalize_settings({"answer": {"enable": llm_answer}})
                result = await AgentEngine(settings, models).process("What is playing?", "en", snapshot())
                self.assertEqual(result["route"], "smarthome")
                self.assertEqual(set(result["state_info"][0]["attributes"]), {"exact", "volume_level", "media_title"})
                if llm_answer:
                    facts = json.loads(models.chat_requests[0][1][-1]["content"])
                    self.assertEqual(set(facts["entities"][0]["attributes"]), {"exact", "volume_level", "media_title"})
                else:
                    properties = models.requests[-1][2]["state_property"]["criteria"]
                    self.assertIn("attribute:exact", properties)
                    self.assertNotIn("attribute:oversized", properties)

    def test_cli_and_saved_settings_override_default_and_require_positive_integer(self):
        settings = normalize_settings({"decider": {"url": "http://local/v1/systemone", "model": "test"}})
        self.assertEqual(settings["decider"]["max_attribute_chars"], 1024)
        settings["decider"]["max_attribute_chars"] = 2048
        args = local.parser().parse_args(["-q", "test"])
        self.assertEqual(local.model_settings(settings, args, {})["decider"]["max_attribute_chars"], 2048)
        args = local.parser().parse_args(["-q", "test", "--max-attribute-chars", "4096"])
        self.assertEqual(local.model_settings(settings, args, {})["decider"]["max_attribute_chars"], 4096)
        for invalid in (0, -1, 1.5, float("nan"), float("inf"), True):
            settings["decider"]["max_attribute_chars"] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                validate_settings(settings)
