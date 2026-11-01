import copy
import unittest

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.optional_parameters import (
    INSTRUCTIONS,
    action_fields,
    decode_parameters,
    parameter_request,
)


def response(probabilities, confidence=0):
    return {"answers": {"optional_parameters": {
        "type": "choice", "choice": max(probabilities, key=probabilities.get),
        "confidence": confidence, "probabilities": probabilities,
    }}}


class OptionalParameterTests(unittest.TestCase):
    def test_nested_groups_are_flattened_but_object_selectors_are_preserved(self):
        fields = {"temperature": {"required": True, "selector": {"number": {"min": 5}}},
                  "advanced": {"collapsed": True, "fields": {
                      "hvac_mode": {"selector": {"select": {"options": ["heat", "off"]}}},
                      "data": {"selector": {"object": {"fields": {"key": {}}}}},
                  }}}
        original = copy.deepcopy(fields)
        flattened = action_fields(fields)
        optional, payload = parameter_request("Set it warmer", "climate.set_temperature", flattened, INSTRUCTIONS)
        self.assertEqual(set(optional), {"hvac_mode", "data"})
        self.assertEqual(payload["state"]["required_fields"], {"temperature": fields["temperature"]})
        question = payload["questions"]["optional_parameters"]
        self.assertEqual(set(question["criteria"]), {"hvac_mode", "data", "none"})
        self.assertNotIn("unsure", question["criteria"])
        self.assertIn('"object"', question["criteria"]["data"])
        self.assertEqual(fields, original)

    def test_multiple_parameters_are_selected_without_using_overall_confidence_or_none_vote(self):
        result = decode_parameters(response({"brightness": 0.25, "hue": 0.2, "child_lock": 0.05, "none": 0.5}),
                                   {"brightness": {}, "hue": {}, "child_lock": {}}, 0.15)
        self.assertEqual(result["selected_parameters"], ["brightness", "hue"])
        self.assertEqual(result["probabilities"]["none"], 0.5)
        self.assertEqual(result["probability_threshold"], 0.15)

    def test_threshold_is_strict_and_none_is_never_a_parameter(self):
        optional = {"brightness": {}, "hue": {}}
        probabilities = {"brightness": 0.15, "hue": 0, "none": 0.85}
        self.assertEqual(decode_parameters(response(probabilities), optional, 0.15)["selected_parameters"], [])
        self.assertEqual(decode_parameters(response(probabilities), optional, 0)["selected_parameters"], ["brightness"])
        self.assertEqual(decode_parameters(response(probabilities), optional, 1)["selected_parameters"], [])

    def test_malformed_responses_are_rejected(self):
        valid = response({"brightness": 0.7, "none": 0.3})
        invalid = [None, {}, {"answers": {}}]
        for changes in ({"choice": "invented"}, {"type": "score"}, {"confidence": True},
                        {"confidence": float("nan")}, {"probabilities": {"brightness": 1}},
                        {"probabilities": {"brightness": -0.1, "none": 1.1}}):
            item = copy.deepcopy(valid)
            item["answers"]["optional_parameters"].update(changes)
            invalid.append(item)
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ModelError):
                decode_parameters(item, {"brightness": {}}, 0.15)
