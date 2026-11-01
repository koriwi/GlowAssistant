import copy
import unittest

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.optional_parameters import (
    INSTRUCTIONS,
    action_fields,
    decode_parameters,
    filter_fields,
    parameter_request,
)


def response(probabilities, confidence=0):
    return {"answers": {"optional_parameters": {
        "type": "choice", "choice": max(probabilities, key=probabilities.get),
        "confidence": confidence, "probabilities": probabilities,
    }}}


class OptionalParameterTests(unittest.TestCase):
    def test_thermostat_field_filters_retain_fields_supported_by_any_target(self):
        fields = action_fields({
            "temperature": {"filter": {"supported_features": [1]}},
            "temperature_range": {"fields": {
                "target_temp_low": {"filter": {"supported_features": [2]}},
                "target_temp_high": {"filter": {"supported_features": [2]}},
            }},
            "hvac_mode": {"selector": {"select": {"options": ["heat", "off"]}}},
        })
        single = {"entity_id": "climate.single", "attributes": {"supported_features": 401}}
        ranged = {"entity_id": "climate.range", "attributes": {"supported_features": 2}}
        original = copy.deepcopy((fields, single, ranged))
        for targets, expected in (([single], {"temperature", "hvac_mode"}),
                                  ([ranged], {"target_temp_low", "target_temp_high", "hvac_mode"}),
                                  ([single, ranged], {"temperature", "target_temp_low", "target_temp_high", "hvac_mode"})):
            with self.subTest(targets=targets):
                kept, report = filter_fields(fields, targets)
                self.assertEqual(set(kept), expected)
                self.assertEqual(report["fields_before"], 4)
                self.assertEqual(report["fields_after"], len(expected))
                self.assertEqual(set(report["removed_fields"]), set(fields) - expected)
                self.assertEqual(report["policy"], "any_resolved_target")
        _, report = filter_fields(fields, [single, ranged])
        self.assertEqual(report["field_support"]["temperature"]["unsupported_entity_ids"], ["climate.range"])
        self.assertEqual(report["field_support"]["target_temp_low"]["unsupported_entity_ids"], ["climate.single"])
        self.assertEqual((fields, single, ranged), original)

    def test_attribute_filters_keep_partially_supported_fields_and_record_coverage(self):
        fields = {
            "color": {"filter": {"attribute": {"supported_color_modes": ["hs", "rgb"]}}},
            "preset": {"filter": {"attribute": {"device_class": ["speaker"]}}},
            "plain": {},
        }
        first = {"entity_id": "light.first", "attributes": {"supported_color_modes": ["onoff", "hs"], "device_class": "speaker"}}
        second = {"entity_id": "light.second", "attributes": {"supported_color_modes": ["rgb"], "device_class": "other"}}
        kept, report = filter_fields(fields, [first, second])
        self.assertEqual(set(kept), {"color", "preset", "plain"})
        self.assertEqual(report["field_support"]["preset"]["unsupported_entity_ids"], ["light.second"])
        second["attributes"]["supported_color_modes"] = ["onoff"]
        kept, report = filter_fields(fields, [first, second])
        self.assertEqual(set(kept), {"color", "preset", "plain"})
        self.assertEqual(report["field_support"]["color"]["supported_entity_ids"], ["light.first"])
        first["attributes"]["supported_color_modes"] = ["onoff"]
        kept, report = filter_fields(fields, [first, second])
        self.assertNotIn("color", kept)
        self.assertEqual(report["removed_fields"]["color"]["unsupported_entity_ids"], ["light.first", "light.second"])

    def test_feature_and_attribute_alternatives_match_ha_filter_rules(self):
        fields = {"value": {"filter": {"supported_features": [1, 2], "attribute": {"mode": ["supported"], "other": ["ok"]}}}}
        for attributes in ({"supported_features": 2}, {"mode": "supported"}, {"other": "ok"}):
            kept, _ = filter_fields(fields, [{"entity_id": "test.target", "attributes": attributes}])
            self.assertEqual(set(kept), {"value"})

    def test_unknown_capabilities_and_unrecognized_filters_do_not_claim_support(self):
        fields = {"feature": {"filter": {"supported_features": [1]}},
                  "attribute": {"filter": {"attribute": {"supported_color_modes": ["hs"]}}},
                  "unresolved": {"filter": {"supported_features": ["climate.ClimateEntityFeature.TARGET_TEMPERATURE"]}},
                  "unknown_rule": {"filter": {"future_rule": True}}, "plain": {}}
        for attributes in ({}, {"supported_features": -1}, {"supported_features": True}):
            kept, report = filter_fields(fields, [{"entity_id": "climate.target", "attributes": attributes}])
            self.assertEqual(set(kept), {"plain"})
            self.assertEqual(len(report["removed_fields"]), 4)

    def test_unfiltered_required_fields_remain_required_and_filtered_fields_are_removed(self):
        fields = {"item": {"required": True}, "feature": {"required": True, "filter": {"supported_features": [1]}}}
        kept, _ = filter_fields(fields, [{"entity_id": "todo.shopping", "attributes": {}}])
        optional, payload = parameter_request("Add milk", "todo.add_item", kept, INSTRUCTIONS)
        self.assertEqual(optional, {})
        self.assertEqual(payload["state"]["required_fields"], {"item": {"required": True}})

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
        result = decode_parameters(response({"brightness": 0.29, "hue": 0.2, "child_lock": 0.01, "none": 0.5}),
                                   {"brightness": {}, "hue": {}, "child_lock": {}}, 0.15)
        self.assertEqual(result["selected_parameters"], ["brightness", "hue"])
        self.assertEqual(result["probabilities"]["none"], 0.5)
        self.assertEqual(result["probability_threshold"], 0.15)

    def test_three_parameters_use_relative_cutoff_excluding_none(self):
        parameters = {"name": {}, "description": {}, "child_lock": {}}
        result = decode_parameters(response({"name": 0.84, "description": 0.16, "child_lock": 0, "none": 0}), parameters, 0.3)
        self.assertEqual(result["selected_parameters"], ["name", "description"])
        self.assertEqual(result["real_option_count"], 3)
        self.assertEqual(result["probability_threshold"], 0.3)
        self.assertAlmostEqual(result["effective_probability_threshold"], 0.1)
        result = decode_parameters(response({"name": 0.901, "description": 0.099, "child_lock": 0, "none": 0}), parameters, 0.3)
        self.assertEqual(result["selected_parameters"], ["name"])

    def test_threshold_is_inclusive_and_none_is_never_a_parameter(self):
        optional = {"brightness": {}, "hue": {}}
        probabilities = {"brightness": 0.075, "hue": 0, "none": 0.925}
        self.assertEqual(decode_parameters(response(probabilities), optional, 0.15)["selected_parameters"], ["brightness"])
        self.assertEqual(decode_parameters(response(probabilities), optional, 0)["selected_parameters"], ["brightness", "hue"])
        self.assertEqual(decode_parameters(response(probabilities), optional, 1)["selected_parameters"], [])
        self.assertEqual(decode_parameters(response({"brightness": 0.5, "hue": 0.5, "none": 0}), optional, 1)["selected_parameters"],
                         ["brightness", "hue"])

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
