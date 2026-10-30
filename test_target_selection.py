import copy
import json
import unittest

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.target_selection import (
    decode_targets,
    target_request,
)


def response_for(choices):
    return {"answers": {
        f"target_{index}": {
            "type": "choice", "choice": choice, "confidence": confidence,
            "probabilities": {"keep": 0.9 if choice == "keep" else 0.1,
                              "drop": 0.1 if choice == "keep" else 0.9},
        }
        for index, (choice, confidence) in enumerate(choices)
    }}


class TargetSelectionTests(unittest.TestCase):
    def test_task_is_sent_once_with_one_keep_drop_question_per_entity(self):
        targets = [{"entity_id": "light.desk", "name": "Desk"}, {"entity_id": "light.ceiling", "name": "Ceiling"}]
        request = target_request({"prompt": "Turn off the lights", "service": "light.turn_off"}, targets, "Custom target rules")
        self.assertEqual(request["state"]["targets"], targets)
        self.assertEqual(json.dumps(request).count("Custom target rules"), 1)
        self.assertEqual(set(request["questions"]), {"target_0", "target_1"})
        self.assertTrue(all(set(question["criteria"]) == {"keep", "drop"} for question in request["questions"].values()))

    def test_queries_keep_uncertain_decisions_while_actions_require_confident_keep(self):
        targets = [{"entity_id": f"light.{index}"} for index in range(4)]
        response = response_for([("keep", 0.9), ("keep", 0.2), ("drop", 0.5), ("drop", 0.9)])
        query = decode_targets(response, targets, 0.5, query=True)
        action = decode_targets(response, targets, 0.5, query=False)
        self.assertEqual(query["selected_entity_ids"], ["light.0", "light.1", "light.2"])
        self.assertEqual(action["selected_entity_ids"], ["light.0"])
        self.assertEqual(query["entities_before"], 4)
        self.assertEqual(query["entities_after"], 3)

    def test_empty_selection_does_not_fall_back_to_all_targets(self):
        response = response_for([("drop", 1)])
        for query in (True, False):
            self.assertEqual(decode_targets(response, [{"entity_id": "light.desk"}], 0.5, query)["selected_entity_ids"], [])

    def test_partial_or_unadvertised_entity_decisions_are_rejected(self):
        for response in (None, {}, {"answers": {}}, response_for([("keep", 1), ("keep", 1)])):
            with self.subTest(response=response), self.assertRaises(ModelError):
                decode_targets(response, [{"entity_id": "light.desk"}], 0.5, query=False)

    def test_invalid_entity_decisions_are_rejected_before_any_selection_is_returned(self):
        valid = response_for([("keep", 1)])
        for change in (
            {"choice": "unsure"}, {"type": "score"}, {"confidence": True},
            {"confidence": float("nan")}, {"confidence": 1.1},
            {"probabilities": {"keep": 1}}, {"probabilities": {"keep": -1, "drop": 2}},
        ):
            response = copy.deepcopy(valid)
            response["answers"]["target_0"].update(change)
            with self.subTest(change=change), self.assertRaises(ModelError):
                decode_targets(response, [{"entity_id": "light.desk"}], 0.5, query=False)
