import copy
import unittest

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.text_values import decode_text


def response_for(choices):
    return {"answers": {
        f"word_{index}": {
            "type": "choice", "choice": choice, "confidence": confidence,
            "probabilities": {"keep": 0.9 if choice == "keep" else 0.1,
                              "drop": 0.1 if choice == "keep" else 0.9},
        }
        for index, (choice, confidence) in enumerate(choices)
    }}


class TextValuesTests(unittest.TestCase):
    def test_only_keep_above_threshold_survives_without_reordering_or_rewriting(self):
        words = ["Add", "two", "Energy", "Energy", "Drinks!", "please"]
        response = response_for([("drop", 0.99), ("keep", 0.9), ("keep", 0.8),
                                 ("keep", 0.7), ("keep", 0.6), ("keep", 0.5)])
        self.assertEqual(decode_text(response, words, 0.5), "two Energy Energy Drinks!")
        self.assertEqual(decode_text(response, words, 0.7), "two Energy")
        self.assertEqual(decode_text(response, words, 1), "")

    def test_all_drop_returns_empty_even_at_zero_threshold(self):
        self.assertEqual(decode_text(response_for([("drop", 1)]), ["nothing"], 0), "")

    def test_partial_or_extra_word_decisions_are_rejected(self):
        for response in (None, {}, {"answers": {}}, response_for([("keep", 1), ("keep", 1)])):
            with self.subTest(response=response), self.assertRaises(ModelError):
                decode_text(response, ["item"], 0.5)

    def test_invalid_word_decisions_are_rejected_before_returning_any_text(self):
        valid = response_for([("keep", 1), ("keep", 1)])
        for change in (
            {"type": "score"}, {"choice": "unsure"}, {"confidence": True},
            {"confidence": float("nan")}, {"confidence": 1.1},
            {"probabilities": {"keep": 1}}, {"probabilities": {"keep": 1, "drop": -1}},
        ):
            response = copy.deepcopy(valid)
            response["answers"]["word_1"].update(change)
            with self.subTest(change=change), self.assertRaises(ModelError):
                decode_text(response, ["Energy", "Drinks"], 0.5)
