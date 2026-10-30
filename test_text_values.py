import copy
import unittest

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.text_values import (
    INSTRUCTIONS,
    MESSAGE_INSTRUCTIONS,
    decode_text,
    text_request,
)
from text_probe import INSTRUCTIONS as PROBE_INSTRUCTIONS
from text_probe import word_request


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
    def test_shopping_request_matches_the_working_probe_exactly(self):
        prompt = "Schreibe Energy Drinks auf die To-Do-Einkaufsliste"
        probe_words, probe_request = word_request(prompt, PROBE_INSTRUCTIONS)
        words, request = text_request(prompt, "todo.add_item", INSTRUCTIONS)
        self.assertEqual(INSTRUCTIONS, PROBE_INSTRUCTIONS)
        self.assertEqual(words, probe_words)
        self.assertEqual(request, {key: value for key, value in probe_request.items() if key != "model"})

    def test_message_actions_send_their_instructions_once_with_short_word_questions(self):
        for service in ("notify.send_message", "assist_satellite.announce"):
            with self.subTest(service=service):
                _, request = text_request("Sage das Essen ist fertig", service, MESSAGE_INSTRUCTIONS)
                self.assertEqual(request["state"]["task"], MESSAGE_INSTRUCTIONS)
                self.assertEqual(str(request).count(MESSAGE_INSTRUCTIONS), 1)
                for question in request["questions"].values():
                    self.assertTrue(question["instructions"].startswith("Apply state.task"))
                    self.assertEqual(question["criteria"]["keep"], "This word belongs to the requested message.")

    def test_only_keep_above_threshold_survives_without_reordering_or_rewriting(self):
        words = ["Add", "two", "Energy", "Energy", "Drinks!", "please"]
        response = response_for([("drop", 0.99), ("keep", 0.9), ("keep", 0.8),
                                 ("keep", 0.7), ("keep", 0.6), ("keep", 0.5)])
        self.assertEqual(decode_text(response, words, 0.5), "two Energy Energy Drinks!")
        self.assertEqual(decode_text(response, words, 0.7), "two Energy")
        self.assertEqual(decode_text(response, words, 1), "")

    def test_all_drop_returns_empty_even_at_zero_threshold(self):
        self.assertEqual(decode_text(response_for([("drop", 1)]), ["nothing"], 0), "")

    def test_uncertain_words_can_be_kept_with_the_switch(self):
        words = ["Energy", "Drinks", "please"]
        response = response_for([("keep", 0.9), ("keep", 0.5), ("drop", 0.2)])
        self.assertEqual(decode_text(response, words, 0.5), "Energy")
        self.assertEqual(decode_text(response, words, 0.5, keep_uncertain=True), "Energy Drinks please")

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
