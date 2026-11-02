import copy
import json
import unittest

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.settings import normalize_settings
from custom_components.glow_assistant.text_values import (
    INSTRUCTIONS,
    MESSAGE_INSTRUCTIONS,
    decode_text,
    text_request,
)


def response_for(probabilities, confidence=0):
    return {"answers": {"text_words": {"type": "choice", "choice": max(probabilities, key=probabilities.get),
                                      "confidence": confidence, "probabilities": probabilities}}}


class TextValuesTests(unittest.TestCase):
    def test_one_choice_contains_each_word_occurrence_with_no_catalog_duplication(self):
        words, request = text_request("Add milk milk to Milk", "todo.add_item", INSTRUCTIONS)
        self.assertEqual(list(words.values()), [
            {"word": "Add", "position": 1}, {"word": "milk", "position": 2},
            {"word": "milk", "position": 3}, {"word": "to", "position": 4}, {"word": "Milk", "position": 5},
        ])
        self.assertEqual(set(request["questions"]), {"text_words"})
        self.assertEqual(set(request["questions"]["text_words"]["criteria"]), {*words, "none"})
        self.assertNotIn("words", request["state"])
        self.assertEqual(str(request).count(INSTRUCTIONS), 1)
        self.assertEqual(request["state"]["parameter"], "item")
        self.assertEqual(request["state"]["other_parameters"], [])

    def test_default_text_threshold_excludes_command_words_from_the_reported_distribution(self):
        prompt = "setze zwiebeln auf die einkaufsliste."
        words, _ = text_request(prompt, "todo.add_item", INSTRUCTIONS)
        probabilities = dict(zip([*words, "none"], [0.00832102473981359, 0.9511921715845962, 0.007895197376288885,
                                                   0.0032521662762517627, 0.0030805915038456546, 0.026258848519203994], strict=True))
        threshold = normalize_settings({})["decider"]["text_probability_threshold"]
        selection = decode_text(response_for(probabilities), words, threshold)
        self.assertEqual(selection["value"], "zwiebeln")
        self.assertEqual(selection["probability_threshold"], 0.4)
        self.assertEqual(selection["effective_probability_threshold"], 0.08)

    def test_message_actions_use_their_own_prompt(self):
        for service in ("notify.send_message", "assist_satellite.announce"):
            with self.subTest(service=service):
                _, request = text_request("Sage das Essen ist fertig", service, MESSAGE_INSTRUCTIONS)
                self.assertEqual(request["state"]["parameter"], "message")
                self.assertEqual(str(request).count(MESSAGE_INSTRUCTIONS), 1)

    def test_preserves_word_order_duplicates_punctuation_and_case_ignoring_confidence_and_none(self):
        words, _ = text_request("Sage Hi, Hi! bitte", "assist_satellite.announce", MESSAGE_INSTRUCTIONS)
        probabilities = {"none": 0.5, "word:2:Hi!": 0.2, "word:1:Hi,": 0.2, "word:3:bitte": 0.05, "word:0:Sage": 0.05}
        selection = decode_text(response_for(probabilities), words, 0.3)
        self.assertEqual(selection["value"], "Hi, Hi!")
        self.assertEqual(selection["selected_choices"], ["word:1:Hi,", "word:2:Hi!"])
        self.assertEqual(selection["probabilities"], probabilities)
        self.assertEqual(selection["real_option_count"], 4)
        self.assertAlmostEqual(selection["effective_probability_threshold"], 0.075)

    def test_threshold_is_inclusive_and_none_can_produce_empty_text(self):
        words, _ = text_request("Add milk", "todo.add_item", INSTRUCTIONS)
        response = response_for({"word:0:Add": 0, "word:1:milk": 0.099, "none": 0.901})
        self.assertEqual(decode_text(response, words, 0.2)["value"], "")
        response["answers"]["text_words"]["probabilities"].update({"word:1:milk": 0.1, "none": 0.9})
        self.assertEqual(decode_text(response, words, 0.2)["value"], "milk")
        response["answers"]["text_words"]["probabilities"].update({"word:1:milk": 0, "none": 1})
        self.assertEqual(decode_text(response, words, 0)["value"], "Add milk")

    def test_malformed_missing_extra_and_invalid_probabilities_are_rejected(self):
        words, _ = text_request("Add milk", "todo.add_item", INSTRUCTIONS)
        valid = response_for({"word:0:Add": 0.1, "word:1:milk": 0.8, "none": 0.1})
        cases = [None, {}, {"answers": []}]
        for key, value in (("type", "score"), ("choice", "invented"), ("confidence", float("nan")),
                           ("probabilities", {"word:1:milk": 1}), ("probabilities", {**dict.fromkeys([*words, "none"], 0), "extra": 1})):
            response = copy.deepcopy(valid)
            response["answers"]["text_words"][key] = value
            cases.append(response)
        for value in (-0.1, 1.1, True, float("nan")):
            response = copy.deepcopy(valid)
            response["answers"]["text_words"]["probabilities"]["word:1:milk"] = value
            cases.append(response)
        extra = copy.deepcopy(valid)
        extra["answers"]["other"] = {}
        cases.append(extra)
        for response in cases:
            with self.subTest(response=response), self.assertRaisesRegex(ModelError, "Malformed text extraction response"):
                decode_text(response, words, 0.02)


class TextEngineTests(unittest.IsolatedAsyncioTestCase):
    async def run_text(self, service, prompt, selected_positions, *, optional=False, settings=None, history=None):
        from test_integration_engine import ScriptedModels, entity

        parameter = "item" if service == "todo.add_item" else "message"
        domain = service.split(".")[0]
        target_id = f"{domain}.shopping" if parameter == "item" else f"{domain}.kitchen"
        snapshot = {
            "entities": [entity(target_id, "Shopping" if parameter == "item" else "Kitchen", area=None)],
            "services": {service: {"target": {}, "fields": {parameter: {"required": not optional, "selector": {"text": {}}}}}},
        }
        words, _ = text_request(prompt, service, INSTRUCTIONS)
        probabilities = {key: (0.9 / len(selected_positions) if word["position"] in selected_positions else 0)
                         for key, word in words.items()}
        probabilities["none"] = 0.1 if selected_positions else 1
        choices = [{"service": service}, {"target_scope": "entities"}, {"action_entities": f"entity:{target_id}"}]
        if optional:
            choices.append({"optional_parameters": parameter})
        choices.append({"text_words": response_for(probabilities)["answers"]["text_words"]})
        client = ScriptedModels(choices)
        events = []
        engine = AgentEngine(settings or normalize_settings({}), client, trace=lambda *event: events.append(event), history=history, dry_run=True)
        result = await engine.process(prompt, "de", snapshot, speaker_area="Office")
        return result, client, events

    async def test_all_three_supported_text_actions_extract_after_target_and_parameter_selection(self):
        cases = (
            ("todo.add_item", "Schreibe Energy Drinks auf die Einkaufsliste", [2, 3], "Energy Drinks", False),
            ("notify.send_message", "Send Dinner is ready to Kitchen", [2, 3, 4], "Dinner is ready", True),
            ("assist_satellite.announce", "Sage in der Küche das Essen ist fertig", [5, 6, 7, 8], "das Essen ist fertig", False),
        )
        for service, prompt, positions, expected, optional in cases:
            with self.subTest(service=service):
                result, client, events = await self.run_text(service, prompt, positions, optional=optional)
                parameter = "item" if service == "todo.add_item" else "message"
                self.assertEqual(result["parameters"], {parameter: expected})
                self.assertEqual(result["execution"]["status"], "planned")
                self.assertFalse(client.chat_requests)
                questions = [set(request[2]) for request in client.requests]
                self.assertEqual(questions, [{"service"}, {"target_scope"}, {"action_entities"}]
                                 + ([{"optional_parameters"}] if optional else []) + [{"text_words"}])
                self.assertEqual(client.requests[-1][1]["target"], result["target"])
                text_output = next(output for stage, _, output in events if stage.startswith("text:") and output)
                self.assertEqual(text_output["value"], expected)
                self.assertEqual(text_output["probability_threshold"], 0.4)
                self.assertIn("probabilities", text_output["decisions"]["text_words"])

    async def test_language_history_and_custom_prompt_reach_text_step_without_leaking_instructions_to_debug(self):
        config = normalize_settings({"history_enabled": True, "decider": {"text_prompt": "Custom item rules"}})
        history = [{"role": "user", "content": "Use the shopping list"}, {"role": "assistant", "content": "Okay"}]
        result, client, events = await self.run_text("todo.add_item", "Add milk", [2], settings=config, history=history)
        state = client.requests[-1][1]
        self.assertEqual(state["language"], "de")
        self.assertEqual(state["history"], history)
        self.assertIn("Custom item rules", client.requests[-1][2]["text_words"]["instructions"])
        self.assertNotIn("Custom item rules", json.dumps(events))
        self.assertEqual(result["parameters"], {"item": "milk"})

    async def test_empty_selection_stops_with_visible_failure_and_no_parameter_value(self):
        result, client, events = await self.run_text("todo.add_item", "Add to Shopping", [])
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["answer"], "ich bin mir nicht sicher")
        self.assertNotIn("parameters", result)
        self.assertEqual(result["failures"][0]["stage"], "text:item")
        self.assertEqual(result["failures"][0]["device_command"], "not_sent")
        self.assertFalse(client.chat_requests)
        self.assertEqual(events[-1][0], "failure")

    async def test_malformed_word_choices_fail_without_retry_or_answer_call(self):
        from test_integration_engine import ScriptedModels, entity

        choices = [{"service": "todo.add_item"}, {"target_scope": "entities"},
                   {"action_entities": "entity:todo.shopping"}, {"text_words": {"type": "choice", "probabilities": {"none": 1}}}]
        client = ScriptedModels(choices)
        events = []
        snapshot = {"entities": [entity("todo.shopping", "Shopping", area=None)],
                    "services": {"todo.add_item": {"fields": {"item": {"required": True, "selector": {"text": {}}}}}}}
        with self.assertRaisesRegex(ModelError, "Malformed text extraction response"):
            await AgentEngine(normalize_settings({}), client, trace=lambda *event: events.append(event), dry_run=True).process("Add milk", "en", snapshot)
        self.assertEqual(len(client.requests), 4)
        self.assertFalse(client.chat_requests)
        self.assertEqual(events[-1][2]["stage"], "text:item")

    async def test_optional_message_not_selected_skips_extraction(self):
        from test_integration_engine import ScriptedModels, entity

        client = ScriptedModels([{"service": "notify.send_message"}, {"target_scope": "entities"},
                                 {"action_entities": "entity:notify.kitchen"}, {"optional_parameters": "none"}])
        snapshot = {"entities": [entity("notify.kitchen", "Kitchen", area=None)],
                    "services": {"notify.send_message": {"fields": {"message": {"selector": {"text": {}}}}}}}
        result = await AgentEngine(normalize_settings({}), client, dry_run=True).process("Notify Kitchen", "en", snapshot)
        self.assertNotIn("parameter_extractions", result)
        self.assertNotIn("parameters", result)
        self.assertEqual(len(client.requests), 4)


if __name__ == "__main__":
    unittest.main()
