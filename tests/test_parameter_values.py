import copy
import json
import unittest
from unittest.mock import AsyncMock

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.parameter_values import (
    NUMERIC_INSTRUCTIONS,
    UnsupportedParameter,
    decode_number,
    field_type,
    numeric_request,
)
from custom_components.glow_assistant.select_values import (
    INSTRUCTIONS as SELECT_INSTRUCTIONS,
)
from custom_components.glow_assistant.select_values import decode_select, select_request
from custom_components.glow_assistant.settings import (
    normalize_settings,
    validate_settings,
)
from custom_components.glow_assistant.text_values import text_request
from tests.test_integration_engine import ScriptedModels, entity


def numeric_answers(score, mode="absolute", confidence=0.9):
    probabilities = dict.fromkeys((str(index) for index in range(9)), 0)
    lower = int(score)
    probabilities[str(lower)] = 1 - (score - lower)
    if lower != score:
        probabilities[str(lower + 1)] = score - lower
    return {
        "numeric_mode": {"type": "choice", "choice": mode, "confidence": confidence,
                         "probabilities": {key: 1 if key == mode else 0 for key in ("absolute", "relative", "unsure")}},
        "numeric_score": {"type": "score", "score": score, "confidence": confidence, "probabilities": probabilities},
    }


def word_answers(prompt, positions):
    words, _ = text_request(prompt, "todo.add_item", "test")
    probabilities = {key: 0.9 / len(positions) if value["position"] in positions else 0 for key, value in words.items()}
    probabilities["none"] = 0.1
    return {"text_words": {"type": "choice", "choice": max(probabilities, key=probabilities.get),
                           "confidence": 0, "probabilities": probabilities}}


class ParameterValueTests(unittest.TestCase):
    def test_supported_scalar_selectors_and_unsupported_formats(self):
        for selector in ({"text": {}}, {"text": {"multiline": True}}, {"number": {"min": 0, "max": 100}},
                         {"select": {"options": ["on", "off"]}}):
            self.assertIn(field_type("value", {"selector": selector}), ("text", "number", "select"))
        for selector in (None, {}, {"boolean": {}}, {"date": {}},
                         {"datetime": {}}, {"object": {}}, {"text": []}, {"text": False}, {"text": {"multiple": True}}, {"text": {"type": "email"}},
                         {"number": {}, "text": {}}):
            with self.subTest(selector=selector), self.assertRaisesRegex(UnsupportedParameter, "not yet supported"):
                field_type("value", {"selector": selector})

    def request(self, service="number.set_value", parameter="value", members=None, selector=None):
        field = {"description": "Requested value", "selector": {"number": selector or {"min": 0, "max": 100, "step": 1}}}
        return numeric_request("test", service, parameter, field, {parameter: field},
                               members or [entity("number.setting", "Setting", state="30")], NUMERIC_INSTRUCTIONS)

    def test_absolute_values_round_to_ha_step_including_zero(self):
        payload = self.request(selector={"min": 0, "max": 100, "step": 5})
        for score, expected in ((0, 0), (4.12, 50), (8, 100)):
            with self.subTest(score=score):
                result = decode_number({"answers": numeric_answers(score)}, payload["state"], 0.5)
                self.assertEqual(result["value"], expected)
                self.assertIsInstance(result["value"], int)
        payload = self.request(selector={"min": 0, "max": 1, "step": "any"})
        self.assertAlmostEqual(decode_number({"answers": numeric_answers(2.96)}, payload["state"], 0)["value"], 0.37)

    def test_climate_uses_existing_target_and_whole_degrees_preserving_direction(self):
        members = [entity("climate.office", "Thermostat", temperature=20, current_temperature=23,
                          min_temp=5, max_temp=45, target_temp_step=1)]
        payload = self.request("climate.set_temperature", "temperature", members, {"min": 16, "max": 26, "step": 0.1})
        self.assertEqual(payload["state"]["current"], 20)
        self.assertEqual(payload["state"]["step"], 1)
        self.assertEqual(payload["state"]["default_change"], 0.5)
        self.assertEqual(decode_number({"answers": numeric_answers(3.28, "relative")}, payload["state"], 0)["value"], 21)
        self.assertEqual(decode_number({"answers": numeric_answers(3.12, "relative")}, payload["state"], 0)["value"], 19)

    def test_relative_changes_require_a_known_common_baseline(self):
        for members in ([entity("climate.office", "Thermostat", current_temperature=23)],
                        [entity("climate.office", "One", temperature=20), entity("climate.second", "Two", temperature=22)]):
            payload = self.request("climate.set_temperature", "temperature", members)
            result = decode_number({"answers": numeric_answers(4, "relative")}, payload["state"], 0)
            self.assertIsNone(result["value"])
            self.assertIn("common current", result["reason"])
        unavailable = self.request(members=[entity("number.setting", "Setting", state="unavailable")])
        self.assertIsNone(unavailable["state"]["current"])

    def test_native_relative_brightness_field_does_not_need_device_baseline(self):
        payload = self.request("light.turn_on", "brightness_step_pct", [entity("light.lamp", "Lamp")],
                               {"min": -100, "max": 100, "step": 1})
        self.assertTrue(payload["state"]["native_delta"])
        self.assertEqual(payload["state"]["current"], 0)
        result = decode_number({"answers": numeric_answers(4.4, "relative")}, payload["state"], 0)
        self.assertEqual(result["value"], 10)
        self.assertIsNone(decode_number({"answers": numeric_answers(4.4, "absolute")}, payload["state"], 0)["value"])
        payload = self.request("light.turn_on", "brightness_pct", [entity("light.lamp", "Lamp", brightness=127.5)])
        self.assertEqual(payload["state"]["current"], 50)
        payload = self.request("cover.set_cover_position", "position", [entity("cover.blind", "Blind", current_position=40)])
        self.assertEqual(payload["state"]["current"], 40)
        self.assertEqual(payload["state"]["default_change"], 10)

    def test_missing_range_and_invalid_steps_stop_explicitly(self):
        for selector in ({"min": 0}, {"max": 100}, {"min": 10, "max": 1}, {"min": 0, "max": 100, "step": 0}):
            with self.subTest(selector=selector), self.assertRaisesRegex(UnsupportedParameter, "not yet supported"):
                self.request(selector=selector)

    def test_bad_numeric_responses_are_rejected_and_low_confidence_stops(self):
        payload = self.request()
        valid = {"answers": numeric_answers(4)}
        cases = [None, {}, {"answers": {}}]
        for name, field, value in (("numeric_mode", "choice", "invented"), ("numeric_score", "score", 9),
                                   ("numeric_score", "score", float("nan")), ("numeric_score", "confidence", True),
                                   ("numeric_score", "probabilities", {"4": 1})):
            response = copy.deepcopy(valid)
            response["answers"][name][field] = value
            cases.append(response)
        for response in cases:
            with self.subTest(response=response), self.assertRaisesRegex(ModelError, "Malformed numeric"):
                decode_number(response, payload["state"], 0)
        self.assertIsNone(decode_number(valid, payload["state"], 0.95)["value"])
        self.assertIsNone(decode_number({"answers": numeric_answers(4, "unsure")}, payload["state"], 0)["value"])


class ParameterEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_item_and_description_extract_separately_with_unique_traces_and_history(self):
        prompt = "setze nudeln (weizen, fussilli, barilla) auf die einkaufsliste."
        fields = {"item": {"required": True, "description": "Short name of the item", "selector": {"text": {}}},
                  "description": {"description": "Additional details", "selector": {"text": {"multiline": True}}}}
        client = ScriptedModels([
            {"service": "todo.add_item"}, {"target_scope": "entities"}, {"action_entities": "entity:todo.shopping"},
            {"optional_parameters": {"type": "choice", "choice": "none", "confidence": 0,
                                     "probabilities": {"description": 0.3, "none": 0.7}}},
            word_answers(prompt, [2]), word_answers(prompt, [3, 4, 5]),
        ])
        snapshot = {"entities": [entity("todo.shopping", "Einkaufsliste", area=None)],
                    "services": {"todo.add_item": {"fields": fields}}}
        events = []
        history = [{"role": "user", "content": "Use the shopping list"}]
        settings = normalize_settings({"history_enabled": True, "decider": {"parameter_probability_threshold": 0.15}})
        result = await AgentEngine(settings, client, trace=lambda *event: events.append(event), history=history, dry_run=True).process(prompt, "de", snapshot)
        self.assertEqual(result["parameters"], {"item": "nudeln", "description": "(weizen, fussilli, barilla)"})
        self.assertEqual(set(result["parameter_extractions"]), {"item", "description"})
        for name in fields:
            stage = f"text:{name}"
            self.assertIn(stage, result["timings_ms"])
            self.assertIn(stage, result["decisions"])
            self.assertEqual(result["usage"][stage], {"input_tokens": 5})
        for request in client.requests[-2:]:
            self.assertEqual(request[1]["selected_fields"], fields)
            self.assertEqual(request[1]["other_parameters"], [name for name in fields if name != request[1]["parameter"]])
            self.assertEqual(request[1]["language"], "de")
            self.assertEqual(request[1]["history"], history)
            self.assertEqual(request[1]["target"]["entity_id"], "todo.shopping")
        self.assertEqual([request[1]["parameter"] for request in client.requests[-2:]], ["item", "description"])
        self.assertEqual([output["parameter"] for stage, _, output in events if stage.startswith("text:") and output],
                         ["item", "description"])
        self.assertNotIn(settings["decider"]["text_prompt"], json.dumps(events))
        self.assertEqual(result["execution"]["status"], "planned")
        self.assertFalse(client.chat_requests)

    async def test_empty_optional_description_preserves_required_item_and_reports_skip(self):
        import io
        from contextlib import redirect_stdout

        from scripts import local as benchmark

        prompt = "setze rote zwiebeln auf die einkaufsliste."
        words, _ = text_request(prompt, "todo.add_item", "test")
        item_probabilities = dict(zip([*words, "none"], [0.006283185918539801, 0.7241613548809597, 0.14640400779782275,
                                                       0.008097890908460838, 0.005208109585383435, 0.003730999014058356,
                                                       0.10611445189477528], strict=True))
        client = ScriptedModels([
            {"service": "todo.add_item"}, {"target_scope": "entities"}, {"action_entities": "entity:todo.shopping"},
            {"optional_parameters": "description"},
            {"text_words": {"type": "choice", "choice": "word:1:rote", "confidence": 0.5518, "probabilities": item_probabilities}},
            {"text_words": {"type": "choice", "choice": "none", "confidence": 0.9,
                            "probabilities": dict.fromkeys(words, 0) | {"none": 1}}},
        ])
        snapshot = {"entities": [entity("todo.shopping", "Einkaufsliste", area=None)], "services": {"todo.add_item": {"fields": {
            "item": {"required": True, "selector": {"text": {}}}, "description": {"selector": {"text": {}}},
        }}}}
        trace = []

        def record(stage, arguments, output=None):
            trace.append({"stage": stage, "input": arguments} if output is None else {"stage": stage, "output": output})

        result = await AgentEngine(normalize_settings({}), client, trace=record, dry_run=True).process(prompt, "de", snapshot)
        self.assertEqual(result["route"], "smarthome")
        self.assertEqual(result["parameters"], {"item": "rote zwiebeln"})
        self.assertEqual(result["parameter_extractions"]["description"]["status"], "skipped")
        self.assertIn("optional", result["parameter_extractions"]["description"]["reason"])
        self.assertNotIn("failures", result)
        self.assertNotIn("error", result)
        self.assertEqual(result["execution"]["status"], "planned")
        self.assertEqual(len(client.requests), 6)
        self.assertFalse(client.chat_requests)
        skipped = next(event["output"] for event in trace if event["stage"] == "text:description" and "output" in event)
        self.assertEqual(skipped["status"], "skipped")
        self.assertEqual(skipped["value"], "")
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.print_report({"result": result, "trace": trace, "context": {"entities": 1, "actions": 1},
                                    "timings_ms": {"snapshot": 0, "engine": 0, "total": 0}})
        self.assertRegex(output.getvalue(), r"item\s+rote zwiebeln")
        self.assertIn("description: skipped (no text)", output.getvalue())

    async def test_required_number_on_a_generic_service_uses_metadata(self):
        client = ScriptedModels([{"service": "number.set_value"}, {"target_scope": "implied_area"}, numeric_answers(4)])
        snapshot = {"entities": [entity("number.setting", "Setting", state="3")], "services": {
            "number.set_value": {"fields": {"value": {"required": True, "selector": {"number": {"min": 0, "max": 10}}}}}}}
        result = await AgentEngine(normalize_settings({}), client, dry_run=True).process("Set the value to five", "en", snapshot, speaker_area="Office")
        self.assertEqual(result["parameters"], {"value": 5})
        self.assertEqual(result["parameter_extractions"]["value"]["type"], "number")
        self.assertEqual(client.requests[-1][1]["current"], 3)
        self.assertIn("numeric:value", result["timings_ms"])
        self.assertEqual(result["execution"]["status"], "planned")
        self.assertFalse(client.chat_requests)

    async def test_text_and_numeric_fields_share_the_required_optional_extraction_loop(self):
        prompt = "Send dinner is ready at volume fifty"
        client = ScriptedModels([{"service": "notify.send_message"}, {"target_scope": "entities"},
                                 {"action_entities": "entity:notify.recipient"}, {"optional_parameters": "volume"},
                                 word_answers(prompt, [2, 3, 4]), numeric_answers(4)])
        snapshot = {"entities": [entity("notify.recipient", "Recipient", area=None)], "services": {"notify.send_message": {"fields": {
            "message": {"required": True, "selector": {"text": {}}},
            "volume": {"selector": {"number": {"min": 0, "max": 100, "step": 5}}},
        }}}}
        result = await AgentEngine(normalize_settings({}), client, dry_run=True).process(prompt, "en", snapshot)
        self.assertEqual(result["parameters"], {"message": "dinner is ready", "volume": 50})
        self.assertEqual(client.requests[-2][1]["other_parameters"], ["volume"])
        self.assertEqual(client.requests[-1][1]["other_parameters"], ["message"])
        self.assertIn("other_parameters", client.requests[-1][2]["numeric_score"]["instructions"])
        self.assertIn("other_parameters", client.requests[-1][2]["numeric_mode"]["instructions"])
        self.assertEqual(set(result["parameter_extractions"]), {"message", "volume"})
        self.assertIn("text:message", result["timings_ms"])
        self.assertIn("numeric:volume", result["timings_ms"])
        self.assertEqual(len(client.requests), 6)
        self.assertFalse(client.chat_requests)

    async def test_numeric_current_context_uses_only_members_supporting_that_field(self):
        client = ScriptedModels([{"service": "light.turn_on:brightness_pct"}, {"target_scope": "implied_area"},
                                 {"optional_parameters": "brightness_step_pct"}, numeric_answers(4.4, "relative")])
        snapshot = {"entities": [entity("light.dimmable", "Dimmable", supported_features=1),
                                 entity("light.switch", "Switch", supported_features=0)],
                    "services": {"light.turn_on": {"fields": {
                        "brightness_step_pct": {"filter": {"supported_features": [1]},
                                                "selector": {"number": {"min": -100, "max": 100, "step": 1}}},
                    }}}}
        result = await AgentEngine(normalize_settings({}), client, dry_run=True).process("A little brighter", "en", snapshot, speaker_area="Office")
        self.assertEqual(result["parameters"], {"brightness_step_pct": 10})
        self.assertEqual(client.requests[-1][1]["supported_entity_ids"], ["light.dimmable"])
        self.assertEqual(set(client.requests[-1][1]["current_values"]), {"light.dimmable"})
        self.assertEqual(result["parameter_extractions"]["brightness_step_pct"]["supported_entity_ids"], ["light.dimmable"])

    async def test_unsupported_field_fails_visibly_before_any_value_extraction(self):
        for selector in ({"boolean": {}}, {"select": {"options": []}}, {"date": {}}, {"object": {}}, None,
                         {"number": {}}):
            with self.subTest(selector=selector):
                fields = {"message": {"required": True, "selector": {"text": {}}},
                          "extra": {"required": True, "selector": selector}}
                client = ScriptedModels([{"service": "notify.send_message"}, {"target_scope": "entities"},
                                         {"action_entities": "entity:notify.recipient"}])
                snapshot = {"entities": [entity("notify.recipient", "Recipient", area=None)],
                            "services": {"notify.send_message": {"fields": fields}}}
                result = await AgentEngine(normalize_settings({}), client, dry_run=True).process("Send a message", "en", snapshot)
                self.assertEqual(result["route"], "unsupported")
                self.assertIn("not yet supported", result["answer"])
                self.assertIn("extra", result["error"])
                self.assertNotIn("parameters", result)
                self.assertNotIn("parameter_extractions", result)
                self.assertEqual(result["failures"][0]["stage"], "parameters")
                self.assertEqual(result["failures"][0]["device_command"], "not_sent")
                self.assertEqual(len(client.requests), 3)
                self.assertFalse(client.chat_requests)

    async def test_selected_unsupported_optional_field_stops_before_extracting_required_text(self):
        client = ScriptedModels([{"service": "todo.add_item"}, {"target_scope": "entities"},
                                 {"action_entities": "entity:todo.shopping"}, {"optional_parameters": "due_date"}])
        snapshot = {"entities": [entity("todo.shopping", "Shopping", area=None)], "services": {"todo.add_item": {"fields": {
            "item": {"required": True, "selector": {"text": {}}}, "due_date": {"selector": {"date": {}}},
        }}}}
        result = await AgentEngine(normalize_settings({}), client, dry_run=True).process("Add milk for tomorrow", "en", snapshot)
        self.assertEqual(result["route"], "unsupported")
        self.assertEqual(result["parameter_selection"]["selected_parameters"], ["due_date"])
        self.assertIn("due_date", result["error"])
        self.assertIn("date format is not yet supported", result["error"])
        self.assertEqual(len(client.requests), 4)
        self.assertNotIn("parameter_extractions", result)

    async def test_unselected_unsupported_optional_field_does_not_block_supported_extraction(self):
        prompt = "Add milk"
        client = ScriptedModels([{"service": "todo.add_item"}, {"target_scope": "entities"},
                                 {"action_entities": "entity:todo.shopping"}, {"optional_parameters": "none"}, word_answers(prompt, [2])])
        snapshot = {"entities": [entity("todo.shopping", "Shopping", area=None)], "services": {"todo.add_item": {"fields": {
            "item": {"required": True, "selector": {"text": {}}}, "due_date": {"selector": {"date": {}}},
        }}}}
        result = await AgentEngine(normalize_settings({}), client, dry_run=True).process(prompt, "en", snapshot)
        self.assertEqual(result["parameters"], {"item": "milk"})
        self.assertNotIn("error", result)


def select_answers(probabilities, multiple=True, confidence=0.9):
    return {"select_values" if multiple else "select": {
        "type": "choice", "choice": max(probabilities, key=probabilities.get), "confidence": confidence,
        "probabilities": probabilities,
    }}


class SelectValueTests(unittest.TestCase):
    def request(self, multiple=True, options=None):
        field = {"description": "Which item statuses to retrieve", "selector": {"select": {
            "options": options if options is not None else ["needs_action", "completed"], "multiple": multiple,
        }}}
        return select_request("Was muss noch eingekauft werden?", "todo.get_items", "status", field,
                              {"status": field, "description": {"selector": {"text": {}}}}, SELECT_INSTRUCTIONS)

    def test_request_preserves_actual_values_labels_and_other_field_metadata(self):
        options, payload = self.request(options=[{"value": "needs_action", "label": "Offen"}, "completed"])
        self.assertEqual(options["value:needs_action"], {"value": "needs_action", "label": "Offen"})
        self.assertEqual(payload["state"]["other_parameters"], ["description"])
        self.assertTrue(payload["state"]["multiple"])
        self.assertEqual(set(payload["questions"]["select_values"]["criteria"]), {*options, "none"})
        self.assertIn("muss noch eingekauft werden", payload["questions"]["select_values"]["instructions"])

    def test_multiple_values_use_relative_cutoff_without_none_veto(self):
        options, _ = self.request()
        for probabilities, expected in (
            ({"value:needs_action": 0.9, "value:completed": 0.05, "none": 0.05}, ["needs_action"]),
            ({"value:needs_action": 0.45, "value:completed": 0.45, "none": 0.1}, ["needs_action", "completed"]),
            ({"value:needs_action": 0.2, "value:completed": 0.05, "none": 0.75}, ["needs_action"]),
            ({"value:needs_action": 0.15, "value:completed": 0.1, "none": 0.75}, ["needs_action"]),
            ({"value:needs_action": 0.1, "value:completed": 0.1, "none": 0.8}, None),
        ):
            with self.subTest(probabilities=probabilities):
                result = decode_select({"answers": select_answers(probabilities, confidence=0)}, options, True, 0.3, 1)
                self.assertEqual(result["value"], expected)
                self.assertEqual(result["effective_probability_threshold"], 0.15)
                self.assertEqual(result["real_option_count"], 2)

    def test_single_select_returns_one_exact_value_and_respects_confidence_and_ambiguity(self):
        options, _ = self.request(multiple=False, options=[{"value": "heat", "label": "Heizen"}, "off"])
        for probabilities, confidence, expected in (
            ({"value:heat": 0.8, "value:off": 0.1, "none": 0.1}, 0.9, "heat"),
            ({"value:heat": 0.8, "value:off": 0.1, "none": 0.1}, 0.49, None),
            ({"value:heat": 0.45, "value:off": 0.45, "none": 0.1}, 0.9, None),
            ({"value:heat": 0.1, "value:off": 0.1, "none": 0.8}, 0.9, None),
        ):
            with self.subTest(probabilities=probabilities, confidence=confidence):
                result = decode_select({"answers": select_answers(probabilities, False, confidence)}, options, False, 0.3, 0.5)
                self.assertEqual(result["value"], expected)

    def test_a_literal_none_option_is_distinct_from_the_fallback(self):
        options, _ = self.request(multiple=False, options=["none", "hold"])
        response = {"answers": select_answers({"value:none": 0.9, "value:hold": 0.05, "none": 0.05}, False)}
        self.assertEqual(decode_select(response, options, False, 0.3, 0)["value"], "none")

    def test_custom_dynamic_duplicate_and_invalid_options_are_explicitly_unsupported(self):
        for selector in ({}, {"options": []}, {"options": "off"}, {"options": ["off"], "custom_value": True},
                         {"options": ["off"], "multiple": "true"}, {"options": ["off", "off"]},
                         {"options": [True]}, {"options": [""]}, {"options": [{"label": "Off"}]},
                         {"options": [{"value": "off", "label": False}]}):
            field = {"selector": {"select": selector}}
            with self.subTest(selector=selector), self.assertRaisesRegex(UnsupportedParameter, "not yet supported"):
                select_request("test", "test.action", "mode", field, {"mode": field}, SELECT_INSTRUCTIONS)

    def test_invalid_or_invented_model_values_are_rejected(self):
        options, _ = self.request()
        valid = {"answers": select_answers({"value:needs_action": 0.9, "value:completed": 0.1, "none": 0})}
        cases = [None, {}, {"answers": {}}, {"answers": select_answers({"invented": 1})}]
        for key, value in (("type", "score"), ("confidence", True), ("probabilities", {"none": 1}), ("choice", "invented")):
            invalid = copy.deepcopy(valid)
            invalid["answers"]["select_values"][key] = value
            cases.append(invalid)
        for response in cases:
            with self.subTest(response=response), self.assertRaisesRegex(ModelError, "Malformed select extraction"):
                decode_select(response, options, True, 0.3, 0)

    def test_settings_reject_invalid_thresholds_and_preserve_custom_prompt(self):
        saved = normalize_settings({"decider": {"select_prompt": "Custom choices"}})
        self.assertEqual(saved["decider"]["select_prompt"], "Custom choices")
        self.assertEqual(saved["decider"]["select_probability_threshold"], 0.3)
        self.assertEqual(normalize_settings({"decider": {"select_prompt": ""}})["decider"]["select_prompt"], SELECT_INSTRUCTIONS)
        for field in ("select_confidence", "select_probability_threshold"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_settings({"decider": {"url": "http://local/v1/systemone", field: 1.1}})


class SelectParameterEngineTests(unittest.IsolatedAsyncioTestCase):
    def snapshot(self):
        return {"entities": [entity("todo.shopping", "Einkaufsliste", area=None, state="2")], "services": {
            "todo.get_items": {"response": {"optional": False}, "fields": {"status": {"selector": {"select": {
                "options": ["needs_action", "completed"], "multiple": True, "custom_value": False,
            }}}}},
        }}

    def choices(self, probabilities):
        return [{"service": "todo.get_items"}, {"target_scope": "entities"},
                {"action_entities": "entity:todo.shopping"}, {"optional_parameters": "status"},
                select_answers(probabilities)]

    async def test_shopping_status_reaches_executor_as_a_list_and_is_visible_in_debug(self):
        client = ScriptedModels(self.choices({"value:needs_action": 0.9, "value:completed": 0.05, "none": 0.05}))
        execute = AsyncMock(return_value={"todo.shopping": {"items": [{"summary": "Milch", "status": "needs_action"}]}})
        events = []
        history = [{"role": "user", "content": "Wir reden über die Einkaufsliste."}, {"role": "assistant", "content": "ok"}]
        result = await AgentEngine(normalize_settings({"history_enabled": True}), client, execute=execute, history=history,
                                   trace=lambda *event: events.append(event)).process(
            "Was steht auf der Einkaufsliste und muss noch eingekauft werden?", "de", self.snapshot())
        self.assertEqual(result["parameters"], {"status": ["needs_action"]})
        self.assertEqual(result["answer"], "Einkaufsliste: Milch.")
        execute.assert_awaited_once_with("todo.get_items", {"status": ["needs_action"], "entity_id": ["todo.shopping"]})
        state = client.requests[-1][1]
        self.assertEqual(state["language"], "de")
        self.assertEqual(state["history"], history)
        self.assertEqual(state["supported_entity_ids"], ["todo.shopping"])
        output = next(output for stage, _, output in events if stage == "select:status" and output)
        self.assertEqual(output["value"], ["needs_action"])
        self.assertEqual(output["decisions"]["select_values"]["effective_probability_threshold"], 0.15)
        self.assertNotIn(SELECT_INSTRUCTIONS, json.dumps(events))

    async def test_no_retained_status_stops_without_broadening_or_executing(self):
        client = ScriptedModels(self.choices({"value:needs_action": 0.1, "value:completed": 0.1, "none": 0.8}))
        execute = AsyncMock()
        result = await AgentEngine(normalize_settings({}), client, execute=execute).process("Read open items", "en", self.snapshot())
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["failures"][0]["stage"], "select:status")
        self.assertNotIn("execution", result)
        execute.assert_not_awaited()

    async def test_required_single_select_uses_the_same_dispatcher_and_preserves_dry_run(self):
        snapshot = {"entities": [entity("climate.office", "Thermostat")], "services": {
            "climate.set_hvac_mode": {"fields": {"hvac_mode": {"required": True, "selector": {"select": {
                "options": [{"value": "heat", "label": "Heizen"}, "off"],
            }}}}},
        }}
        client = ScriptedModels([{"service": "climate.set_hvac_mode"}, {"target_scope": "implied_area"},
                                 select_answers({"value:heat": 0.9, "value:off": 0.05, "none": 0.05}, False)])
        execute = AsyncMock()
        result = await AgentEngine(normalize_settings({}), client, execute=execute, dry_run=True).process(
            "Stelle die Heizung auf Heizbetrieb", "de", snapshot, speaker_area="Office")
        self.assertEqual(result["parameters"], {"hvac_mode": "heat"})
        self.assertEqual(result["execution"]["calls"][0]["arguments"], {"hvac_mode": "heat", "entity_id": ["climate.office"]})
        self.assertEqual(result["execution"]["status"], "planned")
        execute.assert_not_awaited()

    async def test_text_and_select_fields_are_extracted_separately_and_execute_together(self):
        prompt = "Mark milk completed"
        snapshot = {"entities": [entity("todo.shopping", "Shopping", area=None)], "services": {
            "todo.update_item": {"fields": {
                "item": {"required": True, "selector": {"text": {}}},
                "status": {"selector": {"select": {"options": ["needs_action", "completed"]}}},
            }},
        }}
        client = ScriptedModels([
            {"service": "todo.update_item"}, {"target_scope": "entities"}, {"action_entities": "entity:todo.shopping"},
            {"optional_parameters": "status"}, word_answers(prompt, [2]),
            select_answers({"value:needs_action": 0.05, "value:completed": 0.9, "none": 0.05}, False),
        ])
        execute = AsyncMock(return_value=None)
        result = await AgentEngine(normalize_settings({}), client, execute=execute).process(prompt, "en", snapshot)
        self.assertEqual(result["parameters"], {"item": "milk", "status": "completed"})
        self.assertEqual(result["answer"], "Okay")
        self.assertEqual(set(result["parameter_extractions"]), {"item", "status"})
        self.assertEqual(client.requests[-1][1]["other_parameters"], ["item"])
        self.assertEqual(client.requests[-2][1]["other_parameters"], ["status"])
        execute.assert_awaited_once_with("todo.update_item", {"item": "milk", "status": "completed", "entity_id": ["todo.shopping"]})


if __name__ == "__main__":
    unittest.main()
