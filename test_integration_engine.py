import copy
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from custom_components.glow_assistant.client import ModelClient, ModelError
from custom_components.glow_assistant.engine import (
    AgentEngine,
    action_options,
    read_choice,
)
from custom_components.glow_assistant.settings import (
    DEFAULTS,
    normalize_settings,
    validate_settings,
)
from custom_components.glow_assistant.text_values import TEXT_ACTIONS


def entity(entity_id, name, area="office", state="on", **attributes):
    return {
        "entity_id": entity_id, "name": name, "aliases": [], "area_id": area,
        "area_name": area.title() if area else None, "area_aliases": [],
        "domain": entity_id.split(".")[0], "state": state, "attributes": attributes,
    }


SNAPSHOT = {
    "entities": [entity("light.desk", "Desk lamp"), entity("light.ceiling", "Ceiling lamp"),
                 entity("light.bed", "Bed lamp", "bedroom"),
                 entity("sensor.office", "Room temperature", state="22", unit_of_measurement="°C")],
    "services": {
        "light.turn_off": {"target": {}, "fields": {}},
        "light.turn_on": {"target": {}, "fields": {"brightness_pct": {}}},
    },
}


def query_candidates(request):
    return [json.loads(value) for key, value in request[2]["query_targets"]["criteria"].items() if key != "none"]


class ScriptedModels:
    def __init__(self, choices, answers=None, confidence=0.9):
        self.choices = list(choices)
        self.requests = []
        self.answers = list(answers or ["Erledigt."])
        self.chat_requests = []
        self.confidence = confidence

    async def decide(self, config, state, questions):
        self.requests.append(copy.deepcopy((config, state, questions)))
        choices = self.choices.pop(0)
        answers = {}
        for name, question in questions.items():
            if isinstance(choices.get(name), dict):
                answers[name] = copy.deepcopy(choices[name])
                continue
            if name == "query_targets" and name not in choices:
                selected = choices.get("targets")
                if selected is None:
                    legacy = choices.get("target", "unsure")
                    selected = ([legacy.removeprefix("entity:")] if legacy.startswith("entity:")
                                else "all" if legacy.startswith("area:") or legacy == "all_areas" else [])
                keys = [key for key in question["criteria"] if key != "none"
                        and (selected == "all" or key.removeprefix("entity:") in selected)] or ["none"]
                probabilities = dict.fromkeys(question["criteria"], 0)
                probabilities.update({key: 1 / len(keys) for key in keys})
                answers[name] = {"type": "choice", "choice": keys[0], "confidence": self.confidence,
                                 "probabilities": probabilities}
                continue
            choice = choices.get(name, "unsure")
            probabilities = dict.fromkeys(question["criteria"], 0)
            if choice in probabilities:
                probabilities[choice] = 1
            answers[name] = {"type": "choice", "choice": choice, "confidence": self.confidence, "probabilities": probabilities}
        return {"answers": answers, "usage": {"input_tokens": 5}}

    async def chat(self, config, messages):
        self.chat_requests.append(copy.deepcopy((config, messages)))
        reply = self.answers.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, {"completion_tokens": 5}


class IntegrationEngineTests(unittest.IsolatedAsyncioTestCase):
    async def run_agent(self, choices, snapshot=SNAPSHOT, settings=None, answers=None, trace=None, history=None, language="de", prompt="test request", dry_run=False, speaker_area=None):
        models = ScriptedModels(choices, answers)
        agent = AgentEngine(settings or normalize_settings({}), models, trace=trace, history=history, dry_run=dry_run)
        result = await agent.process(prompt, language, snapshot, speaker_area=speaker_area)
        return result, models

    async def run_parameter_stage(self, choices, snapshot=SNAPSHOT, settings=None, trace=None):
        models = ScriptedModels(choices[1:])
        selected = choices[0]["service"]
        service = action_options(snapshot["services"])[selected]["service"]
        result = {"request": "test request", "language": "de", "service": service,
                  "decisions": {}, "usage": {}, "timings_ms": {},
                  "action_metadata": {"service": service, **snapshot["services"][service]}}
        agent = AgentEngine(settings or normalize_settings({}), models, trace=trace)
        await agent._select_parameters(result, [item for item in snapshot["entities"] if item["domain"] == service.split(".")[0]])
        return result, models

    async def test_yes_no_query_skips_entity_filter_and_only_translates_the_decision(self):
        for answer, translated in (("yes", "Ja."), ("no", "Nein.")):
            with self.subTest(answer=answer):
                events = []
                result, models = await self.run_agent([
                    {"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
                    {"query_type": "yes_no"}, {"binary_answer": answer},
                ], prompt="Ist die Lampe im Büro an?", answers=[translated],
                    trace=lambda *event: events.append(event))
                self.assertEqual(result["answer"], translated)
                self.assertEqual(result["state_answer"], answer)
                self.assertEqual(result["query_type"], "yes_no")
                self.assertIsNone(result["target"])
                self.assertNotIn("target_selection", result)
                self.assertNotIn("state_info", result)
                self.assertEqual([set(questions) for _, _, questions in models.requests],
                                 [{"service"}, {"area"}, {"domain"}, {"query_type"}, {"binary_answer"}])
                state = models.requests[-1][1]
                self.assertEqual({item["entity_id"] for item in state["entities"]}, {"light.desk", "light.ceiling"})
                self.assertTrue(all(item["state"] == "on" for item in state["entities"]))
                system, facts = models.chat_requests[0][1]
                self.assertIn("only translate", system["content"])
                self.assertIn("Assist pipeline language: de", system["content"])
                self.assertEqual(json.loads(facts["content"]), {"state_answer": answer})
                outputs = {stage: output for stage, _, output in events if output is not None}
                for stage in ("query_type", "binary_answer"):
                    self.assertIn("duration_ms", outputs[stage])
                    self.assertEqual(outputs[stage]["decisions"][stage]["confidence"], 0.9)
                    self.assertEqual(outputs[stage]["decisions"][stage]["effective_choice"],
                                     "yes_no" if stage == "query_type" else answer)


    async def test_whole_home_yes_no_uses_current_states_and_history_with_no_entity_decisions(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["entities"].extend([
            entity("binary_sensor.left_window", "Left window", device_class="window"),
            entity("binary_sensor.right_window", "Right window", state="off", area="bedroom", device_class="window"),
            entity("binary_sensor.unassigned_window", "Other window", state="unavailable", area=None, temperature=99),
            entity("binary_sensor.motion", "Motion", device_class="motion"),
        ])
        snapshot["entities"][4]["aliases"] = ["Renamed window"]
        history = [{"role": "user", "content": "Are any windows open in the house?"},
                   {"role": "assistant", "content": "Yes."}]
        settings = normalize_settings({"history_enabled": True, "decider": {
            "query_type_prompt": "Custom type rules", "binary_answer_prompt": "Custom binary rules",
            "query_type_confidence": 0.2, "binary_answer_confidence": 0.3,
        }})
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "all_areas"}, {"domain": "binary_sensor"},
            {"query_type": "yes_no"}, {"binary_answer": "yes"},
        ], snapshot=snapshot, history=history, settings=settings, language="fr-CH", prompt="And now?")
        for _, state, _ in models.requests:
            self.assertEqual(state["language"], "fr-CH")
            self.assertEqual(state["history"], history)
        self.assertIn("Custom type rules", models.requests[-2][2]["query_type"]["instructions"])
        self.assertIn("Custom binary rules", models.requests[-1][2]["binary_answer"]["instructions"])
        candidates = {item["entity_id"]: item for item in models.requests[-1][1]["entities"]}
        self.assertEqual(set(candidates), {item["entity_id"] for item in snapshot["entities"]
                                           if item["domain"] == "binary_sensor"})
        self.assertEqual(candidates["binary_sensor.left_window"]["aliases"], ["Renamed window"])
        self.assertEqual(candidates["binary_sensor.right_window"]["area"], "Bedroom")
        self.assertEqual(candidates["binary_sensor.right_window"]["state"], "off")
        self.assertEqual(candidates["binary_sensor.unassigned_window"]["state"], "unavailable")
        self.assertEqual(candidates["binary_sensor.unassigned_window"]["attributes"], {})
        self.assertIsNone(candidates["binary_sensor.unassigned_window"]["area"])
        self.assertIn(json.dumps(history), models.chat_requests[0][1][0]["content"])
        self.assertIn("Assist pipeline language: fr-CH", models.chat_requests[0][1][0]["content"])
        self.assertEqual(result["state_answer"], "yes")


    async def test_uncertain_query_type_or_binary_answer_stops_without_chat_or_entity_filter(self):
        for stage in ("query_type", "binary_answer"):
            for choice, threshold in (("unsure", 0), ("yes_no" if stage == "query_type" else "yes", 0.95)):
                with self.subTest(stage=stage, choice=choice, threshold=threshold):
                    choices = [{"service": "query_state"}, {"area": "area:office"}, {"domain": "light"}]
                    if stage == "binary_answer":
                        choices.append({"query_type": "yes_no"})
                    choices.append({stage: choice})
                    settings = normalize_settings({"decider": {f"{stage}_confidence": threshold}})
                    events = []
                    result, models = await self.run_agent(
                        choices, settings=settings, trace=lambda *event: events.append(event),
                    )
                    self.assertEqual(result["route"], "unsure")
                    self.assertEqual(result["answer"], settings["unsure_reply"])
                    self.assertEqual(result["failures"][0]["stage"], stage)
                    self.assertFalse(models.chat_requests)
                    self.assertNotIn("target", result["decisions"])
                    output = next(output for name, _, output in events if name == stage and output)
                    self.assertEqual(output["decisions"][stage]["minimum_confidence"], threshold)


    async def test_yes_no_translation_failure_preserves_decision_and_reports_answer_failure(self):
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
            {"query_type": "yes_no"}, {"binary_answer": "no"},
        ], answers=[ModelError("Model reply reached its output token limit.")])
        self.assertEqual(result["state_answer"], "no")
        self.assertEqual(result["answer"], DEFAULTS["error_reply"])
        self.assertEqual(result["failures"][0]["stage"], "answer")
        self.assertEqual(len(models.chat_requests), 1)


    async def test_invalid_binary_choice_is_rejected_without_an_answer_or_action(self):
        with self.assertRaisesRegex(ModelError, "Invalid classifier"):
            await self.run_agent([
                {"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
                {"query_type": "yes_no"}, {"binary_answer": "maybe"},
            ])

    async def test_pipeline_language_reaches_every_decider_and_the_answer_model(self):
        settings = normalize_settings({"answer": {"prompt": "Custom answer rules"}})
        _, models = await self.run_agent([
            {"service": "query_state"},
            {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"}, {"target": "entity:light.desk"},
        ], settings=settings, language="fr-CH")
        for _, state, _ in models.requests:
            self.assertEqual(state["language"], "fr-CH")
        system = models.chat_requests[0][1][0]["content"]
        self.assertIn("Custom answer rules", system)
        self.assertIn("Assist pipeline language: fr-CH", system)
        self.assertIn("Write the spoken reply in this language", system)

    async def test_actions_stop_after_selection_in_ha_and_plan_only_mode(self):
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                result, models = await self.run_agent([
                    {"service": "light.turn_on"}, {"target_scope": "areas"}, {"action_areas": "area:office"}, {"optional_parameters": "none"},
                ], dry_run=dry_run, prompt="Turn on the office lights")
                self.assertEqual(result["service"], "light.turn_on")
                self.assertEqual(result["domain"], "light")
                self.assertEqual(result["action_metadata"], {
                    "selection": "light.turn_on", "service": "light.turn_on", **SNAPSHOT["services"]["light.turn_on"],
                })
                self.assertEqual(len(models.requests), 4)
                self.assertEqual(result["target_scope"], "areas")
                self.assertEqual(result["parameter_selection"]["selected_parameters"], [])
                self.assertFalse(models.chat_requests)
                self.assertEqual(result["target"]["area_id"], "office")
                self.assertNotIn("execution", result)
                self.assertNotIn("failures", result)
                self.assertIn("No action was executed", result["answer"])


    async def test_action_target_scope_choices_and_speaker_context(self):
        with patch.object(AgentEngine, "_select_action_targets", new_callable=AsyncMock, return_value=SNAPSHOT["entities"]), \
                patch.object(AgentEngine, "_extract_text", new_callable=AsyncMock):
            for scope, location, prompt in (
                ("entities", None, "Add milk to the shopping list"),
                ("entities", "Kitchen", "Setze Brot auf die Einkaufsliste"),
                ("entities_explicit_area", None, "Turn off the desk lamp in the office"),
                ("entities_explicit_area", "Kitchen", "Schalte die linke und rechte Schreibtischlampe im Büro aus"),
                ("entities_implied_area", "Kitchen", "Turn off the desk lamp"),
                ("areas", "Kitchen", "Mach das Licht im Büro aus"),
                ("areas", None, "Mach das Licht in Küche und Flur aus"),
                ("implied_area", " Kitchen ", "Turn off the lights"),
            ):
                with self.subTest(scope=scope, location=location):
                    snapshot = copy.deepcopy(SNAPSHOT)
                    snapshot["services"]["todo.add_item"] = {"target": {}, "fields": {"item": {"required": True}}}
                    service = "todo.add_item" if scope == "entities" else "light.turn_off"
                    result, models = await self.run_agent([
                        {"service": service}, {"target_scope": scope},
                    ], snapshot=snapshot, prompt=prompt, speaker_area=location)
                    self.assertEqual(result["target_scope"], scope)
                    self.assertEqual(result["speaker_area"], location.strip() if location else None)
                    self.assertEqual(models.requests[-1][1], {
                        "prompt": prompt, "service": service, "speaker_area": result["speaker_area"], "language": "de",
                    })
                    choices = set(models.requests[-1][2]["target_scope"]["criteria"])
                    self.assertEqual(choices, {"entities", "entities_explicit_area", "areas", "unsure"} | ({"entities_implied_area", "implied_area"} if location else set()))
                    self.assertIsNone(result["target"])
                    self.assertNotIn("optional_parameters", result["decisions"])
                    self.assertNotIn("execution", result)
                    self.assertFalse(models.chat_requests)


    async def test_scope_unsure_or_low_confidence_stops_after_scope_decision(self):
        for choice, threshold in (("unsure", 0), ("areas", 0.95)):
            with self.subTest(choice=choice):
                settings = normalize_settings({"decider": {"target_scope_confidence": threshold}})
                events = []
                result, models = await self.run_agent([
                    {"service": "light.turn_off"}, {"target_scope": choice},
                ], settings=settings, trace=lambda *event: events.append(event))
                self.assertEqual(result["route"], "unsure")
                self.assertEqual(result["failures"][0]["stage"], "target_scope")
                self.assertEqual(len(models.requests), 2)
                output = next(output for stage, _, output in events if stage == "target_scope" and output)
                self.assertEqual(output["decisions"]["target_scope"]["effective_choice"], "unsure")
                self.assertEqual(output["decisions"]["target_scope"]["minimum_confidence"], threshold)
                self.assertNotIn("parameter_selection", result)


    async def test_implied_area_choices_cannot_be_selected_without_supplied_location(self):
        for choice in ("entities_implied_area", "implied_area"):
            with self.subTest(choice=choice), self.assertRaisesRegex(ModelError, "Invalid classifier"):
                await self.run_agent([{"service": "light.turn_off"}, {"target_scope": choice}], speaker_area=" ")

    async def test_parameterized_action_returns_all_raw_metadata_without_extraction(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        description = {
            "name": "Set temperature", "description": "Set the thermostat target", "target": {"entity": {"domain": "climate"}},
            "fields": {
                "temperature": {"required": True, "description": "New target", "example": 21,
                                "selector": {"number": {"min": 5, "max": 45, "step": 1, "unit_of_measurement": "°C"}}},
                "advanced": {"collapsed": True, "fields": {"hvac_mode": {"selector": {"select": {"options": ["heat", "off"]}}}}},
            },
        }
        snapshot["services"]["climate.set_temperature"] = description
        snapshot["entities"].append(entity("climate.office", "Thermostat"))
        original = copy.deepcopy(snapshot)
        history = [{"role": "user", "content": "Set the office to 20 degrees"}, {"role": "assistant", "content": "Selected."}]
        settings = normalize_settings({"history_enabled": True})
        result, models = await self.run_agent([
            {"service": "climate.set_temperature:temperature"}, {"target_scope": "areas"}, {"action_areas": "area:office"}, {"optional_parameters": "none"},
        ], snapshot=snapshot, settings=settings, history=history, language="fr-CH")
        self.assertEqual(result["action_metadata"], {"service": "climate.set_temperature",
                                                  "selection": "climate.set_temperature:temperature", **description})
        self.assertEqual(snapshot, original)
        self.assertEqual(models.requests[0][1]["history"], history)
        self.assertEqual(models.requests[0][1]["language"], "fr-CH")
        self.assertEqual(set(result["decisions"]), {"service", "target_scope", "action_areas", "optional_parameters"})
        self.assertEqual(result["parameter_selection"]["selected_parameters"], [])
        self.assertEqual(set(models.requests[1][2]["target_scope"]["criteria"]), {"entities", "entities_explicit_area", "areas", "unsure"})
        self.assertEqual(models.requests[-1][1]["history"], history)
        self.assertEqual(models.requests[-1][1]["language"], "fr-CH")
        self.assertFalse(models.chat_requests)


    async def test_optional_parameters_use_all_choice_probabilities_even_when_none_wins(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["services"]["light.turn_on"]["fields"] = {
            "brightness_pct": {"selector": {"number": {"min": 0, "max": 100}}},
            "hue": {"selector": {"number": {"min": 0, "max": 360}}},
            "child_lock": {"selector": {"boolean": {}}},
        }
        probabilities = {"brightness_pct": 0.25, "hue": 0.2, "child_lock": 0.05, "none": 0.5}
        events = []
        settings = normalize_settings({"decider": {"parameter_prompt": "Custom parameter task", "parameter_probability_threshold": 0.15}})
        result, models = await self.run_parameter_stage([
            {"service": "light.turn_on"}, {"optional_parameters": {
                "type": "choice", "choice": "none", "confidence": 0, "probabilities": probabilities,
            }},
        ], snapshot=snapshot, settings=settings, trace=lambda *event: events.append(event))
        self.assertEqual(result["parameter_selection"]["selected_parameters"], ["brightness_pct", "hue"])
        self.assertEqual(len(models.requests), 1)
        self.assertEqual(set(models.requests[-1][2]), {"optional_parameters"})
        self.assertEqual(models.requests[-1][2]["optional_parameters"]["instructions"], "Custom parameter task")
        self.assertNotIn("entities", models.requests[-1][1])
        self.assertNotIn("targets", models.requests[-1][1])
        output = next(output for stage, _, output in events if stage == "optional_parameters" and output)
        self.assertEqual(output["selected_parameters"], ["brightness_pct", "hue"])
        self.assertEqual(output["probability_threshold"], 0.15)
        self.assertEqual(output["decisions"]["optional_parameters"]["probabilities"], probabilities)
        self.assertNotIn("minimum_confidence", output["decisions"]["optional_parameters"])
        self.assertNotIn("Custom parameter task", json.dumps(events))
        self.assertFalse(models.chat_requests)


    async def test_required_only_action_skips_optional_parameter_call(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["services"]["todo.add_item"] = {"target": {}, "fields": {
            "item": {"required": True, "selector": {"text": {}}},
        }}
        result, models = await self.run_parameter_stage([{"service": "todo.add_item"}], snapshot=snapshot)
        self.assertEqual(result["required_parameters"], ["item"])
        self.assertEqual(result["parameter_selection"]["selected_parameters"], [])
        self.assertEqual(len(models.requests), 0)
        self.assertFalse(models.chat_requests)


    async def test_invalid_optional_parameter_distribution_stops_without_action_or_answer(self):
        events = []
        with self.assertRaisesRegex(ModelError, "Malformed optional parameter"):
            await self.run_parameter_stage([
                {"service": "light.turn_on"}, {"optional_parameters": {
                    "type": "choice", "choice": "brightness_pct", "confidence": 0.9,
                    "probabilities": {"brightness_pct": 1},
                }},
            ], trace=lambda *event: events.append(event))
        self.assertEqual(events[-1][0], "optional_parameters")
        self.assertEqual(events[-1][2]["status"], "error")
        self.assertFalse(any(stage in ("execution", "answer") for stage, _, _ in events))


    async def test_plan_only_query_reads_states_without_an_execution_step(self):
        result, _ = await self.run_agent([
            {"service": "query_state"}, {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"}, {"targets": ["light.desk"]},
        ], dry_run=True)

        self.assertEqual(result["state_info"][0]["entity_id"], "light.desk")
        self.assertNotIn("execution", result)

    async def test_plan_only_general_route_cannot_connect_to_or_execute_mcp_tools(self):
        settings = normalize_settings({"general": {"enable": True, "mcp_servers": [{"enable": True}]}})
        with patch("custom_components.glow_assistant.engine.general_answer", new_callable=AsyncMock) as general:
            with self.assertRaisesRegex(ModelError, "plan-only"):
                await self.run_agent([{"service": "not-smarthome-related"}], settings=settings, dry_run=True)
        general.assert_not_awaited()


    async def test_history_is_sent_to_every_decider_and_the_answer_model(self):
        history = [{"role": "user", "content": "Is the office lamp on?"}, {"role": "assistant", "content": "Yes, it is on."}]
        settings = normalize_settings({"history_enabled": True})
        result, models = await self.run_agent([
            {"service": "query_state"},
            {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"}, {"target": "entity:light.desk"},
        ], settings=settings, history=history)
        self.assertEqual(result["answer"], "Erledigt.")
        for _, state, questions in models.requests:
            self.assertEqual(state["history"], history)
            if "task" in state:
                self.assertIn("resolve references", state["task"])
            else:
                self.assertTrue(all("resolve references" in item["instructions"] for item in questions.values()))
        self.assertIn(json.dumps(history, ensure_ascii=False), models.chat_requests[0][1][0]["content"])
        self.assertEqual(len(models.requests), 5)


    async def test_disabled_history_is_not_forwarded_even_if_supplied(self):
        _, models = await self.run_agent([
            {"service": "query_state"},
            {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"}, {"target": "entity:light.desk"},
        ], history=[{"role": "user", "content": "private previous message"}])
        self.assertNotIn("private previous message", json.dumps(models.requests))
        self.assertNotIn("private previous message", json.dumps(models.chat_requests))

    async def test_general_history_uses_the_same_plain_message_context(self):
        history = [{"role": "user", "content": "How tall is the Eiffel Tower?"}, {"role": "assistant", "content": "330 metres."}]
        settings = normalize_settings({"history_enabled": True, "general": {"enable": True}})
        _, models = await self.run_agent([{"service": "not-smarthome-related"}], settings=settings, history=history)
        self.assertEqual(models.requests[0][1]["history"], history)
        self.assertIn(json.dumps(history, ensure_ascii=False), models.chat_requests[0][1][0]["content"])

    async def test_mcp_runs_only_on_the_enabled_general_route(self):
        from unittest.mock import patch

        settings = normalize_settings({"history_enabled": True, "general": {
            "enable": True, "mcp_servers": [{"enable": True}],
        }})
        history = [{"role": "assistant", "content": "Previous reply"}]
        with patch("custom_components.glow_assistant.engine.general_answer", new_callable=AsyncMock,
                   return_value=("Answer with tools", [{"completion_tokens": 10}])) as general:
            result, models = await self.run_agent(
                [{"service": "not-smarthome-related"}], settings=settings, history=history,
            )
        self.assertEqual(result["answer"], "Answer with tools")
        self.assertIn(json.dumps(history, ensure_ascii=False), general.await_args.args[2][0]["content"])
        self.assertIn("Assist pipeline language: de", general.await_args.args[2][0]["content"])
        self.assertEqual(len(models.requests), 1)
        self.assertFalse(models.chat_requests)


    async def test_enabled_mcp_servers_are_not_contacted_for_smarthome_or_disabled_general(self):
        from unittest.mock import patch

        settings = normalize_settings({"general": {"enable": False, "mcp_servers": [{"enable": True}]}})
        with patch("custom_components.glow_assistant.engine.general_answer", new_callable=AsyncMock) as general:
            result, _ = await self.run_agent([
                {"service": "query_state"},
                {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"}, {"target": "entity:light.desk"},
            ], settings=settings)
            self.assertEqual(result["route"], "smarthome")
            result, _ = await self.run_agent([{"service": "not-smarthome-related"}], settings=settings)
            self.assertEqual(result["route"], "unsure")
        general.assert_not_awaited()

    async def test_action_selection_trace_contains_metadata_and_no_downstream_calls(self):
        events = []
        result, models = await self.run_agent([
            {"service": "light.turn_off"}, {"target_scope": "entities_explicit_area"},
            {"action_areas": "area:office"}, {"action_entities": "entity:light.desk"},
        ], trace=lambda *event: events.append(event))
        self.assertEqual([stage for stage, _, _ in events], ["service", "service", "target_scope", "target_scope", "action_areas", "action_areas", "action_entities", "action_entities", "parameter_capabilities", "parameter_capabilities"])
        output = events[1][2]
        self.assertEqual(output["action_metadata"], result["action_metadata"])
        self.assertEqual(output["decisions"]["service"]["choice"], "light.turn_off")
        self.assertEqual(output["decisions"]["service"]["confidence"], 0.9)
        self.assertEqual(output["usage"], {"input_tokens": 5})
        self.assertGreaterEqual(output["duration_ms"], 0)
        self.assertEqual(output["status"], "completed")
        self.assertEqual(len(models.requests), 4)
        self.assertFalse(models.chat_requests)
        self.assertEqual(result["target_scope"], "entities_explicit_area")
        self.assertEqual(result["parameter_selection"]["selected_parameters"], [])
        self.assertNotIn("optional_parameters", result["timings_ms"])


    async def test_trace_explains_confidence_rejection(self):
        events = []
        settings = normalize_settings({"decider": {"service_confidence": 0.95}})
        result, _ = await self.run_agent(
            [{"service": "light.turn_off"}], settings=settings, trace=lambda *event: events.append(event),
        )
        choice = next(output["decisions"]["service"] for stage, _, output in events
                      if stage == "service" and output is not None)
        self.assertEqual(choice["choice"], "light.turn_off")
        self.assertEqual(choice["effective_choice"], "unsure")
        self.assertEqual(choice["minimum_confidence"], 0.95)
        self.assertEqual(result["route"], "unsure")
        self.assertNotIn("action_metadata", result)
        failure = events[-1][2]
        self.assertEqual(events[-1][0], "failure")
        self.assertEqual(failure, result["failures"][0])
        self.assertIn("90%", failure["reason"])
        self.assertIn("95%", failure["reason"])
        self.assertEqual(failure["stage"], "service")
        self.assertEqual(failure["device_command"], "not_sent")


    async def test_trace_retains_failed_answer_timing_and_cause(self):
        events = []
        result, _ = await self.run_agent([
            {"service": "query_state"},
            {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"}, {"target": "area:office"},
        ], answers=[ModelError("Model endpoint returned HTTP 401.")], trace=lambda *event: events.append(event))
        output = next(output for stage, _, output in events if stage == "answer" and output is not None)
        self.assertEqual(output["status"], "error")
        self.assertEqual(output["error"], result["answer_error"])
        self.assertIn("duration_ms", output)
        failure = events[-1][2]
        self.assertEqual(failure["stage"], "answer")
        self.assertEqual(failure["http_status"], 401)
        self.assertIn("API key", failure["suggestion"])
        self.assertIn("Smart-home answer", failure["suggestion"])


    async def test_trace_retains_decider_failure_without_running_later_stages(self):
        events = []
        models = ScriptedModels([])
        models.decide = AsyncMock(side_effect=ModelError("Model connection failed."))
        agent = AgentEngine(normalize_settings({}), models, trace=lambda *event: events.append(event))
        with self.assertRaisesRegex(ModelError, "connection failed"):
            await agent.process("test", "de", SNAPSHOT)
        self.assertEqual([stage for stage, _, _ in events], ["service", "service", "failure", "failure"])
        self.assertEqual(events[1][2]["status"], "error")
        self.assertIn("duration_ms", events[1][2])
        self.assertIn("could not be reached", events[-1][2]["reason"])


    async def test_answer_character_limit_reaches_stage_debug_and_failure_summary_with_effective_settings(self):
        settings = normalize_settings({
            "decider": {"model": "decider", "max_input_chars": 150000},
            "answer": {"model": "answer-model", "max_input_chars": 100},
        })
        events = []
        session = Mock()
        models = ScriptedModels([
            {"service": "query_state"}, {"area": "all_areas"},
            {"domain": "light"}, {"query_type": "complex"}, {"target": "all_areas"},
        ])
        models.chat = ModelClient(session).chat
        result = await AgentEngine(settings, models, trace=lambda *event: events.append(event)).process(
            "sind noch fenster im haus offen?", "de", SNAPSHOT,
        )
        output = next(output for stage, _, output in events if stage == "answer" and output is not None)
        failure = result["failures"][0]
        self.assertEqual(output["max_input_chars"], 100)
        self.assertEqual(failure["max_input_chars"], 100)
        self.assertEqual(failure["input_chars"], output["input_chars"])
        self.assertGreater(failure["input_chars"], 100)
        self.assertEqual(failure["settings_section"], "Smart-home answer")
        self.assertEqual(failure["model"], "answer-model")
        self.assertIn(f"{failure['input_chars']:,} characters", failure["reason"])
        self.assertIn("100 characters", failure["reason"])
        arguments = next(arguments for stage, arguments, output in events if stage == "answer" and output is None)
        payload = {key: arguments[key] for key in ("model", "messages", "max_tokens", "temperature")}
        self.assertEqual(len(json.dumps(payload, ensure_ascii=False)), failure["input_chars"])
        session.post.assert_not_called()


    async def test_decider_character_limit_reaches_the_stage_and_final_failure_summary(self):
        settings = normalize_settings({"decider": {"model": "local-decider", "max_input_chars": 100}})
        session = Mock()
        events = []
        agent = AgentEngine(settings, ModelClient(session), trace=lambda *event: events.append(event))
        with self.assertRaises(ModelError):
            await agent.process("sind fenster im haus offen?", "de", SNAPSHOT)
        output = next(output for stage, _, output in events if stage == "service" and output is not None)
        failure = events[-1][2]
        self.assertEqual(failure["settings_section"], "Decider")
        self.assertEqual(failure["model"], "local-decider")
        self.assertEqual(failure["input_chars"], output["input_chars"])
        self.assertEqual(failure["max_input_chars"], 100)
        session.post.assert_not_called()


    async def test_trace_does_not_include_model_credentials_or_prompt_templates(self):
        events = []
        settings = normalize_settings({"decider": {"api_key": "secret-decider", "service_prompt": "private-template"},
                                       "answer": {"api_key": "secret-answer"}})
        await self.run_agent([
            {"service": "query_state"},
            {"area": "area:office"}, {"target": "area:office"},
        ], settings=settings, trace=lambda *event: events.append(event))
        serialized = json.dumps(events)
        for secret in ("secret-decider", "secret-answer", "private-template"):
            self.assertNotIn(secret, serialized)

    async def test_answer_debug_shows_the_exact_messages_and_settings_sent_to_the_model(self):
        history = [{"role": "user", "content": "Previous question"}, {"role": "assistant", "content": "Previous answer"}]
        settings = normalize_settings({"history_enabled": True, "answer": {
            "prompt": "Answer rules", "api_key": "private-api-key", "temperature": 0.3,
        }})
        events = []
        _, models = await self.run_agent([
            {"service": "query_state"}, {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"}, {"target": "entity:light.desk"},
        ], settings=settings, history=history, trace=lambda *event: events.append(event))
        arguments = next(arguments for stage, arguments, output in events if stage == "answer" and output is None)
        self.assertEqual(arguments["messages"], models.chat_requests[0][1])
        self.assertIn("Answer rules", arguments["messages"][0]["content"])
        self.assertIn("Previous answer", arguments["messages"][0]["content"])
        self.assertEqual(arguments["temperature"], 0.3)
        self.assertEqual(arguments["max_input_chars"], settings["answer"]["max_input_chars"])
        self.assertNotIn("private-api-key", json.dumps(arguments))


    async def test_state_query_bypasses_area_or_entities_scope_decision(self):
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "area:office"}, {"domain": "light"},
            {"query_type": "complex"}, {"targets": ["light.desk"]},
        ])
        self.assertEqual(len(models.requests), 5)
        self.assertNotIn("target_scope", result["decisions"])
        self.assertEqual(set(models.requests[-1][2]), {"query_targets"})


    async def test_query_uses_cached_state_and_does_not_execute(self):
        result, models = await self.run_agent([
            {"service": "query_state"},
            {"area": "area:office"}, {"domain": "sensor"}, {"query_type": "complex"}, {"target": "entity:sensor.office"},
        ], answers=["Es sind 22 Grad."])

        self.assertEqual(result["state_info"][0]["state"], "22")
        facts = json.loads(models.chat_requests[0][1][1]["content"])
        self.assertEqual(facts["entities"][0]["attributes"]["unit_of_measurement"], "°C")
        self.assertEqual(result["answer"], "Es sind 22 Grad.")

    async def test_unavailable_query_omits_stale_attributes(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["entities"][-1]["state"] = "unavailable"
        result, _ = await self.run_agent([
            {"service": "query_state"},
            {"area": "all_areas"}, {"domain": "sensor"}, {"query_type": "complex"},
            {"target": "entity:sensor.office"},
        ], snapshot=snapshot)
        self.assertEqual(result["state_info"][0]["attributes"], {})

    async def test_unsure_at_each_stage_bypasses_chat_and_execution(self):
        cases = [
            ([{"service": "unsure"}], "service"),
            ([{"service": "query_state"}, {"area": "unsure"}], "area"),
            ([{"service": "query_state"}, {"area": "all_areas"}, {"domain": "light"},
              {"query_type": "complex"}, {"targets": []}], "target"),
        ]
        for choices, expected_stage in cases:
            with self.subTest(stage=expected_stage):
                result, models = await self.run_agent(choices)
                self.assertEqual(result["route"], "unsure")
                self.assertEqual(result["answer"], DEFAULTS["unsure_reply"])
                self.assertEqual(result["failures"][0]["stage"], expected_stage)
                self.assertNotIn("action_metadata", result)
                self.assertFalse(models.chat_requests)


    async def test_disabled_general_returns_fixed_unsure(self):
        result, models = await self.run_agent([{"service": "not-smarthome-related"}])
        self.assertEqual(result["answer"], "ich bin mir nicht sicher")
        self.assertIn("general model is disabled", result["failures"][0]["reason"])
        self.assertFalse(models.chat_requests)


    async def test_enabled_general_uses_its_own_model_and_original_request(self):
        settings = normalize_settings({"general": {"enable": True, "model": "smart-model"}})
        result, models = await self.run_agent(
            [{"service": "not-smarthome-related"}], settings=settings, answers=["330 Meter."],
        )
        self.assertEqual(result["route"], "not-smarthome-related")
        self.assertEqual(len(models.requests), 1)
        self.assertEqual(models.chat_requests[0][0]["model"], "smart-model")
        self.assertEqual(models.chat_requests[0][1][1]["content"], "test request")
        self.assertIn("Assist pipeline language: de", models.chat_requests[0][1][0]["content"])
        self.assertEqual(result["answer"], "330 Meter.")


    async def test_confidence_threshold_stops_action(self):
        settings = normalize_settings({"decider": {"service_confidence": 0.95}})
        result, models = await self.run_agent([{"service": "light.turn_off"}], settings=settings)
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(len(models.requests), 1)


    async def test_room_query_selects_one_available_domain_before_entity_choice(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["entities"].extend([
            entity("binary_sensor.office_window", "Office window", device_class="window"),
            entity("fan.bedroom", "Fan", area="bedroom"),
        ])
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "area:office"},
            {"domain": "binary_sensor"}, {"query_type": "complex"}, {"targets": ["binary_sensor.office_window"]},
        ], snapshot=snapshot, prompt="sind fenster im buero offen?")
        self.assertEqual(set(models.requests[0][2]), {"service"})
        self.assertEqual(set(models.requests[2][2]), {"domain"})
        self.assertEqual(set(models.requests[2][2]["domain"]["criteria"]), {"binary_sensor", "light", "sensor", "unsure"})
        self.assertNotIn("entities", models.requests[2][1])
        self.assertNotIn("targets", models.requests[2][1])
        targets = query_candidates(models.requests[4])
        self.assertEqual({target["entity_id"] for target in targets}, {"binary_sensor.office_window"})
        self.assertEqual(models.requests[4][1]["domain"], "binary_sensor")
        self.assertEqual(result["domain"], "binary_sensor")
        self.assertEqual(result["state_info"][0]["entity_id"], "binary_sensor.office_window")


    async def test_room_query_answer_only_receives_the_selected_domain(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["entities"].append(entity("binary_sensor.office_window", "Window", device_class="window"))
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"}, {"targets": "all"},
        ], snapshot=snapshot)
        facts = json.loads(models.chat_requests[0][1][1]["content"])
        self.assertEqual({item["entity_id"] for item in facts["entities"]}, {"light.desk", "light.ceiling"})
        self.assertEqual(facts["entities"], result["state_info"])
        self.assertNotIn("target", facts)
        self.assertNotIn("state_info", facts)
        self.assertEqual(result["domain"], "light")
        self.assertEqual(result["domain_filter"]["entities_before"], 4)
        self.assertEqual(result["domain_filter"]["entities_after"], 2)


    async def test_query_choice_receives_current_states_for_state_filtered_questions(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["entities"][1]["state"] = "off"
        snapshot["entities"][2]["state"] = "unavailable"
        original = copy.deepcopy(snapshot)
        events = []
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "all_areas"}, {"domain": "light"},
            {"query_type": "complex"}, {"targets": ["light.desk", "light.bed"]},
        ], snapshot=snapshot, prompt="Welche Lichter im Haus sind an?", trace=lambda *event: events.append(event))
        targets = query_candidates(models.requests[-1])
        self.assertEqual({target["entity_id"]: target["state"] for target in targets},
                         {"light.desk": "on", "light.ceiling": "off", "light.bed": "unavailable"})
        choice = json.loads(models.requests[-1][2]["query_targets"]["criteria"]["entity:light.desk"])
        self.assertEqual(choice["entity_id"], "light.desk")
        self.assertEqual(choice["state"], "on")
        self.assertNotIn("targets", models.requests[-1][1])
        self.assertTrue(all("attributes" not in target for target in targets))
        self.assertEqual(snapshot, original)
        trace_input = next(arguments for stage, arguments, output in events if stage == "target" and output is None)
        self.assertEqual(trace_input["candidate_counts"], {"query_targets": len(targets) + 1})
        facts = json.loads(models.chat_requests[0][1][1]["content"])
        self.assertEqual([(item["entity_id"], item["state"]) for item in facts["entities"]],
                         [("light.desk", "on"), ("light.bed", "unavailable")])
        self.assertEqual(result["target_selection"]["selected_entity_ids"], ["light.desk", "light.bed"])


    async def test_unspecified_room_query_uses_all_areas_and_filters_domains_before_targets(self):
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "all_areas"},
            {"domain": "light"}, {"query_type": "complex"}, {"target": "entity:light.bed"},
        ], prompt="Is the bed lamp on?")
        area_question = models.requests[1][2]["area"]
        self.assertNotIn("no_area", area_question["criteria"])
        self.assertIn("all_areas", area_question["criteria"])
        self.assertIn("when no specific room is identified", area_question["instructions"])
        self.assertNotIn("Choose no_area when no room is identified.", area_question["instructions"])
        self.assertEqual(set(models.requests[2][2]), {"domain"})
        targets = {target["entity_id"] for target in query_candidates(models.requests[4])}
        self.assertIn("light.bed", targets)
        self.assertIn("light.desk", targets)
        self.assertNotIn("sensor.office", targets)
        self.assertEqual(result["domain"], "light")
        self.assertEqual([item["entity_id"] for item in result["state_info"]], ["light.bed"])
        self.assertEqual(len(models.requests), 5)


    async def test_whole_home_query_includes_all_rooms_and_unassigned_entities(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["entities"].extend([
            entity("binary_sensor.office_window", "Window", device_class="window"),
            entity("binary_sensor.bedroom_window", "Window", area="bedroom", device_class="window"),
            entity("binary_sensor.unassigned_window", "Unassigned window", area=None, device_class="window"),
        ])
        settings = normalize_settings({"decider": {"area_prompt": "Custom area rules", "query_target_prompt": "Custom target rules"}})
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "all_areas"},
            {"domain": "binary_sensor"}, {"query_type": "complex"},
            {"target": "all_areas"},
        ], snapshot=snapshot, settings=settings, prompt="are there any windows open in the house?")
        area_question = models.requests[1][2]["area"]
        targets = {target["entity_id"] for target in query_candidates(models.requests[4])}
        self.assertIn("all_areas", area_question["criteria"])
        self.assertNotIn("no_area", area_question["criteria"])
        self.assertIn("Custom area rules", area_question["instructions"])
        self.assertIn("choose all_areas", area_question["instructions"])
        self.assertEqual(models.requests[4][2]["query_targets"]["instructions"], settings["decider"]["query_target_prompt"])
        self.assertIn("binary_sensor.unassigned_window", targets)
        self.assertNotIn("light.desk", targets)
        self.assertNotIn("sensor.office", targets)
        self.assertEqual(result["domain_filter"]["kept_domains"], ["binary_sensor"])
        self.assertEqual(result["domain_filter"]["entities_before"], len(snapshot["entities"]))
        self.assertEqual(result["domain_filter"]["entities_after"], 3)
        self.assertEqual(result["target"]["type"], "entities")
        facts = json.loads(models.chat_requests[0][1][1]["content"])
        self.assertEqual({item["entity_id"] for item in facts["entities"]},
                         {item["entity_id"] for item in snapshot["entities"] if item["domain"] == "binary_sensor"})
        self.assertNotIn("domain_filter", facts)
        windows = {item["entity_id"]: item for item in facts["entities"] if item["domain"] == "binary_sensor"}
        self.assertEqual(windows["binary_sensor.office_window"]["area"], "Office")
        self.assertEqual(windows["binary_sensor.bedroom_window"]["area"], "Bedroom")
        self.assertIsNone(windows["binary_sensor.unassigned_window"]["area"])
        self.assertEqual(settings["decider"]["query_target_prompt"], "Custom target rules")


    async def test_named_room_query_skips_domain_filter_and_offers_only_individual_entities(self):
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "area:office"}, {"domain": "sensor"}, {"query_type": "complex"}, {"target": "entity:sensor.office"},
        ])
        self.assertTrue(all(target["type"] == "entity" for target in query_candidates(models.requests[4])))
        self.assertEqual(len(models.requests), 5)
        self.assertIn("domain", result["decisions"])


    async def test_whole_home_query_without_exposed_entities_stops_before_target_selection(self):
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "all_areas"},
        ], snapshot={"entities": [], "services": {}})
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["failures"][0]["stage"], "target")
        self.assertEqual(len(models.requests), 2)
        self.assertFalse(models.chat_requests)


    async def test_query_keeps_multiple_window_contacts_and_excludes_other_entity_states_from_answer(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["entities"].extend([
            entity("binary_sensor.left_window", "Left window", state="on", device_class="window"),
            entity("binary_sensor.right_window", "Right window", state="off", device_class="window"),
            entity("binary_sensor.motion", "Motion", device_class="motion"),
        ])
        events = []
        result, _ = await self.run_agent([
            {"service": "query_state"}, {"area": "area:office"},
            {"domain": "binary_sensor"}, {"query_type": "complex"}, {"targets": ["binary_sensor.left_window", "binary_sensor.right_window"]},
        ], snapshot=snapshot, trace=lambda *event: events.append(event), prompt="Sind Fenster im Büro offen?")
        self.assertEqual(result["target"]["type"], "entities")
        arguments = next(arguments for stage, arguments, output in events if stage == "answer" and output is None)
        states = json.loads(arguments["messages"][1]["content"])["entities"]
        self.assertEqual([state["entity_id"] for state in states], ["binary_sensor.left_window", "binary_sensor.right_window"])
        self.assertEqual([state["state"] for state in states], ["on", "off"])
        self.assertEqual([state["name"] for state in states], ["Left window", "Right window"])
        output = next(output for stage, _, output in events if stage == "target" and output is not None)
        self.assertEqual(output["entities_before"], 3)
        self.assertEqual(output["entities_after"], 2)
        self.assertEqual(output["decisions"]["query_targets"]["probabilities"]["entity:binary_sensor.left_window"], 0.5)


    async def test_query_choice_uses_all_probabilities_and_editable_prompt(self):
        events = []
        settings = normalize_settings({"decider": {"query_target_prompt": "private-target-task",
                                                  "query_target_probability_threshold": 0.15}})
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"},
            {"query_targets": {"type": "choice", "choice": "none", "confidence": 0,
                               "probabilities": {"entity:light.desk": 0.25, "entity:light.ceiling": 0.2, "none": 0.55}}},
        ], settings=settings, trace=lambda *event: events.append(event))
        self.assertEqual(models.requests[-1][2]["query_targets"]["instructions"], "private-target-task")
        self.assertNotIn("private-target-task", json.dumps(events))
        self.assertEqual(result["target_selection"]["entities_after"], 2)
        output = next(output for stage, _, output in events if stage == "target" and output)
        self.assertEqual(output["probability_threshold"], 0.15)
        self.assertEqual(output["decisions"]["query_targets"]["confidence"], 0)
        self.assertNotIn("minimum_confidence", output["decisions"]["query_targets"])


    async def test_query_choice_threshold_is_strict_and_empty_selection_stops(self):
        settings = normalize_settings({"decider": {"query_target_probability_threshold": 0.3}})
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"},
            {"query_targets": {"type": "choice", "choice": "none", "confidence": 0.9,
                               "probabilities": {"entity:light.desk": 0.3, "entity:light.ceiling": 0.2, "none": 0.5}}},
        ], settings=settings)
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["target_selection"]["selected_entity_ids"], [])
        self.assertFalse(models.chat_requests)


    async def test_54_query_candidates_use_one_choice_and_retain_multiple_matching_states(self):
        snapshot = {"services": {}, "entities": [
            entity(f"sensor.value_{index}", f"Room value {index}", area=None,
                   state="22" if index < 18 else "50", device_class="temperature" if index < 18 else "humidity",
                   unit_of_measurement="°C" if index < 18 else "%") for index in range(54)
        ]}
        probabilities = {f"entity:sensor.value_{index}": 0.05 if index < 18 else 0.0025 for index in range(54)}
        probabilities["none"] = 0.01
        settings = normalize_settings({"decider": {"target_filter_confidence": 1, "query_target_keep_uncertain": True}})
        events = []
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "all_areas"}, {"domain": "sensor"}, {"query_type": "complex"},
            {"query_targets": {"type": "choice", "choice": "entity:sensor.value_0", "confidence": 0.001,
                               "probabilities": probabilities}},
        ], snapshot=snapshot, settings=settings, prompt="wie warm ist es?", trace=lambda *event: events.append(event))
        self.assertEqual(len(models.requests), 5)
        self.assertEqual(set(models.requests[-1][2]), {"query_targets"})
        self.assertEqual(len(models.requests[-1][2]["query_targets"]["criteria"]), 55)
        self.assertNotIn("targets", models.requests[-1][1])
        self.assertEqual(result["target_selection"]["selected_entity_ids"], [f"sensor.value_{index}" for index in range(18)])
        self.assertEqual([item["entity_id"] for item in result["state_info"]], result["target"]["entity_ids"])
        facts = json.loads(models.chat_requests[0][1][1]["content"])
        self.assertEqual(len(facts["entities"]), 18)
        self.assertTrue(all(item["attributes"]["unit_of_measurement"] == "°C" for item in facts["entities"]))
        output = next(output for stage, _, output in events if stage == "target" and output)
        self.assertEqual(output["entities_before"], 54)
        self.assertEqual(output["entities_after"], 18)
        self.assertEqual(output["decisions"]["query_targets"]["probability_threshold"], 0.02)


    async def test_malformed_query_choice_stops_before_answer(self):
        events = []
        with self.assertRaises(ModelError):
            await self.run_agent([
                {"service": "query_state"}, {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"},
                {"query_targets": {"type": "choice", "choice": "entity:light.desk", "confidence": 0.9,
                                   "probabilities": {"entity:light.desk": 1}}},
            ], trace=lambda *event: events.append(event))
        self.assertFalse(any(stage in ("execution", "answer") for stage, _, _ in events))
        self.assertEqual(events[-1][2]["stage"], "target")


    async def test_whole_home_domain_selection_is_one_question_and_reports_reduction(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["entities"].append(entity("binary_sensor.window", "Window", device_class="window"))
        history = [{"role": "user", "content": "What about the windows?"}]
        settings = normalize_settings({"history_enabled": True, "decider": {"domain_prompt": "private-domain-prompt"}})
        events = []
        result, models = await self.run_agent([
            {"service": "query_state"}, {"area": "all_areas"}, {"domain": "binary_sensor"}, {"query_type": "complex"}, {"targets": "all"},
        ], snapshot=snapshot, settings=settings, history=history, trace=lambda *event: events.append(event))
        _, state, questions = models.requests[2]
        self.assertEqual(state["history"], history)
        self.assertEqual(state["language"], "de")
        self.assertEqual(set(questions), {"domain"})
        self.assertEqual(set(questions["domain"]["criteria"]), {"binary_sensor", "sensor", "light", "unsure"})
        self.assertIn("private-domain-prompt", questions["domain"]["instructions"])
        self.assertNotIn("private-domain-prompt", json.dumps(events))
        targets = {target["entity_id"] for target in query_candidates(models.requests[4])}
        self.assertEqual(targets, {"binary_sensor.window"})
        output = next(output for stage, _, output in events if stage == "domain" and output is not None)
        self.assertEqual(output["selected_domain"], "binary_sensor")
        self.assertEqual(output["entities_before"], 5)
        self.assertEqual(output["entities_after"], 1)
        self.assertEqual(output["decisions"]["domain"]["effective_choice"], "binary_sensor")
        self.assertEqual(output["decisions"]["domain"]["minimum_confidence"], 0.5)
        self.assertNotIn("keep_uncertain", output["decisions"]["domain"])
        self.assertIn("domain", result["usage"])


    async def test_uncertain_domain_selection_stops_without_restoring_all_domains(self):
        for choice, threshold in (("unsure", 0.5), ("light", 0.95)):
            with self.subTest(choice=choice, threshold=threshold):
                settings = normalize_settings({"decider": {"domain_confidence": threshold}})
                events = []
                result, models = await self.run_agent([
                    {"service": "query_state"}, {"area": "all_areas"}, {"domain": choice}, {"query_type": "complex"},
                ], settings=settings, trace=lambda *event: events.append(event))
                self.assertEqual(result["route"], "unsure")
                self.assertEqual(result["domain_filter"]["kept_domains"], [])
                self.assertEqual(result["domain_filter"]["entities_after"], 0)
                self.assertEqual(len(models.requests), 3)
                self.assertFalse(models.chat_requests)
                self.assertEqual(result["failures"][0]["stage"], "domain")


    async def test_malformed_domain_selection_stops_before_building_the_target_request(self):
        events = []
        with self.assertRaises(ModelError):
            await self.run_agent([
                {"service": "query_state"}, {"area": "all_areas"}, {"domain": "invented"}, {"query_type": "complex"},
            ], trace=lambda *event: events.append(event))
        self.assertFalse(any(stage == "target" for stage, _, _ in events))
        output = next(output for stage, _, output in events if stage == "domain" and output is not None)
        self.assertEqual(output["status"], "error")
        self.assertEqual(events[-1][2]["stage"], "domain")


    async def test_query_answer_failure_preserves_state_and_error_details(self):
        result, models = await self.run_agent([
            {"service": "query_state"},
            {"area": "area:office"}, {"domain": "light"}, {"query_type": "complex"}, {"target": "entity:light.desk"},
        ], answers=[ModelError("Model reply reached its output token limit.")])
        self.assertEqual(result["answer"], DEFAULTS["error_reply"])
        self.assertEqual(result["answer_error"], "Model reply reached its output token limit.")
        self.assertEqual(result["state_info"][0]["state"], "on")
        self.assertIn("answer", result["timings_ms"])
        self.assertEqual(len(models.chat_requests), 1)


    async def test_invalid_decision_after_response_still_emits_a_failure_summary(self):
        events = []
        models = ScriptedModels([])
        models.decide = AsyncMock(return_value={"answers": {}})
        agent = AgentEngine(normalize_settings({}), models, trace=lambda *event: events.append(event))
        with self.assertRaises(ModelError):
            await agent.process("test", "de", SNAPSHOT)
        self.assertEqual(events[-1][0], "failure")
        self.assertEqual(events[-1][2]["stage"], "service")
        self.assertIn("invalid decision", events[-1][2]["reason"])


class IntegrationSettingsTests(unittest.TestCase):
    def local_settings(self):
        return normalize_settings({
            "decider": {"url": "http://server:8321/v1/systemone", "api_key": ""},
            "answer": {"url": "http://server:8322/v1/chat/completions", "model": "small-model"},
        })

    def test_local_models_and_disabled_general_need_no_keys(self):
        settings = validate_settings(self.local_settings())
        self.assertFalse(settings["general"]["enable"])
        self.assertEqual(settings["decider"]["api_key"], "")

    def test_parameter_probability_setting_is_validated_independently_of_confidence(self):
        self.assertEqual(self.local_settings()["decider"]["parameter_probability_threshold"], 0.3)
        for value in (-0.1, 1.1, True, float("nan")):
            settings = self.local_settings()
            settings["decider"]["parameter_probability_threshold"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_settings(settings)
        reset = normalize_settings({"decider": {"parameter_prompt": " "}})
        self.assertEqual(reset["decider"]["parameter_prompt"], DEFAULTS["decider"]["parameter_prompt"])


    def test_removed_settings_are_ignored_and_new_text_cutoff_is_independent(self):
        removed = {"target_filter_prompt": "old rules", "target_filter_confidence": 1,
                   "target_prompt": "old target", "target_confidence": 1, "numeric_prompt": "old numeric",
                   "numeric_confidence": 1, "temperature_min": 27, "position_step": 0.5,
                   "query_target_keep_uncertain": True, "action_target_keep_uncertain": True,
                   "text_confidence": 0.9, "text_keep_uncertain": True}
        saved = normalize_settings({"decider": removed})
        self.assertFalse(set(removed) & set(saved["decider"]))
        self.assertEqual(saved["decider"]["text_probability_threshold"], 0.02)
        self.assertEqual(normalize_settings({"decider": {"text_probability_threshold": 0.05}})["decider"]["text_probability_threshold"], 0.05)

    def test_old_stock_text_prompts_upgrade_and_custom_prompts_are_preserved(self):
        from custom_components.glow_assistant.text_values import LEGACY_PROMPTS

        for old in LEGACY_PROMPTS:
            saved = normalize_settings({"decider": {"text_prompt": old, "message_prompt": old}})["decider"]
            self.assertEqual(saved["text_prompt"], DEFAULTS["decider"]["text_prompt"])
            self.assertEqual(saved["message_prompt"], DEFAULTS["decider"]["message_prompt"])
        custom = normalize_settings({"decider": {"text_prompt": "My item rules", "message_prompt": "My message rules"}})["decider"]
        self.assertEqual(custom["text_prompt"], "My item rules")
        self.assertEqual(custom["message_prompt"], "My message rules")
        shared = normalize_settings({"decider": {"text_prompt": "My shared rules"}})["decider"]
        self.assertEqual(shared["message_prompt"], "My shared rules")

    def test_stock_scope_prompt_upgrades_and_custom_prompt_is_preserved(self):
        from custom_components.glow_assistant.const import (
            LEGACY_ACTION_TARGET_SCOPE_PROMPT,
            LEGACY_TARGET_SCOPE_PROMPT,
            LEGACY_UNSCOPED_TARGET_SCOPE_PROMPT,
            TARGET_SCOPE_PROMPT,
        )

        for previous in (LEGACY_TARGET_SCOPE_PROMPT, LEGACY_ACTION_TARGET_SCOPE_PROMPT, LEGACY_UNSCOPED_TARGET_SCOPE_PROMPT):
            upgraded = normalize_settings({"decider": {"target_scope_prompt": previous}})
            self.assertEqual(upgraded["decider"]["target_scope_prompt"], TARGET_SCOPE_PROMPT)
        custom = normalize_settings({"decider": {"target_scope_prompt": "Custom scope rules"}})
        self.assertEqual(custom["decider"]["target_scope_prompt"], "Custom scope rules")

    def test_domain_selection_settings_are_preserved_and_blank_prompt_restores_default(self):
        settings = normalize_settings({"decider": {"domain_prompt": "Custom rules", "domain_confidence": 0.2}})
        self.assertEqual(settings["decider"]["domain_prompt"], "Custom rules")
        self.assertEqual(settings["decider"]["domain_confidence"], 0.2)
        reset = normalize_settings({"decider": {"domain_prompt": " "}})
        self.assertEqual(reset["decider"]["domain_prompt"], DEFAULTS["decider"]["domain_prompt"])


    def test_domain_selection_migrates_the_previous_threshold_but_not_keep_drop_prompts(self):
        settings = normalize_settings({"decider": {"domain_filter_prompt": "Keep/drop rules", "domain_filter_confidence": 0.3,
                                                  "domain_filter_keep_uncertain": True}})
        self.assertEqual(settings["decider"]["domain_prompt"], DEFAULTS["decider"]["domain_prompt"])
        self.assertEqual(settings["decider"]["domain_confidence"], 0.3)
        self.assertNotIn("domain_filter_keep_uncertain", settings["decider"])
        custom = normalize_settings({"decider": {"domain_filter_confidence": 0.3, "domain_confidence": 0.7}})
        self.assertEqual(custom["decider"]["domain_confidence"], 0.7)

    def test_http_mcp_settings_are_normalized_without_changing_model_keys(self):
        settings = self.local_settings()
        settings["general"]["mcp_servers"] = [{"name": " Web ", "url": "http://search:8000/mcp", "token": " server-key "}]
        result = validate_settings(settings)
        server = result["general"]["mcp_servers"][0]
        self.assertEqual(server["name"], "Web")
        self.assertEqual(server["token"], "server-key")
        self.assertFalse(server["enable"])
        self.assertEqual(server["timeout"], 30)
        self.assertEqual(server["tools"], [])
        self.assertEqual(result["general"]["api_key"], "")
        self.assertFalse(result["history_enabled"])

    def test_mcp_settings_reject_stdio_and_invalid_server_definitions(self):
        for definition in (
            {"name": "stdio", "url": "stdio://server"},
            {"name": "command", "command": "run-server"},
            {"name": "bad-url", "url": "file:///server"},
            {"name": "credentials", "url": "https://user:password@server/mcp"},
            {"name": "bad-timeout", "url": "http://server/mcp", "timeout": 0},
            {"name": "bad-tools", "url": "http://server/mcp", "tools": "search"},
        ):
            settings = self.local_settings()
            settings["general"]["mcp_servers"] = [definition]
            with self.subTest(definition=definition), self.assertRaises((ValueError, TypeError)):
                validate_settings(settings)

    def test_history_window_has_no_expiry_setting_and_accepts_an_exact_message_count(self):
        settings = self.local_settings()
        settings.update(history_enabled=True, history_messages=3.0)
        result = validate_settings(settings)
        self.assertEqual(result["history_messages"], 3)
        self.assertNotIn("history_timeout", result)
        for size in (0, 101, 1.5):
            settings["history_messages"] = size
            with self.subTest(size=size), self.assertRaises(ValueError):
                validate_settings(settings)

    def test_legacy_stock_general_prompt_is_updated_but_custom_prompts_are_preserved(self):
        from custom_components.glow_assistant.const import (
            GENERAL_PROMPT,
            LEGACY_GENERAL_PROMPT,
        )

        self.assertEqual(normalize_settings({"general": {"prompt": LEGACY_GENERAL_PROMPT}})["general"]["prompt"], GENERAL_PROMPT)
        self.assertEqual(normalize_settings({"general": {"prompt": "My custom rules"}})["general"]["prompt"], "My custom rules")

    def test_stock_action_prompt_is_updated_for_text_but_custom_rules_are_preserved(self):
        from custom_components.glow_assistant.const import (
            LEGACY_SERVICE_PROMPT,
            SERVICE_PROMPT,
        )

        saved = normalize_settings({"decider": {"service_prompt": LEGACY_SERVICE_PROMPT}})
        self.assertEqual(saved["decider"]["service_prompt"], SERVICE_PROMPT)
        self.assertIn("todo.add_item", saved["decider"]["service_prompt"])
        self.assertEqual(saved["decider"]["text_probability_threshold"], 0.02)
        custom = normalize_settings({"decider": {"service_prompt": "Custom action rules", "text_prompt": "Custom extraction rules"}})
        self.assertEqual(custom["decider"]["service_prompt"], "Custom action rules")
        self.assertEqual(custom["decider"]["text_prompt"], "Custom extraction rules")


    def test_ui_numeric_selectors_produce_integer_token_limits(self):
        settings = self.local_settings()
        settings["answer"]["max_tokens"] = 512.0
        settings["decider"]["max_input_chars"] = 100000.0
        saved = validate_settings(settings)
        self.assertIsInstance(saved["answer"]["max_tokens"], int)
        self.assertIsInstance(saved["decider"]["max_input_chars"], int)

    def test_prompt_reset_and_customization_survive_normalization(self):
        settings = normalize_settings({"decider": {"query_target_prompt": "My rules", "area_prompt": " ", "target_scope_prompt": " "}})
        self.assertEqual(settings["decider"]["query_target_prompt"], "My rules")
        self.assertEqual(settings["decider"]["area_prompt"], DEFAULTS["decider"]["area_prompt"])
        self.assertEqual(settings["decider"]["target_scope_prompt"], DEFAULTS["decider"]["target_scope_prompt"])
        self.assertNotEqual(id(settings["decider"]), id(DEFAULTS["decider"]))
        for field in ("query_type_prompt", "binary_answer_prompt"):
            self.assertEqual(normalize_settings({"decider": {field: " "}})["decider"][field], DEFAULTS["decider"][field])
            self.assertEqual(normalize_settings({"decider": {field: "Custom rules"}})["decider"][field], "Custom rules")

    def test_invalid_ranges_and_models_are_rejected(self):
        for group, field, value in (
            ("decider", "service_confidence", 1.1), ("decider", "url", "localhost:8321"),
            ("decider", "text_probability_threshold", -0.1), ("decider", "text_probability_threshold", 1.1),
            ("decider", "domain_confidence", -0.1), ("decider", "domain_confidence", 1.1),
            ("decider", "query_type_confidence", -0.1), ("decider", "query_type_confidence", 1.1),
            ("decider", "binary_answer_confidence", -0.1), ("decider", "binary_answer_confidence", 1.1),
            ("decider", "target_scope_confidence", -0.1), ("decider", "target_scope_confidence", 1.1),
            ("decider", "action_area_probability_threshold", -0.1), ("decider", "action_area_probability_threshold", 1.1),
            ("decider", "action_entity_probability_threshold", -0.1), ("decider", "action_entity_probability_threshold", 1.1),
            ("answer", "model", ""), ("answer", "max_tokens", 0),
            ("answer", "max_input_chars", float("inf")), ("answer", "temperature", True),
        ):
            settings = self.local_settings()
            settings[group][field] = value
            with self.subTest(group=group, field=field), self.assertRaises(ValueError):
                validate_settings(settings)
        settings = self.local_settings()
        settings["general"]["enable"] = True
        with self.assertRaisesRegex(ValueError, "general"):
            validate_settings(settings)

    def test_cloud_keys_are_required_per_model_without_cross_provider_fallback(self):
        settings = self.local_settings()
        settings["answer"]["url"] = "https://openrouter.ai/api/v1/chat/completions"
        settings["decider"]["api_key"] = "decider-only-key"
        with self.assertRaisesRegex(ValueError, "answer"):
            validate_settings(settings)

    def test_action_catalog_omits_unsupported_required_values(self):
        options = action_options({
            "light.turn_on": {"fields": {}}, "climate.set_temperature": {"fields": {}},
            "climate.set_hvac_mode": {"fields": {"hvac_mode": {"required": True}}},
            "todo.get_items": {"fields": {}, "response": {"optional": False}},
            "cover.set_cover_position": {"fields": {"position": {"required": True}}},
        })
        self.assertIn("light.turn_on", options)
        self.assertIn("light.turn_on:brightness_pct", options)
        self.assertIn("climate.set_temperature:temperature", options)
        self.assertNotIn("climate.set_temperature", options)
        self.assertNotIn("climate.set_hvac_mode", options)
        self.assertNotIn("todo.get_items", options)
        self.assertIn("cover.set_cover_position:position", options)
        self.assertNotIn("cover.set_cover_position", options)

    def test_text_catalog_is_limited_to_the_three_explicit_action_field_pairs(self):
        services = {
            name: {"fields": {parameter: {"required": name != "assist_satellite.announce"}}}
            for name, (parameter, _) in TEXT_ACTIONS.items()
        }
        services.update({
            "input_text.set_value": {"fields": {"value": {"required": True, "selector": {"text": {}}}}},
            "todo.remove_item": {"fields": {"item": {"required": True}}},
            "todo.update_item": {"fields": {"item": {"required": True}}},
        })
        options = action_options(services)
        self.assertEqual(set(options), set(TEXT_ACTIONS))
        for name, (parameter, _) in TEXT_ACTIONS.items():
            self.assertEqual(options[name]["parameter"], parameter)
            self.assertEqual(options[name]["parameter_type"], "text")

    def test_text_action_is_not_offered_when_another_required_field_is_present(self):
        self.assertEqual(action_options({
            "notify.send_message": {"fields": {"message": {"required": True}, "title": {"required": True}}},
            "todo.add_item": {"fields": {}},
        }), {})

    def test_legacy_other_and_malformed_responses_are_errors(self):
        response = {"answers": {"target": {"type": "choice", "choice": "other", "confidence": 1, "probabilities": {"unsure": 1}}}}
        with self.assertRaises(ModelError):
            read_choice(response, "target", {}, 0)

    def test_translation_fields_match_settings_blocks(self):
        root = Path(__file__).parent / "custom_components/glow_assistant"
        strings = json.loads((root / "strings.json").read_text())
        self.assertEqual(strings, json.loads((root / "translations/en.json").read_text()))
        for group in ("decider", "answer", "general"):
            labels = strings["config"]["step"]["user"]["sections"][group]["data"]
            self.assertEqual(set(labels), set(DEFAULTS[group]) | {"test_connection"})
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertTrue(manifest["config_flow"])
        self.assertEqual(manifest["domain"], "glow_assistant")


if __name__ == "__main__":
    unittest.main()
