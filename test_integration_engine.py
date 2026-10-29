import copy
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.engine import (
    ActionError,
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
            if question["type"] == "score":
                score = choices[name]
                probabilities = {str(index): 0 for index in range(len(question["criteria"]))}
                lower = int(score)
                probabilities[str(lower)] = 1 - (score - lower)
                if score != lower:
                    probabilities[str(lower + 1)] = score - lower
                answers[name] = {"type": "score", "score": score, "confidence": self.confidence, "probabilities": probabilities}
            else:
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
    async def run_agent(self, choices, snapshot=SNAPSHOT, settings=None, answers=None, executor=None, trace=None, history=None, language="de", prompt="test request"):
        models = ScriptedModels(choices, answers)
        execute = executor or AsyncMock()
        agent = AgentEngine(settings or normalize_settings({}), models, execute, trace=trace, history=history)
        result = await agent.process(prompt, language, snapshot)
        return result, models, execute

    async def test_pipeline_language_reaches_every_decider_and_the_answer_model(self):
        settings = normalize_settings({"answer": {"prompt": "Custom answer rules"}})
        _, models, _ = await self.run_agent([
            {"service": "query_state", "domain": "light"},
            {"area": "area:office"}, {"target": "entity:light.desk"},
        ], settings=settings, language="fr-CH")
        for _, state, _ in models.requests:
            self.assertEqual(state["language"], "fr-CH")
        system = models.chat_requests[0][1][0]["content"]
        self.assertIn("Custom answer rules", system)
        self.assertIn("Assist pipeline language: fr-CH", system)
        self.assertIn("Write the spoken reply in this language", system)

    async def test_history_is_sent_to_every_decider_and_the_answer_model(self):
        history = [{"role": "user", "content": "Is the office lamp on?"}, {"role": "assistant", "content": "Yes, it is on."}]
        settings = normalize_settings({"history_enabled": True})
        result, models, execute = await self.run_agent([
            {"service": "query_state", "domain": "light"},
            {"area": "area:office"}, {"target": "entity:light.desk"},
        ], settings=settings, history=history)
        self.assertEqual(result["answer"], "Erledigt.")
        for _, state, questions in models.requests:
            self.assertEqual(state["history"], history)
            self.assertTrue(all("resolve references" in item["instructions"] for item in questions.values()))
        self.assertIn(json.dumps(history, ensure_ascii=False), models.chat_requests[0][1][0]["content"])
        self.assertEqual(len(models.requests), 3)
        execute.assert_not_awaited()

    async def test_disabled_history_is_not_forwarded_even_if_supplied(self):
        _, models, _ = await self.run_agent([
            {"service": "query_state", "domain": "light"},
            {"area": "area:office"}, {"target": "entity:light.desk"},
        ], history=[{"role": "user", "content": "private previous message"}])
        self.assertNotIn("private previous message", json.dumps(models.requests))
        self.assertNotIn("private previous message", json.dumps(models.chat_requests))

    async def test_general_history_uses_the_same_plain_message_context(self):
        history = [{"role": "user", "content": "How tall is the Eiffel Tower?"}, {"role": "assistant", "content": "330 metres."}]
        settings = normalize_settings({"history_enabled": True, "general": {"enable": True}})
        _, models, _ = await self.run_agent([{"service": "not-smarthome-related"}], settings=settings, history=history)
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
            result, models, execute = await self.run_agent(
                [{"service": "not-smarthome-related"}], settings=settings, history=history,
            )
        self.assertEqual(result["answer"], "Answer with tools")
        self.assertIn(json.dumps(history, ensure_ascii=False), general.await_args.args[2][0]["content"])
        self.assertIn("Assist pipeline language: de", general.await_args.args[2][0]["content"])
        self.assertEqual(len(models.requests), 1)
        self.assertFalse(models.chat_requests)
        execute.assert_not_awaited()

    async def test_enabled_mcp_servers_are_not_contacted_for_smarthome_or_disabled_general(self):
        from unittest.mock import patch

        settings = normalize_settings({"general": {"enable": False, "mcp_servers": [{"enable": True}]}})
        with patch("custom_components.glow_assistant.engine.general_answer", new_callable=AsyncMock) as general:
            result, _, _ = await self.run_agent([
                {"service": "query_state", "domain": "light"},
                {"area": "area:office"}, {"target": "entity:light.desk"},
            ], settings=settings)
            self.assertEqual(result["route"], "smarthome")
            result, _, _ = await self.run_agent([{"service": "not-smarthome-related"}], settings=settings)
            self.assertEqual(result["route"], "unsure")
        general.assert_not_awaited()

    async def test_trace_reports_each_decision_and_action_without_extra_requests(self):
        events = []
        def trace(stage, arguments, result):
            events.append(copy.deepcopy((stage, arguments, result)))
        result, models, execute = await self.run_agent([
            {"service": "light.turn_off", "domain": "unsure"},
            {"area": "area:office"}, {"target": "area:office"},
        ], trace=trace)
        expected = ["service", "area", "target", "execution", "answer"]
        self.assertEqual([stage for stage, _, output in events if output is None], expected)
        self.assertEqual([stage for stage, _, output in events if output is not None], expected)
        outputs = {stage: output for stage, _, output in events if output is not None}
        choice = outputs["service"]["decisions"]["service"]
        self.assertEqual(choice["choice"], "light.turn_off")
        self.assertEqual(choice["confidence"], 0.9)
        self.assertEqual(choice["selected"]["service"], "light.turn_off")
        self.assertFalse(outputs["service"]["decisions"]["domain"]["used"])
        self.assertEqual(outputs["target"]["decisions"]["target"]["selected"]["name"], "Office")
        for stage in expected:
            self.assertGreaterEqual(outputs[stage]["duration_ms"], 0)
            self.assertEqual(outputs[stage]["status"], "completed")
        self.assertEqual(outputs["service"]["usage"], {"input_tokens": 5})
        self.assertEqual(outputs["execution"]["arguments"]["entity_id"], ["light.desk", "light.ceiling"])
        self.assertEqual(outputs["answer"]["answer"], result["answer"])
        self.assertEqual(len(models.requests), 3)
        self.assertEqual(len(models.chat_requests), 1)
        execute.assert_awaited_once()

    async def test_trace_explains_confidence_rejection(self):
        events = []
        settings = normalize_settings({"decider": {"service_confidence": 0.95}})
        result, _, execute = await self.run_agent(
            [{"service": "light.turn_off"}], settings=settings, trace=lambda *event: events.append(event),
        )
        choice = next(output["decisions"]["service"] for stage, _, output in events
                      if stage == "service" and output is not None)
        self.assertEqual(choice["choice"], "light.turn_off")
        self.assertEqual(choice["effective_choice"], "unsure")
        self.assertEqual(choice["minimum_confidence"], 0.95)
        self.assertEqual(result["route"], "unsure")
        failure = events[-1][2]
        self.assertEqual(events[-1][0], "failure")
        self.assertEqual(failure, result["failures"][0])
        self.assertIn("90%", failure["reason"])
        self.assertIn("95%", failure["reason"])
        self.assertEqual(failure["stage"], "service")
        self.assertEqual(failure["device_command"], "not_sent")
        execute.assert_not_awaited()

    async def test_trace_retains_failed_answer_timing_and_cause(self):
        events = []
        result, _, execute = await self.run_agent([
            {"service": "query_state", "domain": "light"},
            {"area": "area:office"}, {"target": "area:office"},
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
        execute.assert_not_awaited()

    async def test_trace_retains_decider_failure_without_running_later_stages(self):
        events = []
        models = ScriptedModels([])
        models.decide = AsyncMock(side_effect=ModelError("Model connection failed."))
        execute = AsyncMock()
        agent = AgentEngine(normalize_settings({}), models, execute, trace=lambda *event: events.append(event))
        with self.assertRaisesRegex(ModelError, "connection failed"):
            await agent.process("test", "de", SNAPSHOT)
        self.assertEqual([stage for stage, _, _ in events], ["service", "service", "failure", "failure"])
        self.assertEqual(events[1][2]["status"], "error")
        self.assertIn("duration_ms", events[1][2])
        self.assertIn("could not be reached", events[-1][2]["reason"])
        execute.assert_not_awaited()

    async def test_trace_does_not_include_model_credentials_or_prompt_templates(self):
        events = []
        settings = normalize_settings({"decider": {"api_key": "secret-decider", "service_prompt": "private-template"},
                                       "answer": {"api_key": "secret-answer"}})
        await self.run_agent([
            {"service": "query_state", "domain": "light"},
            {"area": "area:office"}, {"target": "area:office"},
        ], settings=settings, trace=lambda *event: events.append(event))
        serialized = json.dumps(events)
        for secret in ("secret-decider", "secret-answer", "private-template"):
            self.assertNotIn(secret, serialized)

    async def test_direct_action_uses_its_domain_and_filtered_room_targets(self):
        result, models, execute = await self.run_agent([
            {"service": "light.turn_off", "domain": "unsure"},
            {"area": "area:office"}, {"target": "area:office"},
        ])
        execute.assert_awaited_once_with("light.turn_off", {"entity_id": ["light.desk", "light.ceiling"]})
        self.assertEqual(result["domain"], "light")
        self.assertEqual(result["answer"], "Erledigt.")
        self.assertEqual(models.requests[0][1], {"prompt": "test request", "language": "de"})
        self.assertEqual(models.requests[1][1], {"prompt": "test request", "language": "de"})
        criteria = models.requests[2][2]["target"]["criteria"]
        self.assertEqual(set(criteria), {"entity:light.desk", "entity:light.ceiling", "area:office", "unsure"})
        self.assertNotIn("attributes", json.loads(criteria["entity:light.desk"]))
        facts = json.loads(models.chat_requests[0][1][1]["content"])
        self.assertEqual(facts["execution"]["status"], "completed")
        self.assertNotIn("decisions", facts)

    async def test_query_uses_cached_state_and_does_not_execute(self):
        result, models, execute = await self.run_agent([
            {"service": "query_state", "domain": "sensor"},
            {"area": "area:office"}, {"target": "entity:sensor.office"},
        ], answers=["Es sind 22 Grad."])
        execute.assert_not_awaited()
        self.assertEqual(result["state_info"][0]["state"], "22")
        facts = json.loads(models.chat_requests[0][1][1]["content"])
        self.assertEqual(facts["state_info"][0]["attributes"]["unit_of_measurement"], "°C")
        self.assertEqual(result["answer"], "Es sind 22 Grad.")

    async def test_unavailable_query_omits_stale_attributes(self):
        snapshot = copy.deepcopy(SNAPSHOT)
        snapshot["entities"][-1]["state"] = "unavailable"
        result, _, _ = await self.run_agent([
            {"service": "query_state", "domain": "sensor"},
            {"area": "no_area"}, {"target": "entity:sensor.office"},
        ], snapshot=snapshot)
        self.assertEqual(result["state_info"][0]["attributes"], {})

    async def test_unsure_at_each_stage_bypasses_chat_and_execution(self):
        cases = [
            [{"service": "unsure"}],
            [{"service": "query_state", "domain": "unsure"}],
            [{"service": "light.turn_off"}, {"area": "unsure"}],
            [{"service": "light.turn_off"}, {"area": "no_area"}, {"target": "unsure"}],
        ]
        for choices, expected_stage in zip(cases, ("service", "domain", "area", "target")):
            with self.subTest(choices=choices):
                result, models, execute = await self.run_agent(choices)
                self.assertEqual(result["answer"], "ich bin mir nicht sicher")
                self.assertEqual(result["route"], "unsure")
                self.assertEqual(result["failures"][0]["stage"], expected_stage)
                self.assertIn("selected unsure", result["failures"][0]["reason"])
                self.assertFalse(models.chat_requests)
                execute.assert_not_awaited()

    async def test_disabled_general_returns_fixed_unsure(self):
        result, models, execute = await self.run_agent([{"service": "not-smarthome-related"}])
        self.assertEqual(result["answer"], "ich bin mir nicht sicher")
        self.assertIn("general model is disabled", result["failures"][0]["reason"])
        self.assertFalse(models.chat_requests)
        execute.assert_not_awaited()

    async def test_enabled_general_uses_its_own_model_and_original_request(self):
        settings = normalize_settings({"general": {"enable": True, "model": "smart-model"}})
        result, models, execute = await self.run_agent(
            [{"service": "not-smarthome-related"}], settings=settings, answers=["330 Meter."],
        )
        self.assertEqual(result["route"], "not-smarthome-related")
        self.assertEqual(len(models.requests), 1)
        self.assertEqual(models.chat_requests[0][0]["model"], "smart-model")
        self.assertEqual(models.chat_requests[0][1][1]["content"], "test request")
        self.assertIn("Assist pipeline language: de", models.chat_requests[0][1][0]["content"])
        self.assertEqual(result["answer"], "330 Meter.")
        execute.assert_not_awaited()

    async def test_confidence_threshold_stops_action(self):
        settings = normalize_settings({"decider": {"service_confidence": 0.95}})
        result, models, execute = await self.run_agent([{"service": "light.turn_off"}], settings=settings)
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(len(models.requests), 1)
        execute.assert_not_awaited()

    async def test_no_matching_domain_in_area_stops_selection(self):
        result, models, execute = await self.run_agent([
            {"service": "query_state", "domain": "sensor"}, {"area": "area:bedroom"},
        ])
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(len(models.requests), 2)
        self.assertEqual(result["failures"][0]["stage"], "target")
        self.assertIn("No exposed devices", result["failures"][0]["reason"])
        execute.assert_not_awaited()

    async def test_answer_failure_after_action_preserves_outcome_and_never_retries(self):
        result, models, execute = await self.run_agent([
            {"service": "light.turn_off"}, {"area": "no_area"}, {"target": "entity:light.desk"},
        ], answers=[ModelError("No answer")])
        execute.assert_awaited_once()
        self.assertEqual(result["execution"]["status"], "completed")
        self.assertEqual(result["answer"], DEFAULTS["answer_error_reply"])
        self.assertEqual(result["answer_error"], "No answer")
        self.assertIn("answer", result["timings_ms"])
        self.assertEqual(len(models.chat_requests), 1)
        self.assertEqual(result["failures"][0]["device_command"], "completed")

    async def test_query_answer_failure_preserves_state_and_error_details(self):
        result, models, execute = await self.run_agent([
            {"service": "query_state", "domain": "light"},
            {"area": "area:office"}, {"target": "entity:light.desk"},
        ], answers=[ModelError("Model reply reached its output token limit.")])
        self.assertEqual(result["answer"], DEFAULTS["error_reply"])
        self.assertEqual(result["answer_error"], "Model reply reached its output token limit.")
        self.assertEqual(result["state_info"][0]["state"], "on")
        self.assertIn("answer", result["timings_ms"])
        self.assertEqual(len(models.chat_requests), 1)
        execute.assert_not_awaited()

    async def test_action_failure_is_reported_as_unconfirmed(self):
        execute = AsyncMock(side_effect=ActionError("partial failure"))
        result, models, _ = await self.run_agent([
            {"service": "light.turn_off"}, {"area": "no_area"}, {"target": "entity:light.desk"},
        ], executor=execute)
        execute.assert_awaited_once()
        self.assertEqual(result["execution"]["status"], "error")
        facts = json.loads(models.chat_requests[0][1][1]["content"])
        self.assertIn("unconfirmed", facts["execution"]["error"])
        self.assertEqual(result["failures"][0]["stage"], "execution")
        self.assertEqual(result["failures"][0]["device_command"], "unconfirmed")
        self.assertNotIn("failures", facts)

    async def run_numeric(self, currents=(21,), score=4.02, mode="relative", parameter="temperature", settings=None, trace=None, history=None):
        climate = parameter == "temperature"
        name = "climate.set_temperature" if climate else "light.turn_on"
        field = "temperature" if climate else "brightness_pct"
        snapshot = {
            "entities": [entity(f"{'climate' if climate else 'light'}.device_{index}", f"Device {index}",
                                temperature=current, brightness=current, min_temp=5, max_temp=45, target_temp_step=1)
                         for index, current in enumerate(currents)],
            "services": {name: {"target": {}, "fields": {field: {"required": climate}}}},
        }
        return await self.run_agent([
            {"service": f"{name}:{field}"}, {"area": "area:office"}, {"target": "area:office"},
            {"numeric_mode": mode, "numeric_score": score},
        ], snapshot=snapshot, settings=settings, trace=trace, history=history)

    async def test_numeric_history_does_not_replace_the_fresh_current_setting(self):
        history = [{"role": "user", "content": "Make the office warmer"}, {"role": "assistant", "content": "Target is 25 degrees"}]
        settings = normalize_settings({"history_enabled": True})
        _, models, execute = await self.run_numeric(currents=(21,), settings=settings, history=history)
        numeric_state = models.requests[3][1]
        self.assertEqual(numeric_state["history"], history)
        self.assertEqual(numeric_state["language"], "de")
        self.assertEqual(numeric_state["current"], 21)
        self.assertEqual(execute.await_args.args[1]["temperature"], 22)

    async def test_numeric_trace_shows_score_confidence_and_actual_applied_value(self):
        events = []
        await self.run_numeric(trace=lambda *event: events.append(event))
        numeric_input = next(arguments for stage, arguments, output in events if stage == "numeric" and output is None)
        outputs = {stage: output for stage, _, output in events if output is not None}
        self.assertEqual(numeric_input["current"], 21)
        self.assertEqual(numeric_input["candidate_counts"]["numeric_score"], 9)
        self.assertEqual(outputs["numeric"]["decisions"]["numeric_score"]["score"], 4.02)
        self.assertEqual(outputs["numeric"]["decisions"]["numeric_score"]["confidence"], 0.9)
        self.assertEqual(outputs["numeric"]["decisions"]["numeric_mode"]["choice"], "relative")
        self.assertEqual(outputs["execution"]["arguments"]["temperature"], 22)

    async def test_temperature_is_quantized_and_relative_direction_preserved(self):
        result, models, execute = await self.run_numeric()
        execute.assert_awaited_once_with("climate.set_temperature", {"entity_id": ["climate.device_0"], "temperature": 22})
        self.assertEqual(result["route"], "smarthome")
        state = models.requests[3][1]
        self.assertEqual((state["minimum"], state["maximum"], state["current"], state["baseline"]), (16, 26, 21, "target"))

    async def test_fractional_baseline_does_not_reverse_relative_direction(self):
        _, _, execute = await self.run_numeric(currents=(21.01,), score=4.02)
        self.assertEqual(execute.await_args.args[1]["temperature"], 22)

    async def test_relative_area_with_different_settings_is_unsure(self):
        result, models, execute = await self.run_numeric(currents=(20, 22))
        self.assertEqual(result["route"], "unsure")
        self.assertFalse(models.chat_requests)
        self.assertIn("differ between", result["failures"][0]["reason"])
        self.assertEqual(result["failures"][0]["current_values"], [20, 22])
        execute.assert_not_awaited()

    async def test_missing_current_value_explains_why_relative_change_stopped(self):
        result, models, execute = await self.run_numeric(currents=(None,))
        self.assertEqual(result["failures"][0]["stage"], "numeric")
        self.assertIn("unavailable", result["failures"][0]["reason"])
        self.assertIn("absolute value", result["failures"][0]["suggestion"])
        self.assertFalse(models.chat_requests)
        execute.assert_not_awaited()

    async def test_invalid_decision_after_response_still_emits_a_failure_summary(self):
        events = []
        models = ScriptedModels([])
        models.decide = AsyncMock(return_value={"answers": {}})
        execute = AsyncMock()
        agent = AgentEngine(normalize_settings({}), models, execute, trace=lambda *event: events.append(event))
        with self.assertRaises(ModelError):
            await agent.process("test", "de", SNAPSHOT)
        self.assertEqual(events[-1][0], "failure")
        self.assertEqual(events[-1][2]["stage"], "service")
        self.assertIn("invalid decision", events[-1][2]["reason"])
        execute.assert_not_awaited()

    async def test_absolute_area_can_set_a_common_temperature(self):
        _, _, execute = await self.run_numeric(currents=(20, 22), score=4.8, mode="absolute")
        execute.assert_awaited_once_with("climate.set_temperature", {
            "entity_id": ["climate.device_0", "climate.device_1"], "temperature": 22,
        })

    async def test_brightness_uses_percent_and_configured_default_change(self):
        _, models, execute = await self.run_numeric(currents=(127.5,), score=3, parameter="brightness")
        self.assertEqual(models.requests[3][1]["current"], 50)
        self.assertEqual(models.requests[3][1]["default_step"], 10)
        execute.assert_awaited_once_with("light.turn_on", {"entity_id": ["light.device_0"], "brightness_pct": 60})

    async def test_numeric_unsure_skips_action_and_answer_model(self):
        result, models, execute = await self.run_numeric(mode="unsure")
        self.assertEqual(result["answer"], "ich bin mir nicht sicher")
        self.assertFalse(models.chat_requests)
        execute.assert_not_awaited()

    async def run_text(self, service="todo.add_item", settings=None, trace=None, history=None, target_area=False, answers=None, executor=None):
        domain = service.split(".")[0]
        parameter = TEXT_ACTIONS[service][0]
        prompt = "Schreibe Energy Energy Drinks auf die Einkaufsliste"
        choices = [
            {"service": service}, {"area": "no_area"},
            {"target": "area:office" if target_area else f"entity:{domain}.destination"},
            {f"word_{index}": "keep" if index in (1, 2, 3) else "drop" for index in range(len(prompt.split()))},
        ]
        members = [entity(f"{domain}.destination", "Einkaufsliste" if domain == "todo" else "Destination")]
        if target_area:
            members.append(entity(f"{domain}.second", "Second destination"))
        if domain == "todo":
            members[0].update(area_id=None, area_name=None)
        snapshot = {
            "entities": [*members, entity("light.unrelated", "Unrelated lamp")],
            "services": {service: {"target": {}, "fields": {parameter: {"required": domain != "assist_satellite"}}}},
        }
        return await self.run_agent(choices, snapshot=snapshot, settings=settings, trace=trace, history=history,
                                    prompt=prompt, answers=answers, executor=executor)

    async def test_three_allowed_text_actions_copy_payload_into_the_correct_field(self):
        for service, (parameter, _) in TEXT_ACTIONS.items():
            with self.subTest(service=service):
                result, models, execute = await self.run_text(service)
                domain = service.split(".")[0]
                execute.assert_awaited_once_with(service, {"entity_id": [f"{domain}.destination"], parameter: "Energy Energy Drinks"})
                self.assertEqual(result["execution"]["status"], "completed")
                self.assertEqual(len(models.requests), 4)
                state = models.requests[-1][1]
                prompt_key = "text_prompt" if service == "todo.add_item" else "message_prompt"
                self.assertEqual(state["task"], DEFAULTS["decider"][prompt_key])
                self.assertEqual(state["language"], "de")
                self.assertEqual(state["words"], result["request"].split())
                facts = json.loads(models.chat_requests[0][1][1]["content"])
                self.assertEqual(facts["execution"]["arguments"][parameter], "Energy Energy Drinks")

    async def test_announcement_area_target_keeps_the_existing_entity_scope(self):
        _, _, execute = await self.run_text("assist_satellite.announce", target_area=True)
        execute.assert_awaited_once_with("assist_satellite.announce", {
            "entity_id": ["assist_satellite.destination", "assist_satellite.second"], "message": "Energy Energy Drinks",
        })

    async def test_text_threshold_equality_drops_words_and_empty_payload_never_executes(self):
        settings = normalize_settings({"decider": {"text_confidence": 0.9}})
        result, models, execute = await self.run_text(settings=settings)
        self.assertEqual(result["route"], "unsure")
        self.assertEqual(result["failures"][0]["stage"], "text")
        self.assertIn("No payload words", result["failures"][0]["reason"])
        self.assertFalse(models.chat_requests)
        execute.assert_not_awaited()

    async def test_text_trace_shows_confidence_threshold_and_kept_words_without_prompt_template(self):
        events = []
        settings = normalize_settings({"history_enabled": True, "decider": {
            "text_prompt": "private-text-template", "text_confidence": 0.7,
        }})
        history = [{"role": "assistant", "content": "Previous item"}]
        _, models, _ = await self.run_text(settings=settings, trace=lambda *event: events.append(event), history=history)
        self.assertEqual(models.requests[-1][1]["history"], history)
        self.assertEqual(models.requests[-1][1]["task"], "private-text-template")
        self.assertTrue(all("Apply state.task" in question["instructions"] for question in models.requests[-1][2].values()))
        self.assertNotIn("private-text-template", json.dumps(events))
        output = next(output for stage, _, output in events if stage == "text" and output is not None)
        self.assertGreaterEqual(output["duration_ms"], 0)
        self.assertEqual(output["decisions"]["word_1"]["confidence"], 0.9)
        self.assertEqual(output["decisions"]["word_1"]["minimum_confidence"], 0.7)
        self.assertTrue(output["decisions"]["word_1"]["kept"])
        self.assertEqual(output["decisions"]["word_0"]["effective_choice"], "drop")

    async def test_answer_failure_after_text_action_does_not_repeat_the_action(self):
        result, _, execute = await self.run_text(answers=[ModelError("Model reply reached its output token limit.")])
        execute.assert_awaited_once()
        self.assertEqual(result["execution"]["status"], "completed")
        self.assertEqual(result["failures"][0]["device_command"], "completed")

    async def test_shopping_extraction_preserves_probe_questions_with_pipeline_language_and_history(self):
        from text_probe import INSTRUCTIONS, word_request

        history = [{"role": "user", "content": "Use the shopping list"}]
        settings = normalize_settings({"history_enabled": True})
        result, models, _ = await self.run_text(settings=settings, history=history)
        _, probe = word_request(result["request"], INSTRUCTIONS)
        _, state, questions = models.requests[-1]
        self.assertEqual(questions, probe["questions"])
        self.assertEqual(state, {**probe["state"], "language": "de", "history": history})

    async def test_custom_message_prompt_is_used_for_both_message_actions(self):
        settings = normalize_settings({"decider": {"text_prompt": "Item rules", "message_prompt": "Message rules"}})
        for service in ("notify.send_message", "assist_satellite.announce"):
            with self.subTest(service=service):
                _, models, _ = await self.run_text(service=service, settings=settings)
                self.assertEqual(models.requests[-1][1]["task"], "Message rules")

    async def test_incomplete_text_response_never_executes_and_has_a_debug_failure(self):
        original = ScriptedModels.decide

        async def incomplete(model, config, state, questions):
            response = await original(model, config, state, questions)
            if "word_0" in questions:
                response["answers"].pop("word_0")
            return response

        events = []
        execute = AsyncMock()
        with patch.object(ScriptedModels, "decide", new=incomplete), self.assertRaises(ModelError):
            await self.run_text(executor=execute, trace=lambda *event: events.append(event))
        execute.assert_not_awaited()
        failure = events[-1][2]
        self.assertEqual(failure["stage"], "text")
        self.assertIn("incomplete word selections", failure["reason"])


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
        self.assertEqual(saved["decider"]["text_confidence"], 0.5)
        custom = normalize_settings({"decider": {"service_prompt": "Custom action rules", "text_prompt": "Custom extraction rules"}})
        self.assertEqual(custom["decider"]["service_prompt"], "Custom action rules")
        self.assertEqual(custom["decider"]["text_prompt"], "Custom extraction rules")

    def test_text_prompt_upgrade_restores_probe_wording_and_preserves_custom_shared_prompts(self):
        from custom_components.glow_assistant.text_values import (
            INSTRUCTIONS,
            LEGACY_INSTRUCTIONS,
            MESSAGE_INSTRUCTIONS,
        )

        updated = normalize_settings({"decider": {"text_prompt": LEGACY_INSTRUCTIONS}})["decider"]
        self.assertEqual(updated["text_prompt"], INSTRUCTIONS)
        self.assertEqual(updated["message_prompt"], MESSAGE_INSTRUCTIONS)
        custom = normalize_settings({"decider": {"text_prompt": "Shared custom rules"}})["decider"]
        self.assertEqual(custom["text_prompt"], "Shared custom rules")
        self.assertEqual(custom["message_prompt"], "Shared custom rules")

    def test_ui_numeric_selectors_produce_integer_token_limits(self):
        settings = self.local_settings()
        settings["answer"]["max_tokens"] = 512.0
        settings["decider"]["max_input_chars"] = 100000.0
        saved = validate_settings(settings)
        self.assertIsInstance(saved["answer"]["max_tokens"], int)
        self.assertIsInstance(saved["decider"]["max_input_chars"], int)

    def test_prompt_reset_and_customization_survive_normalization(self):
        settings = normalize_settings({"decider": {"target_prompt": "My rules", "area_prompt": " "}})
        self.assertEqual(settings["decider"]["target_prompt"], "My rules")
        self.assertEqual(settings["decider"]["area_prompt"], DEFAULTS["decider"]["area_prompt"])
        self.assertNotEqual(id(settings["decider"]), id(DEFAULTS["decider"]))

    def test_invalid_ranges_and_models_are_rejected(self):
        for group, field, value in (
            ("decider", "temperature_min", 27), ("decider", "temperature_levels", 11),
            ("decider", "temperature_levels", 2.5), ("decider", "temperature_step", 0),
            ("decider", "service_confidence", 1.1), ("decider", "url", "localhost:8321"),
            ("decider", "text_confidence", -0.1), ("decider", "text_confidence", 1.1),
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
        })
        self.assertIn("light.turn_on", options)
        self.assertIn("light.turn_on:brightness_pct", options)
        self.assertIn("climate.set_temperature:temperature", options)
        self.assertNotIn("climate.set_temperature", options)
        self.assertNotIn("climate.set_hvac_mode", options)
        self.assertNotIn("todo.get_items", options)

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
