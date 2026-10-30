import json
import math
from time import perf_counter

from .client import ModelError
from .const import GENERAL, HISTORY_INSTRUCTIONS, QUERY, UNSURE
from .debug import STAGE_LABELS, decision_details, failure_details, model_failure
from .domain_filter import decode_domains, domain_request
from .general import general_answer
from .history import with_history
from .numeric_values import decode_numeric, finite_number, numeric_request
from .target_selection import decode_targets, target_request
from .text_values import TEXT_ACTIONS, decode_text, text_request


class Uncertain(Exception):
    def __init__(self, stage, reason, suggestion, **details):
        super().__init__(reason)
        self.details = failure_details(stage, reason, suggestion, status="stopped", **details)


class ActionError(Exception):
    pass


def choice_question(options, instructions):
    return {
        "type": "choice", "instructions": instructions,
        "criteria": {
            **{key: json.dumps(value, ensure_ascii=False) for key, value in options.items()},
            UNSURE: "Unclear, unsupported, or no single matching choice.",
        },
    }


def read_choice(response, question, options, threshold):
    try:
        answer = response["answers"][question]
        labels = set(options) | {UNSURE}
        probabilities = answer["probabilities"]
        if (
            answer["type"] != "choice" or answer["choice"] not in labels
            or set(probabilities) != labels
            or not all(finite_number(value) and 0 <= value <= 1
                       for value in [answer["confidence"], *probabilities.values()])
        ):
            raise ModelError("Invalid classifier choice or probabilities.")
        return answer["choice"] if answer["confidence"] >= threshold else UNSURE
    except (KeyError, TypeError, AttributeError) as error:
        raise ModelError("Malformed classifier response.") from error


def action_options(services):
    options = {}
    for name, description in services.items():
        if description.get("response", {}).get("optional") is False:
            continue
        fields = description.get("fields") or {}
        required = {key for key, field in fields.items() if field.get("required")}
        base = {"service": name, "description": description.get("description", name)}
        if name in TEXT_ACTIONS:
            parameter, explanation = TEXT_ACTIONS[name]
            if parameter in fields and required <= {parameter}:
                options[name] = {**base, "parameter": parameter, "parameter_type": "text",
                                 "parameter_description": explanation, "mode": "Only the named free-text parameter; no extra fields."}
            continue
        if not required and name != "climate.set_temperature":
            options[name] = {**base, "parameter": None, "mode": "target-only; no extra values"}
        parameter = {"climate.set_temperature": "temperature", "light.turn_on": "brightness_pct"}.get(name)
        allowed = {"temperature"} if parameter == "temperature" else {"brightness", "brightness_pct"}
        if parameter and required <= allowed:
            options[f"{name}:{parameter}"] = {**base, "parameter": parameter}
    return options


def areas_from(entities):
    return {
        entity["area_id"]: {
            "name": entity["area_name"], "aliases": entity.get("area_aliases", []),
        }
        for entity in entities if entity.get("area_id")
    }


def target_options(entities):
    options = {}
    for entity in entities:
        attributes = entity.get("attributes", {})
        options[f"entity:{entity['entity_id']}"] = {
            "type": "entity", "entity_id": entity["entity_id"], "name": entity["name"],
            "domain": entity["domain"],
            "aliases": entity.get("aliases", []), "area": entity.get("area_name"),
            "device_class": attributes.get("device_class"),
            "unit": attributes.get("unit_of_measurement"),
        }
    return options


def current_setting(entity, parameter, baseline):
    if entity["state"] in ("unknown", "unavailable"):
        return None
    attributes = entity.get("attributes", {})
    if parameter == "temperature":
        value = attributes.get("temperature" if baseline == "target" else "current_temperature")
    elif entity["state"] == "off":
        value = 0
    else:
        value = attributes.get("brightness")
        value = value * 100 / 255 if finite_number(value) else None
    return value if finite_number(value) else None


class AgentEngine:
    def __init__(self, settings, client, execute, trace=None, history=None):
        self.settings = settings
        self.client = client
        self.execute = execute
        self.trace = trace
        self.history = list(history or []) if settings["history_enabled"] else []
        self.stage = "service"

    def _trace(self, stage, arguments, result=None):
        if result is None and stage != "failure":
            self.stage = stage
        if self.trace:
            self.trace(stage, arguments, result)

    def _failure(self, result, details):
        execution = result.get("execution", {}).get("status")
        details = {**details, "device_command":
                   "completed" if execution == "completed" else "unconfirmed" if execution == "error" else "not_sent"}
        result.setdefault("failures", []).append(details)
        self._trace("failure", details)
        self._trace("failure", {}, details)

    async def _decide(self, result, stage, state, questions, summarize=None):
        config = self.settings["decider"]
        state = {**state, "language": result["language"]}
        if self.history:
            state = {**state, "history": self.history}
            if "task" not in state:
                questions = {name: {**question, "instructions": f"{question['instructions']} {HISTORY_INSTRUCTIONS}"}
                             for name, question in questions.items()}
        self._trace(stage, {
            "model": config["model"], **{key: value for key, value in state.items() if key != "task"},
            "candidate_counts": {name: len(question["criteria"]) for name, question in questions.items()},
        })
        started = perf_counter()
        details = {"status": "incomplete"}
        try:
            response = await self.client.decide(config, state, questions)
            result["usage"][stage] = response.get("usage", {})
            result["decisions"][stage] = response.get("answers", {})
            details.update(
                status="completed",
                decisions=decision_details(questions, result["decisions"][stage], config,
                                           query=result["service"] == QUERY, targets=state.get("targets")),
                usage=result["usage"][stage],
            )
            if summarize:
                details.update(summarize(response))
            return response
        except ModelError as error:
            details.update(status="error", error=str(error), **error.details)
            raise
        finally:
            result["timings_ms"][stage] = round((perf_counter() - started) * 1000)
            details["duration_ms"] = result["timings_ms"][stage]
            self._trace(stage, {}, details)

    def _choice(self, response, question, options):
        self.stage = "numeric" if question == "numeric_mode" else question
        threshold = self.settings["decider"][f"{question if question != 'numeric_mode' else 'numeric'}_confidence"]
        choice = read_choice(response, question, options, threshold)
        if choice == UNSURE:
            confidence = response["answers"][question]["confidence"]
            reason = (f"The model's confidence ({confidence:.0%}) is below the configured minimum ({threshold:.0%})."
                      if confidence < threshold else "The model selected unsure; it could not make one clear selection.")
            raise Uncertain(question, reason,
                            f"Rephrase the request with a specific action, room, or device. Review {STAGE_LABELS[question]} above.",
                            confidence=confidence, minimum_confidence=threshold)
        return choice

    async def process(self, prompt, language, snapshot):
        result = {
            "request": prompt, "language": language, "route": "smarthome",
            "service": None, "domain": None, "target": None,
            "decisions": {}, "usage": {}, "timings_ms": {},
        }
        try:
            await self._process(result, snapshot)
        except Uncertain as error:
            result.update(route=UNSURE, answer=self.settings["unsure_reply"])
            self._failure(result, error.details)
        except ModelError as error:
            self._failure(result, model_failure(self.stage, error))
            raise
        return result

    async def _process(self, result, snapshot):
        prompt = result["request"]
        language_instruction = (
            f"Assist pipeline language: {result['language']}. Write the spoken reply in this language."
        )
        config = self.settings["decider"]
        entities = snapshot["entities"]
        actions = action_options(snapshot["services"])
        services = {
            **actions, QUERY: "Query current home states or measurements, no action.",
            GENERAL: "General knowledge or conversation unrelated to the home.",
        }
        response = await self._decide(result, "service", {"prompt": prompt}, {
            "service": choice_question(services, config["service_prompt"]),
        })
        service_choice = self._choice(response, "service", services)
        if service_choice == GENERAL:
            if not self.settings["general"]["enable"]:
                raise Uncertain("general", "The request is not about the smart home, but the general model is disabled.",
                                "Enable General model to answer general questions.")
            result["route"] = GENERAL
            messages = [
                {"role": "system", "content": f"{self.settings['general']['prompt']}\n{language_instruction}"},
                {"role": "user", "content": prompt},
            ]
            if any(server["enable"] for server in self.settings["general"]["mcp_servers"]):
                started = perf_counter()
                try:
                    result["answer"], rounds = await general_answer(
                        self.client, self.settings["general"], with_history(messages, self.history), trace=self._trace,
                    )
                    result["usage"]["general"] = {"rounds": rounds}
                finally:
                    result["timings_ms"]["general"] = round((perf_counter() - started) * 1000)
            else:
                await self._chat(result, "general", messages)
            return
        query = service_choice == QUERY
        action = actions.get(service_choice)
        domain = None if query else action["service"].split(".", 1)[0]
        result["service"] = QUERY if query else action["service"]
        result["domain"] = domain
        areas = {**{f"area:{key}": value for key, value in areas_from(entities).items()},
                 "no_area": "No specific area is identified."}
        area_prompt = config["area_prompt"]
        if query:
            areas["all_areas"] = "All areas: the whole home, including exposed entities without a room assignment."
            area_prompt += (
                " For state queries about the whole home, choose all_areas instead of a single room or no_area. "
                "Examples: 'are there any windows open in the house?' and 'sind irgendwo im Haus Fenster offen?'. "
                "This is one supported scope, not an ambiguous request for multiple areas."
            )
        response = await self._decide(result, "area", {"prompt": prompt}, {
            "area": choice_question(areas, area_prompt),
        })
        area = self._choice(response, "area", areas)
        candidates = [entity for entity in entities if (query or entity["domain"] == domain)
                      and (area in ("no_area", "all_areas") or entity.get("area_id") == area.removeprefix("area:"))]
        if not candidates:
            raise Uncertain("target", "No exposed devices match the selected action and room filter.",
                            "Check the device's room, Assist exposure settings, and GlowAssistant exclusions.",
                            domain=domain, area=areas[area])
        if query and area == "all_areas":
            candidates = await self._filter_domains(result, candidates)
        target_state = {
            "prompt": prompt, "service": result["service"],
            **({"domain": domain} if not query else {}),
            "area": areas[area] if area != "no_area" else None,
        }
        # Native HA target selectors accept entity lists
        multiple_targets = query or snapshot["services"][action["service"]].get("target") is not None
        if multiple_targets:
            members = await self._select_targets(result, candidates, target_state, query)
        else:
            options = target_options(candidates)
            response = await self._decide(result, "target", target_state, {
                "target": choice_question(options, config["target_prompt"]),
            })
            target = self._choice(response, "target", options)
            result["target"] = options[target]
            members = [entity for entity in candidates if entity["entity_id"] == result["target"]["entity_id"]]
        if not members:
            raise Uncertain("target", "The selected target contains no matching exposed devices.",
                            "Check Assist exposure settings and the device's room assignment.")
        if query:
            member_domains = {entity["domain"] for entity in members}
            result["domain"] = next(iter(member_domains)) if len(member_domains) == 1 else None
            result["state_info"] = [
                {"entity_id": entity["entity_id"], "domain": entity["domain"], "area": entity.get("area_name"),
                 "name": entity["name"], "state": entity["state"], "attributes":
                 entity.get("attributes", {}) if entity["state"] not in ("unknown", "unavailable") else {}}
                for entity in members
            ]
        else:
            arguments = {"entity_id": [entity["entity_id"] for entity in members]}
            if action["parameter"]:
                if action.get("parameter_type") == "text":
                    arguments[action["parameter"]] = await self._text(result)
                else:
                    arguments[action["parameter"]] = await self._numeric(result, members, action["parameter"])
            self._trace("execution", {"service": action["service"], "arguments": arguments})
            execution_started = perf_counter()
            try:
                await self.execute(action["service"], arguments)
            except ActionError as error:
                result["execution"] = {"status": "error", "error": f"Action failed; device outcome unconfirmed: {error}"}
            else:
                result["execution"] = {"status": "completed", "arguments": arguments}
            finally:
                result["timings_ms"]["execution"] = round((perf_counter() - execution_started) * 1000)
                self._trace("execution", {}, {
                    **result.get("execution", {"status": "incomplete"}),
                    "duration_ms": result["timings_ms"]["execution"],
                })
            if result["execution"]["status"] == "error":
                self._failure(result, failure_details("execution", "Home Assistant could not confirm the device command.",
                                                     "Check the device's availability, Assist exposure, and Home Assistant logs."))
        facts = {key: value for key, value in result.items()
                 if key not in ("decisions", "usage", "timings_ms", "failures", "domain_filter", "target_selection")}
        try:
            await self._chat(result, "answer", [
                {"role": "system", "content": f"{self.settings['answer']['prompt']}\n{language_instruction}"},
                {"role": "user", "content": json.dumps(facts, ensure_ascii=False, default=str)},
            ])
        except ModelError as error:
            completed = result.get("execution", {}).get("status") == "completed"
            result["answer"] = self.settings["answer_error_reply" if completed else "error_reply"]
            result["answer_error"] = str(error)
            self._failure(result, model_failure("answer", error))

    async def _chat(self, result, stage, messages):
        config = self.settings[stage]
        messages = with_history(messages, self.history)
        arguments = {"model": config["model"], "max_tokens": config["max_tokens"]}
        if stage == "answer":
            arguments.update(messages=messages, temperature=config["temperature"], max_input_chars=config["max_input_chars"])
        self._trace(stage, arguments)
        started = perf_counter()
        details = {"status": "incomplete"}
        try:
            result["answer"], result["usage"][stage] = await self.client.chat(config, messages)
            details.update(status="completed", answer=result["answer"], usage=result["usage"][stage])
        except ModelError as error:
            details.update(status="error", error=str(error), **error.details)
            raise
        finally:
            result["timings_ms"][stage] = round((perf_counter() - started) * 1000)
            details["duration_ms"] = result["timings_ms"][stage]
            self._trace(stage, {}, details)

    async def _select_targets(self, result, entities, state, query):
        config = self.settings["decider"]
        targets = list(target_options(entities).values())
        payload = target_request(state, targets, config["target_filter_prompt"])

        def summarize(response):
            policy = "query_target_keep_uncertain" if query else "action_target_keep_uncertain"
            selected = decode_targets(response, targets, config["target_filter_confidence"], config[policy])
            result["target_selection"] = selected
            return selected

        await self._decide(result, "target", payload["state"], payload["questions"], summarize=summarize)
        selected = result["target_selection"]["selected_entity_ids"]
        if not selected:
            raise Uncertain("target", "No matching entities were retained by target selection.",
                            "Name the requested devices explicitly. Review the target keep/drop decisions and confidence threshold.")
        members = [entity for entity in entities if entity["entity_id"] in selected]
        if len(members) == 1:
            result["target"] = next(target for target in targets if target["entity_id"] == selected[0])
        else:
            result["target"] = {"type": "entities", "entity_ids": selected, "names": [entity["name"] for entity in members]}
        return members

    async def _filter_domains(self, result, entities):
        config = self.settings["decider"]
        domains = sorted({entity["domain"] for entity in entities})
        payload = domain_request(result["request"], domains, config["domain_filter_prompt"])

        def summarize(response):
            selected = decode_domains(response, domains, config["domain_filter_confidence"], config["domain_filter_keep_uncertain"])
            selected.update(entities_before=len(entities),
                            entities_after=sum(entity["domain"] in selected["kept_domains"] for entity in entities))
            result["domain_filter"] = selected
            return selected

        await self._decide(result, "domain_filter", payload["state"], payload["questions"], summarize=summarize)
        if not result["domain_filter"]["kept_domains"]:
            raise Uncertain("domain_filter", "No domains were retained by the whole-home domain filter.",
                            "Review the domain decisions, confidence threshold, and Keep uncertain domains setting.")
        return [entity for entity in entities if entity["domain"] in result["domain_filter"]["kept_domains"]]

    async def _text(self, result):
        config = self.settings["decider"]
        prompt_key = "text_prompt" if result["service"] == "todo.add_item" else "message_prompt"
        words, payload = text_request(result["request"], result["service"], config[prompt_key])
        response = await self._decide(result, "text", payload["state"], payload["questions"])
        text = decode_text(response, words, config["text_confidence"], config["text_keep_uncertain"])
        if not text:
            raise Uncertain("text", "No payload words were selected above the text confidence threshold.",
                            "State the item or message explicitly. Review Text extraction confidence under Decider.",
                            confidence_threshold=config["text_confidence"])
        return text

    async def _numeric(self, result, members, service_parameter):
        self.stage = "numeric"
        parameter = "temperature" if service_parameter == "temperature" else "brightness"
        config = self.settings["decider"]
        values = [current_setting(entity, parameter, config["temperature_baseline"]) for entity in members]
        current = values[0] if None not in values and len(set(values)) == 1 else None
        minimum, maximum = config[f"{parameter}_min"], config[f"{parameter}_max"]
        step = config[f"{parameter}_step"]
        unit = "%"
        if parameter == "temperature":
            units = {entity.get("attributes", {}).get("temperature_unit", "°C") for entity in members}
            if len(units) != 1:
                raise Uncertain("numeric", "The selected thermostats use different temperature units.",
                                "Select one thermostat, or set the devices to the same temperature unit.")
            unit = units.pop()
            for entity in members:
                attributes = entity.get("attributes", {})
                minimum = max(minimum, attributes.get("min_temp", minimum))
                maximum = min(maximum, attributes.get("max_temp", maximum))
                step = max(step, attributes.get("target_temp_step", step))
        elif any(set(entity.get("attributes", {}).get("supported_color_modes", [])) == {"onoff"} for entity in members):
            raise Uncertain("numeric", "A selected light supports only on/off, not brightness changes.",
                            "Select a dimmable light, or ask to turn the light on or off.")
        if minimum >= maximum:
            raise Uncertain("numeric", "The configured limits and device limits have no usable range in common.",
                            "Check the numeric limits under Decider and the device's supported range.")
        anchors, payload = numeric_request(
            result["request"], parameter, current, minimum, maximum,
            int(config[f"{parameter}_levels"]), unit, method="score",
            default_step=config[f"{parameter}_default_change"], instructions=config["numeric_prompt"],
        )
        modes = {"absolute": "An explicit new final setting.", "relative": "A change from the current setting."}
        payload["state"].update(current_values=values, baseline=config["temperature_baseline"], target=result["target"])
        payload["questions"]["numeric_mode"] = choice_question(modes, config["numeric_mode_prompt"])
        response = await self._decide(result, "numeric", payload["state"], payload["questions"])
        mode = self._choice(response, "numeric_mode", modes)
        if mode == "relative" and current is None:
            raise Uncertain("numeric", "The current settings are unavailable or differ between the selected devices.",
                            "Specify an absolute value, or select one device with a known current setting.",
                            current_values=values)
        try:
            estimate = decode_numeric(response, anchors, method="score")["score"]
        except ValueError as error:
            raise ModelError("Malformed numeric response.") from error
        if estimate["confidence"] < config["numeric_confidence"]:
            raise Uncertain("numeric", "The model's confidence in the new value is below the configured minimum.",
                            "Specify an explicit new value. Review the numeric confidence threshold under Decider.",
                            confidence=estimate["confidence"], minimum_confidence=config["numeric_confidence"])
        raw = estimate["value"]
        value = math.floor(raw / step + 0.5) * step
        if mode == "relative":
            if math.isclose(raw, current):
                raise Uncertain("numeric", "The model's new value would not change the current setting.",
                                "Specify the desired value or the size of the change.")
            if raw > current and value <= current:
                value = (math.floor(current / step) + 1) * step
            elif raw < current and value >= current:
                value = (math.ceil(current / step) - 1) * step
        if not minimum <= value <= maximum:
            raise Uncertain("numeric", "The new value is outside the allowed range after rounding to the device's step.",
                            "Choose a value inside the supported range, or review the numeric limits under Decider.",
                            value=value, minimum=minimum, maximum=maximum)
        for entity in members:
            if parameter == "temperature":
                attributes = entity.get("attributes", {})
                device_step = attributes.get("target_temp_step", step)
                if not math.isclose(value / device_step, round(value / device_step), abs_tol=1e-6):
                    raise Uncertain("numeric", "The new value does not match a selected thermostat's temperature step.",
                                    "Choose a supported whole step, or select one thermostat.", value=value, device_step=device_step)
        return int(value) if float(value).is_integer() else round(value, 6)
