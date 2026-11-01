import json
from time import perf_counter

from .action_targets import decode_selection, selection_question
from .client import ModelError
from .const import (
    GENERAL,
    HISTORY_INSTRUCTIONS,
    QUERY,
    TARGET_SCOPES,
    UNSURE,
)
from .debug import STAGE_LABELS, decision_details, failure_details, model_failure
from .general import general_answer
from .history import with_history
from .numeric_values import finite_number
from .optional_parameters import (
    action_fields,
    decode_parameters,
    filter_fields,
    parameter_request,
)
from .target_selection import decode_targets, target_request
from .text_values import TEXT_ACTIONS


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
        if not required and name not in ("climate.set_temperature", "cover.set_cover_position"):
            options[name] = {**base, "parameter": None, "mode": "target-only; no extra values"}
        parameter = {"climate.set_temperature": "temperature", "light.turn_on": "brightness_pct",
                     "cover.set_cover_position": "position"}.get(name)
        allowed = {"brightness", "brightness_pct"} if parameter == "brightness_pct" else {parameter}
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


class AgentEngine:
    def __init__(self, settings, client, execute, trace=None, history=None, dry_run=False):
        self.settings = settings
        self.client = client
        self.execute = execute
        self.trace = trace
        self.history = list(history or []) if settings["history_enabled"] else []
        self.stage = "service"
        self.dry_run = dry_run

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
        self.stage = question
        threshold = self.settings["decider"][f"{question}_confidence"]
        choice = read_choice(response, question, options, threshold)
        if choice == UNSURE:
            confidence = response["answers"][question]["confidence"]
            reason = (f"The model's confidence ({confidence:.0%}) is below the configured minimum ({threshold:.0%})."
                      if confidence < threshold else "The model selected unsure; it could not make one clear selection.")
            raise Uncertain(question, reason,
                            f"Rephrase the request with a specific action, room, or device. Review {STAGE_LABELS[question]} above.",
                            confidence=confidence, minimum_confidence=threshold)
        return choice

    async def process(self, prompt, language, snapshot, speaker_area=None):
        result = {
            "request": prompt, "language": language, "route": "smarthome",
            "service": None, "domain": None, "target": None,
            "decisions": {}, "usage": {}, "timings_ms": {},
        }
        try:
            await self._process(result, snapshot, (speaker_area or "").strip() or None)
        except Uncertain as error:
            result.update(route=UNSURE, answer=self.settings["unsure_reply"])
            self._failure(result, error.details)
        except ModelError as error:
            self._failure(result, model_failure(self.stage, error))
            raise
        return result

    async def _process(self, result, snapshot, speaker_area):
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
        def summarize_action(response):
            selected = read_choice(response, "service", services, config["service_confidence"])
            if selected not in actions:
                return {}
            service = actions[selected]["service"]
            metadata = {"selection": selected, "service": service, **snapshot["services"][service]}
            result["action_metadata"] = metadata
            return {"action_metadata": metadata}

        response = await self._decide(result, "service", {"prompt": prompt}, {
            "service": choice_question(services, config["service_prompt"]),
        }, summarize=summarize_action)
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
                if self.dry_run:
                    self.stage = "general"
                    raise ModelError("General MCP execution is disabled in plan-only benchmarks.")
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
        if service_choice != QUERY:
            action = actions[service_choice]
            result.update(
                service=action["service"], domain=action["service"].split(".", 1)[0],
                answer=f"Selected action: {action['service']}. No action was executed.",
            )
            result["speaker_area"] = speaker_area
            scopes = {key: value for key, value in TARGET_SCOPES.items()
                      if key not in ("entities_implied_area", "implied_area") or speaker_area}
            response = await self._decide(result, "target_scope", {
                "prompt": prompt, "service": result["service"], "speaker_area": speaker_area,
            }, {"target_scope": choice_question(scopes, config["target_scope_prompt"])})
            result["target_scope"] = self._choice(response, "target_scope", scopes)
            members = await self._select_action_targets(result, entities)
            await self._select_parameters(result, members)
            return

        result["service"] = QUERY
        areas = {f"area:{key}": value for key, value in areas_from(entities).items()}
        areas["all_areas"] = "All areas: the whole home, including exposed entities without a room assignment."
        area_prompt = config["area_prompt"].replace("Choose no_area when no room is identified. ", "")
        area_prompt += (
            " For state queries, choose all_areas when no specific room is identified or the whole home is requested. "
            "Examples: 'is the lamp Aurora on?', 'are there any windows open in the house?' and "
            "'sind irgendwo im Haus Fenster offen?'. Choose a specific room only when identified by the request or history. "
            "all_areas means search across all rooms, including entities without a room assignment; "
            "it does not mean every entity is requested. Any no_area guidance applies to all_areas for state queries; "
            "no_area is not an available choice. The whole home is one supported scope, not ambiguous multiple areas."
        )
        response = await self._decide(result, "area", {"prompt": prompt}, {
            "area": choice_question(areas, area_prompt),
        })
        area = self._choice(response, "area", areas)
        candidates = [entity for entity in entities
                      if area == "all_areas" or entity.get("area_id") == area.removeprefix("area:")]
        if not candidates:
            raise Uncertain("target", "No exposed devices match the selected room filter.",
                            "Check the device's room, Assist exposure settings, and GlowAssistant exclusions.",
                            area=areas[area])
        candidates = await self._select_domain(result, candidates, areas[area])
        query_types = {"yes_no": "A yes/no question about current states.",
                       "complex": "A question requesting values, names, counts or other details."}
        response = await self._decide(result, "query_type", {"prompt": prompt}, {
            "query_type": choice_question(query_types, config["query_type_prompt"]),
        })
        result["query_type"] = self._choice(response, "query_type", query_types)
        if result["query_type"] == "yes_no":
            await self._answer_yes_no(result, candidates, areas[area], language_instruction)
            return
        target_state = {"prompt": prompt, "service": QUERY, "domain": result["domain"], "area": areas[area]}
        members = await self._select_targets(result, candidates, target_state)
        result["state_info"] = [
            {"entity_id": entity["entity_id"], "domain": entity["domain"], "area": entity.get("area_name"),
             "name": entity["name"], "state": entity["state"], "attributes":
             entity.get("attributes", {}) if entity["state"] not in ("unknown", "unavailable") else {}}
            for entity in members
        ]
        facts = {key: value for key, value in result.items()
                 if key not in ("decisions", "usage", "timings_ms", "failures", "domain_filter", "target_selection",
                                "target", "state_info")}
        facts["entities"] = result["state_info"]
        if self.dry_run:
            language_instruction += " This is a plan-only benchmark. No device command was sent; describe actions as plans."
        await self._answer(result, [
            {"role": "system", "content": f"{self.settings['answer']['prompt']}\n{language_instruction}"},
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False, default=str)},
        ])

    async def _select_action_targets(self, result, entities):
        scope = result["target_scope"]
        areas = {f"area:{key}": {"area_id": key, **value} for key, value in areas_from(entities).items()}
        selected_areas = []
        result["area_source"] = "none"
        if scope in ("entities_implied_area", "implied_area"):
            location = result["speaker_area"].casefold()
            selected_areas = [area for area in areas.values()
                              if location in {str(value).casefold() for value in
                                              (area["area_id"], area["name"], *area.get("aliases", []))}]
            if len(selected_areas) != 1:
                raise Uncertain("action_areas", "The supplied speaker area does not match one exposed snapshot area.",
                                "Check the speaker's room assignment or --speaker-area. No other room was substituted.")
            result["area_source"] = "speaker_context"
        elif scope in ("entities_explicit_area", "areas"):
            if not areas:
                raise Uncertain("action_areas", "No areas with exposed entities are available.",
                                "Check room assignments and Assist exposure settings.")
            selected = await self._action_selection(result, "action_areas", areas,
                                                    single=scope == "entities_explicit_area")
            selected_areas = [areas[key] for key in selected]
            result["area_source"] = "decider"
        result["selected_areas"] = selected_areas
        area_ids = {area["area_id"] for area in selected_areas}
        domain_entities = [entity for entity in entities if entity["domain"] == result["domain"]]
        candidates = [entity for entity in domain_entities if not area_ids or entity.get("area_id") in area_ids]
        result["action_target_filter"] = {"entities_before": len(entities), "entities_after_domain": len(domain_entities),
                                          "entities_after_area": len(candidates), "area_ids": sorted(area_ids)}
        if not candidates:
            raise Uncertain("action_entities", "No exposed entities match the action domain and area filter.",
                            "Check the selected areas, entity domains and Assist exposure. No broader search was used.")
        if scope in ("areas", "implied_area"):
            result["target"] = ({"type": "area", **selected_areas[0]} if len(selected_areas) == 1 else
                                {"type": "areas", "area_ids": [area["area_id"] for area in selected_areas],
                                 "names": [area["name"] for area in selected_areas]})
            return candidates
        options = target_options(candidates)
        selected = await self._action_selection(result, "action_entities", options)
        result["target"] = (options[selected[0]] if len(selected) == 1 else
                            {"type": "entities", "entity_ids": [options[key]["entity_id"] for key in selected],
                             "names": [options[key]["name"] for key in selected]})
        selected_ids = {options[key]["entity_id"] for key in selected}
        return [entity for entity in candidates if entity["entity_id"] in selected_ids]

    async def _action_selection(self, result, stage, options, single=False):
        config = self.settings["decider"]
        kind = "area" if stage == "action_areas" else "entity"
        threshold = None if single else config[f"action_{kind}_probability_threshold"]

        def summarize(response):
            selection = decode_selection(response, stage, options, threshold)
            selection["selected_names"] = [options[key]["name"] for key in selection["selected_choices"]]
            result[stage] = selection
            return selection

        await self._decide(result, stage, {
            "prompt": result["request"], "service": result["service"], "target_scope": result["target_scope"],
            "selected_areas": result.get("selected_areas", []),
            **({"entity_counts": result["action_target_filter"]} if stage == "action_entities" else {}),
        }, {stage: selection_question(options, config[f"action_{kind}_prompt"])}, summarize=summarize)
        selected = result[stage]["selected_choices"]
        if not selected:
            raise Uncertain(stage, "No unambiguous action targets were retained.",
                            "Name the requested rooms or devices. Review the option probabilities and selection threshold.")
        return selected

    async def _select_parameters(self, result, members):
        config = self.settings["decider"]
        fields = action_fields(result["action_metadata"].get("fields") or {})
        started = perf_counter()
        self._trace("parameter_capabilities", {"target_entity_ids": [member["entity_id"] for member in members],
                                                "fields_before": len(fields)})
        fields, capabilities = filter_fields(fields, members)
        result["parameter_capabilities"] = capabilities
        duration = round((perf_counter() - started) * 1000)
        result["timings_ms"]["parameter_capabilities"] = duration
        self._trace("parameter_capabilities", {}, {"status": "completed", "model_call": False,
                                                   **capabilities, "duration_ms": duration})
        optional, payload = parameter_request(result["request"], result["service"], fields, config["parameter_prompt"])
        payload["state"]["capability_filter_policy"] = capabilities["policy"]
        if result.get("target"):
            payload["state"]["target"] = result["target"]
        result["required_parameters"] = [name for name, field in fields.items() if field.get("required", False)]
        if not optional:
            result["parameter_selection"] = {"selected_parameters": [], "probabilities": {},
                                             "probability_threshold": config["parameter_probability_threshold"]}
            return

        def summarize(response):
            selection = decode_parameters(response, optional, config["parameter_probability_threshold"])
            result["parameter_selection"] = selection
            return {**selection, "required_parameters": result["required_parameters"], "capability_filter": capabilities}

        await self._decide(result, "optional_parameters", payload["state"], payload["questions"], summarize=summarize)

    async def _answer_yes_no(self, result, entities, area, language_instruction):
        candidates = [
            {**target, "state": entity["state"], "attributes":
             entity.get("attributes", {}) if entity["state"] not in ("unknown", "unavailable") else {}}
            for target, entity in zip(target_options(entities).values(), entities, strict=True)
        ]
        options = {"yes": "The proposition in the user's question is true.",
                   "no": "The proposition in the user's question is false."}
        response = await self._decide(result, "binary_answer", {
            "prompt": result["request"], "area": area, "domain": result["domain"], "entities": candidates,
        }, {"binary_answer": choice_question(options, self.settings["decider"]["binary_answer_prompt"])})
        result["state_answer"] = self._choice(response, "binary_answer", options)
        await self._answer(result, [
            {"role": "system", "content": (
                f"{self.settings['answer']['prompt']}\n{language_instruction}\n"
                "For this reply, only translate the supplied state_answer (yes or no) into the Assist pipeline language. "
                "Output only that translated word with optional final punctuation. Do not explain, expand or reconsider it."
            )},
            {"role": "user", "content": json.dumps({"state_answer": result["state_answer"]})},
        ])

    async def _answer(self, result, messages):
        try:
            await self._chat(result, "answer", messages)
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

    async def _select_targets(self, result, entities, state):
        config = self.settings["decider"]
        targets = list(target_options(entities).values())
        current_states = {entity["entity_id"]: entity["state"] for entity in entities}
        targets = [{**target, "state": current_states[target["entity_id"]]} for target in targets]
        payload = target_request(state, targets, config["target_filter_prompt"])

        def summarize(response):
            selected = decode_targets(response, targets, config["target_filter_confidence"], config["query_target_keep_uncertain"])
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

    async def _select_domain(self, result, entities, area):
        config = self.settings["decider"]
        domains = {domain: domain for domain in sorted({entity["domain"] for entity in entities})}

        def summarize(response):
            domain = read_choice(response, "domain", domains, config["domain_confidence"])
            selected = {"selected_domain": None if domain == UNSURE else domain,
                        "kept_domains": [domain] if domain != UNSURE else [],
                        "entities_before": len(entities),
                        "entities_after": sum(entity["domain"] == domain for entity in entities)}
            result["domain_filter"] = selected
            return selected

        response = await self._decide(result, "domain", {"prompt": result["request"], "area": area}, {
            "domain": choice_question(domains, config["domain_prompt"]),
        }, summarize=summarize)
        domain = self._choice(response, "domain", domains)
        result["domain"] = domain
        return [entity for entity in entities if entity["domain"] == domain]
