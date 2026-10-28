import json
import math
from time import perf_counter

from .client import ModelError
from .const import GENERAL, QUERY, UNSURE
from .numeric_values import decode_numeric, finite_number, numeric_request


class Uncertain(Exception):
    pass


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
            "aliases": entity.get("aliases", []), "area": entity.get("area_name"),
            "device_class": attributes.get("device_class"),
            "unit": attributes.get("unit_of_measurement"),
        }
    for area_id, area in areas_from(entities).items():
        options[f"area:{area_id}"] = {"type": "area", "area_id": area_id, **area}
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
    def __init__(self, settings, client, execute):
        self.settings = settings
        self.client = client
        self.execute = execute

    async def _decide(self, result, stage, state, questions):
        started = perf_counter()
        response = await self.client.decide(self.settings["decider"], state, questions)
        result["timings_ms"][stage] = round((perf_counter() - started) * 1000)
        result["usage"][stage] = response.get("usage", {})
        result["decisions"][stage] = response.get("answers", {})
        return response

    def _choice(self, response, question, options):
        threshold = self.settings["decider"][f"{question if question != 'numeric_mode' else 'numeric'}_confidence"]
        return read_choice(response, question, options, threshold)

    async def process(self, prompt, language, snapshot):
        result = {
            "request": prompt, "language": language, "route": "smarthome",
            "service": None, "domain": None, "target": None,
            "decisions": {}, "usage": {}, "timings_ms": {},
        }
        try:
            await self._process(result, snapshot)
        except Uncertain:
            result.update(route=UNSURE, answer=self.settings["unsure_reply"])
        return result

    async def _process(self, result, snapshot):
        prompt = result["request"]
        config = self.settings["decider"]
        entities = snapshot["entities"]
        actions = action_options(snapshot["services"])
        services = {
            **actions, QUERY: "Query current home states or measurements, no action.",
            GENERAL: "General knowledge or conversation unrelated to the home.",
        }
        domains = {entity["domain"]: entity["domain"] for entity in entities}
        response = await self._decide(result, "service", {"prompt": prompt}, {
            "service": choice_question(services, config["service_prompt"]),
            "domain": choice_question(domains, config["domain_prompt"]),
        })
        service_choice = self._choice(response, "service", services)
        if service_choice == UNSURE:
            raise Uncertain
        if service_choice == GENERAL:
            if not self.settings["general"]["enable"]:
                raise Uncertain
            result["route"] = GENERAL
            await self._chat(result, "general", [
                {"role": "system", "content": self.settings["general"]["prompt"]},
                {"role": "user", "content": prompt},
            ])
            return
        query = service_choice == QUERY
        action = actions.get(service_choice)
        domain = self._choice(response, "domain", domains) if query else action["service"].split(".", 1)[0]
        if domain == UNSURE:
            raise Uncertain
        result["service"] = QUERY if query else action["service"]
        result["domain"] = domain
        areas = {**{f"area:{key}": value for key, value in areas_from(entities).items()},
                 "no_area": "No specific area is identified."}
        response = await self._decide(result, "area", {"prompt": prompt}, {
            "area": choice_question(areas, config["area_prompt"]),
        })
        area = self._choice(response, "area", areas)
        if area == UNSURE:
            raise Uncertain
        candidates = [entity for entity in entities if entity["domain"] == domain
                      and (area == "no_area" or entity.get("area_id") == area.removeprefix("area:"))]
        if not candidates:
            raise Uncertain
        options = target_options(candidates)
        response = await self._decide(result, "target", {
            "prompt": prompt, "service": result["service"], "domain": domain,
            "area": areas[area] if area != "no_area" else None,
        }, {"target": choice_question(options, config["target_prompt"])})
        target = self._choice(response, "target", options)
        if target == UNSURE:
            raise Uncertain
        result["target"] = options[target]
        if result["target"]["type"] == "entity":
            members = [entity for entity in candidates if entity["entity_id"] == result["target"]["entity_id"]]
        else:
            members = [entity for entity in candidates if entity.get("area_id") == result["target"]["area_id"]]
        if not members:
            raise Uncertain
        if query:
            result["state_info"] = [
                {"name": entity["name"], "state": entity["state"], "attributes":
                 entity.get("attributes", {}) if entity["state"] not in ("unknown", "unavailable") else {}}
                for entity in members
            ]
        else:
            arguments = {"entity_id": [entity["entity_id"] for entity in members]}
            if action["parameter"]:
                arguments[action["parameter"]] = await self._numeric(result, members, action["parameter"])
            try:
                await self.execute(action["service"], arguments)
            except ActionError:
                result["execution"] = {"status": "error", "error": "Action failed; device outcome unconfirmed."}
            else:
                result["execution"] = {"status": "completed", "arguments": arguments}
        facts = {key: value for key, value in result.items()
                 if key not in ("decisions", "usage", "timings_ms")}
        try:
            await self._chat(result, "answer", [
                {"role": "system", "content": self.settings["answer"]["prompt"]},
                {"role": "user", "content": json.dumps(facts, ensure_ascii=False, default=str)},
            ])
        except ModelError as error:
            completed = result.get("execution", {}).get("status") == "completed"
            result["answer"] = self.settings["answer_error_reply" if completed else "error_reply"]
            result["answer_error"] = str(error)

    async def _chat(self, result, stage, messages):
        started = perf_counter()
        try:
            result["answer"], result["usage"][stage] = await self.client.chat(self.settings[stage], messages)
        finally:
            result["timings_ms"][stage] = round((perf_counter() - started) * 1000)

    async def _numeric(self, result, members, service_parameter):
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
                raise Uncertain
            unit = units.pop()
            for entity in members:
                attributes = entity.get("attributes", {})
                minimum = max(minimum, attributes.get("min_temp", minimum))
                maximum = min(maximum, attributes.get("max_temp", maximum))
                step = max(step, attributes.get("target_temp_step", step))
        elif any(set(entity.get("attributes", {}).get("supported_color_modes", [])) == {"onoff"} for entity in members):
            raise Uncertain
        if minimum >= maximum:
            raise Uncertain
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
        if mode == UNSURE or (mode == "relative" and current is None):
            raise Uncertain
        try:
            estimate = decode_numeric(response, anchors, method="score")["score"]
        except ValueError as error:
            raise ModelError("Malformed numeric response.") from error
        if estimate["confidence"] < config["numeric_confidence"]:
            raise Uncertain
        raw = estimate["value"]
        value = math.floor(raw / step + 0.5) * step
        if mode == "relative":
            if math.isclose(raw, current):
                raise Uncertain
            if raw > current and value <= current:
                value = (math.floor(current / step) + 1) * step
            elif raw < current and value >= current:
                value = (math.ceil(current / step) - 1) * step
        if not minimum <= value <= maximum:
            raise Uncertain
        for entity in members:
            if parameter == "temperature":
                attributes = entity.get("attributes", {})
                device_step = attributes.get("target_temp_step", step)
                if not math.isclose(value / device_step, round(value / device_step), abs_tol=1e-6):
                    raise Uncertain
        return int(value) if float(value).is_integer() else round(value, 6)
