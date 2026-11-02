from .optional_parameters import action_fields


class ActionError(Exception):
    def __init__(self, message, command_sent=False):
        super().__init__(message)
        self.command_sent = command_sent


def action_calls(result, members):
    parameters = result.get("parameters", {})
    if set(parameters) & {"entity_id", "area_id", "device_id", "floor_id", "label_id"}:
        raise ActionError("Target selector fields cannot be supplied as action parameters.")
    fields = action_fields(result["action_metadata"].get("fields") or {})
    required = {name for name, field in fields.items() if field.get("required")}
    missing = required - parameters.keys()
    if missing:
        raise ActionError("Missing required action parameters: " + ", ".join(sorted(missing)))
    support = result["parameter_capabilities"]["field_support"]
    groups = {}
    for member in members:
        entity_id = member["entity_id"]
        if member["domain"] != result["domain"]:
            raise ActionError("An action target does not match the selected domain.")
        names = tuple(name for name in parameters if entity_id in support[name]["supported_entity_ids"])
        if not required <= set(names):
            raise ActionError(f"Target '{member['name']}' does not support every required action parameter.")
        groups.setdefault(names, []).append(entity_id)
    if not groups:
        raise ActionError("No exposed targets are available for this action.")
    return [{"service": result["service"], "arguments": {
        **{name: parameters[name] for name in names}, "entity_id": entity_ids,
    }, "status": "planned"} for names, entity_ids in groups.items()]


def device_outcome(execution):
    if not execution:
        return "not_sent"
    calls = execution.get("calls", [])
    if any(call["status"] == "unconfirmed" for call in calls):
        return "unconfirmed"
    if execution["status"] == "completed":
        return "completed"
    if any(call["status"] == "completed" for call in calls):
        return "partially_completed"
    return "not_sent"
