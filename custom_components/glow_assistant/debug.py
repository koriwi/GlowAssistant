import json
import re

from .validation import finite_number, probability_cutoff

STAGE_LABELS = {
    "context": "Read Home Assistant devices",
    "service": "Choose an action",
    "optional_parameters": "Choose optional action parameters",
    "parameter_capabilities": "Filter action fields by target capabilities",
    "area": "Select query areas",
    "domain": "Choose a state-query domain",
    "query_type": "Choose a yes/no or detailed answer",
    "binary_answer": "Answer the yes/no state question",
    "state_property": "Select requested state properties",
    "target_scope": "Classify entity targets and explicit or implied areas",
    "action_areas": "Select action areas",
    "action_entities": "Select action entities",
    "target": "Choose a device or room",
    "text": "Extract text parameter",
    "numeric": "Extract numeric parameter",
    "select": "Extract select parameter",
    "parameters": "Extract action parameter values",
    "execution": "Call the Home Assistant action",
    "answer": "Generate the smart-home answer",
    "general": "Generate the general answer",
    "mcp_connect": "Connect to MCP servers",
    "mcp": "Call an MCP tool",
}


def failure_details(stage, reason, suggestion, status="error", **details):
    return {
        "status": status, "stage": stage, "step": STAGE_LABELS.get(stage.split(":", 1)[0], stage),
        "reason": reason, "suggestion": suggestion, **details,
    }


def failure_text(details):
    text = f"{details['step']}: {details['reason']} Try: {details['suggestion']}"
    if details.get("device_command") == "completed":
        text += " Home Assistant accepted the command; only the reply failed. Check the device before repeating it."
    elif details.get("device_command") in ("unconfirmed", "partially_completed"):
        text += " Some device effects may already have happened. Check the devices before repeating the request."
    return text


def model_failure(stage, error):
    message = str(error)
    details = getattr(error, "details", {})
    section = "Smart-home answer" if stage == "answer" else "General model" if stage in ("general", "mcp", "mcp_connect") else "Decider"
    http = re.fullmatch(r"Model endpoint returned HTTP (\d{3})\.", message)
    if http:
        code = int(http[1])
        reason = f"The model server returned an error (HTTP {code})."
        if code in (401, 403):
            suggestion = f"Check the API key and access permissions under {section}."
        elif code == 404:
            suggestion = f"Check the server URL and model name under {section}."
        elif code == 429:
            suggestion = "Check the model provider's usage limits, credits, and server load."
        else:
            suggestion = f"Check that the model server is running and the URL under {section} is correct."
        return failure_details(stage, reason, suggestion, http_status=code)
    if "output token limit" in message:
        reason = "The model reached its output token limit before completing the answer."
        suggestion = f"Increase Maximum output tokens under {section}."
    elif "input exceeds" in message:
        reason = "The request is larger than the configured input limit."
        if "input_chars" in details and "max_input_chars" in details:
            reason = (f"The request contains {details['input_chars']:,} characters; "
                      f"the configured limit is {details['max_input_chars']:,} characters ({section}).")
        suggestion = f"Increase Maximum input characters under {section}, or reduce the history window."
    elif message == "Model request timed out.":
        reason = "The model server did not reply within the configured time."
        suggestion = f"Check the server load or increase the request timeout under {section}."
    elif message == "Model connection failed.":
        reason = "The model server could not be reached."
        suggestion = f"Check the server URL under {section} and network access from Home Assistant."
    elif message == "Model endpoint returned invalid JSON.":
        reason = "The model server returned an unreadable response."
        suggestion = f"Check the endpoint URL and API compatibility under {section}."
    elif "classifier" in message:
        reason = "The decider returned an invalid decision."
        suggestion = "Check that Decider uses a native decision endpoint, such as /v1/systemone on llama.cpp."
    elif message.startswith("Malformed numeric extraction response"):
        reason = "The decider returned an invalid numeric value."
        suggestion = "Check the Decider endpoint and numeric extraction prompt. Review the numeric decisions."
    elif message.startswith("Malformed text extraction response"):
        reason = "The decider returned invalid or incomplete word selections."
        suggestion = "Check the Decider endpoint and text extraction prompt. Review the preceding word decisions."
    elif message.startswith("Malformed target selection response"):
        reason = "The decider returned invalid or incomplete target selections."
        suggestion = "Check the Decider endpoint, target selection prompt and thresholds. Review the preceding target decisions."
    elif "no spoken answer" in message:
        reason = "The model returned no text to speak."
        suggestion = f"Check the prompt and output token limit under {section}."
    elif "MCP" in message:
        if "limit" in message:
            reason = "The general request reached a configured MCP limit."
            suggestion = "Check the tool-round, tool-call, and result-size limits under General model."
        elif "unconfirmed" in message or "not retried" in message:
            reason = "An MCP operation failed; its outcome could not be confirmed."
            suggestion = "Check the MCP server and any affected state before repeating the request."
        elif "connection" in message or "discovery" in message or "cleanup" in message:
            reason = "The MCP server connection or tool discovery failed."
            suggestion = "Check the enabled MCP server URLs, bearer tokens, and HTTP support under General model."
        else:
            reason = "The MCP tool request or configuration was invalid."
            suggestion = "Check the tool names and allowlists under General model, and the preceding tool details."
    else:
        reason = "The model request failed or returned an unsupported response."
        suggestion = f"Check the endpoint, model, and prompt under {section}. See the Home Assistant logs for details."
    return failure_details(stage, reason, suggestion, settings_section=section, **details)


def decision_details(questions, answers, config):
    details = {}
    for name, question in questions.items():
        answer = answers.get(name, {}) if isinstance(answers, dict) else {}
        if not isinstance(answer, dict):
            answer = {}
        confidence = answer.get("confidence")
        confidence = confidence if finite_number(confidence) else None
        probability_fields = {
            "area": "area_confidence",
            "state_property": "state_property_confidence",
            "optional_parameters": "parameter_probability_threshold",
            "action_areas": "action_area_probability_threshold",
            "action_entities": "action_entity_probability_threshold",
            "query_targets": "query_target_probability_threshold",
            "text_words": "text_probability_threshold",
            "select_values": "select_probability_threshold",
        }
        if name in probability_fields:
            threshold = config[probability_fields[name]]
            option_count = len(set(question["criteria"]) - {"none", "unsure"})
            details[name] = {
                "type": question["type"], "choice": answer.get("choice"), "confidence": confidence,
                "probabilities": answer.get("probabilities", {}),
                "probability_threshold": threshold,
                "effective_probability_threshold": probability_cutoff(threshold, option_count),
                "real_option_count": option_count,
                "selection_basis": "relative_probability_cutoff; overall confidence is not a filter",
            }
            continue
        threshold = config["numeric_confidence"] if name in ("numeric_mode", "numeric_score") else config[f"{name}_confidence"]
        if question["type"] == "score":
            details[name] = {"type": "score", "score": answer.get("score"), "confidence": confidence,
                             "minimum_confidence": threshold, "probabilities": answer.get("probabilities", {})}
            continue
        info = {"type": question["type"], "confidence": confidence, "minimum_confidence": threshold}
        choice = answer.get("choice")
        choice = choice if isinstance(choice, str) else None
        probabilities = answer.get("probabilities")
        probabilities = probabilities if isinstance(probabilities, dict) else {}
        probability = probabilities.get(choice)
        info.update(
            choice=choice, probability=probability if finite_number(probability) else None,
            effective_choice=choice if confidence is not None and confidence >= threshold else "unsure",
        )
        criteria = question["criteria"]
        if choice in criteria:
            selected = criteria[choice]
            try:
                selected = json.loads(selected)
            except (ValueError, TypeError):
                pass
            info["selected"] = selected
        alternatives = sorted(
            ((label, value) for label, value in probabilities.items() if label != choice and finite_number(value)),
            key=lambda item: item[1], reverse=True,
        )[:3]
        info["alternatives"] = [{"choice": label, "probability": value} for label, value in alternatives]
        details[name] = info
    return details
