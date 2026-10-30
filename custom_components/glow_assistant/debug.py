import json
import re

from .numeric_values import finite_number

STAGE_LABELS = {
    "context": "Read Home Assistant devices",
    "service": "Choose an action",
    "area": "Choose a room",
    "domain_filter": "Filter whole-home entity domains",
    "target": "Choose a device or room",
    "numeric": "Choose a new value",
    "numeric_mode": "Choose a new value",
    "text": "Extract item or message text",
    "execution": "Send the device command",
    "answer": "Generate the smart-home answer",
    "general": "Generate the general answer",
    "mcp_connect": "Connect to MCP servers",
    "mcp": "Call an MCP tool",
}


def failure_details(stage, reason, suggestion, status="error", **details):
    return {
        "status": status, "stage": stage, "step": STAGE_LABELS.get(stage, stage),
        "reason": reason, "suggestion": suggestion, **details,
    }


def failure_text(details):
    text = f"{details['step']}: {details['reason']} Try: {details['suggestion']}"
    if details.get("device_command") == "completed":
        text += " Home Assistant accepted the device command; only the spoken answer failed. Check the device before repeating the command."
    elif details.get("device_command") == "unconfirmed":
        text += " The device outcome is unconfirmed. Check its state before repeating the command."
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
    elif "classifier" in message or message == "Malformed numeric response.":
        reason = "The decider returned an invalid decision."
        suggestion = "Check that Decider uses a native decision endpoint, such as /v1/systemone on llama.cpp."
    elif message.startswith("Malformed text extraction response"):
        reason = "The decider returned invalid or incomplete word selections."
        suggestion = "Check the Decider endpoint and text extraction prompt. Review the preceding word decisions."
    elif message.startswith("Malformed domain filter response"):
        reason = "The decider returned invalid or incomplete domain selections."
        suggestion = "Check the Decider endpoint and whole-home domain filter prompt. Review the preceding domain decisions."
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
        text_word = name.startswith("word_")
        domain_filter = name.startswith("domain_")
        if text_word:
            confidence_stage = "text"
        elif domain_filter:
            confidence_stage = "domain_filter"
        elif name in ("service", "area", "target"):
            confidence_stage = name
        else:
            confidence_stage = "numeric"
        threshold = config[f"{confidence_stage}_confidence"]
        info = {"type": question["type"], "confidence": confidence, "minimum_confidence": threshold}
        if question["type"] == "choice":
            choice = answer.get("choice")
            choice = choice if isinstance(choice, str) else None
            probabilities = answer.get("probabilities")
            probabilities = probabilities if isinstance(probabilities, dict) else {}
            probability = probabilities.get(choice)
            info.update(
                choice=choice, probability=probability if finite_number(probability) else None,
                effective_choice=choice if confidence is not None and confidence >= threshold else "unsure",
            )
            if text_word:
                info["kept"] = choice == "keep" and confidence is not None and confidence > threshold
                info["effective_choice"] = "keep" if info["kept"] else "drop"
            elif domain_filter:
                info["kept"] = choice == "keep" or confidence is not None and confidence <= threshold
                info["effective_choice"] = "keep" if info["kept"] else "drop"
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
        else:
            score = answer.get("score")
            info["score"] = score if finite_number(score) else None
        details[name] = info
    return details
