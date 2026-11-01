import math
from copy import deepcopy
from urllib.parse import urlsplit

from .action_targets import AREA_INSTRUCTIONS as ACTION_AREA_INSTRUCTIONS
from .action_targets import ENTITY_INSTRUCTIONS as ACTION_ENTITY_INSTRUCTIONS
from .const import (
    ANSWER_PROMPT,
    AREA_PROMPT,
    BINARY_ANSWER_PROMPT,
    DOMAIN_PROMPT,
    GENERAL_PROMPT,
    LEGACY_ACTION_TARGET_SCOPE_PROMPT,
    LEGACY_GENERAL_PROMPT,
    LEGACY_SERVICE_PROMPT,
    LEGACY_TARGET_SCOPE_PROMPT,
    LEGACY_TEXT_SERVICE_PROMPT,
    LEGACY_UNSCOPED_TARGET_SCOPE_PROMPT,
    QUERY_TARGET_PROMPT,
    QUERY_TYPE_PROMPT,
    SERVICE_PROMPT,
    TARGET_SCOPE_PROMPT,
)
from .optional_parameters import INSTRUCTIONS as PARAMETER_INSTRUCTIONS
from .parameter_values import (
    LEGACY_NUMERIC_INSTRUCTIONS,
    LEGACY_NUMERIC_MODE_INSTRUCTIONS,
    NUMERIC_INSTRUCTIONS,
    NUMERIC_MODE_INSTRUCTIONS,
)
from .text_values import INSTRUCTIONS as TEXT_INSTRUCTIONS
from .text_values import LEGACY_PROMPTS, MESSAGE_INSTRUCTIONS

DEFAULTS = {
    "name": "GlowAssistant",
    "decider": {
        "url": "https://openrouter.ai/api/alpha/decisions",
        "model": "typesafe/jev-1.14", "api_key": "", "timeout": 60,
        "max_input_chars": 100000,
        "service_prompt": SERVICE_PROMPT,
        "parameter_prompt": PARAMETER_INSTRUCTIONS, "parameter_probability_threshold": 0.15,
        "action_area_prompt": ACTION_AREA_INSTRUCTIONS, "action_area_probability_threshold": 0.15,
        "action_entity_prompt": ACTION_ENTITY_INSTRUCTIONS, "action_entity_probability_threshold": 0.15,
        "domain_prompt": DOMAIN_PROMPT, "domain_confidence": 0.5,
        "query_type_prompt": QUERY_TYPE_PROMPT, "query_type_confidence": 0,
        "query_target_prompt": QUERY_TARGET_PROMPT, "query_target_probability_threshold": 0.02,
        "binary_answer_prompt": BINARY_ANSWER_PROMPT, "binary_answer_confidence": 0,
        "area_prompt": AREA_PROMPT,
        "target_scope_prompt": TARGET_SCOPE_PROMPT, "target_scope_confidence": 0,
        "text_prompt": TEXT_INSTRUCTIONS, "message_prompt": MESSAGE_INSTRUCTIONS,
        "text_probability_threshold": 0.3,
        "numeric_prompt": NUMERIC_INSTRUCTIONS, "numeric_mode_prompt": NUMERIC_MODE_INSTRUCTIONS, "numeric_confidence": 0,
        "service_confidence": 0, "area_confidence": 0,
    },
    "answer": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "", "api_key": "", "timeout": 60, "max_input_chars": 100000,
        "max_tokens": 256, "temperature": 0.2, "prompt": ANSWER_PROMPT,
    },
    "general": {
        "enable": False, "url": "", "model": "", "api_key": "",
        "timeout": 60, "max_input_chars": 100000, "max_tokens": 2048,
        "temperature": 0.2, "prompt": GENERAL_PROMPT,
        "mcp_servers": [], "mcp_max_rounds": 5, "mcp_max_calls": 10, "mcp_max_result_chars": 20000,
    },
    "history_enabled": False,
    "history_messages": 10,
    "excluded_entities": [],
    "unsure_reply": "ich bin mir nicht sicher",
    "error_reply": "Ich konnte die Anfrage nicht erfolgreich abschließen.",
}


def normalize_settings(data):
    settings = deepcopy(DEFAULTS)
    for key, default in DEFAULTS.items():
        if isinstance(default, dict):
            settings[key].update({field: value for field, value in data.get(key, {}).items() if field in default})
            for field in default:
                value = settings[key][field]
                if isinstance(value, str):
                    value = value.strip()
                    if not value and (field == "prompt" or field.endswith("_prompt")):
                        value = default[field]
                    settings[key][field] = value
        elif key in data:
            settings[key] = data[key]
    if settings["general"]["prompt"] == LEGACY_GENERAL_PROMPT:
        settings["general"]["prompt"] = GENERAL_PROMPT
    if settings["decider"]["service_prompt"] in (LEGACY_SERVICE_PROMPT, LEGACY_TEXT_SERVICE_PROMPT):
        settings["decider"]["service_prompt"] = SERVICE_PROMPT
    decider = data.get("decider", {})
    if settings["decider"]["target_scope_prompt"] in (
        LEGACY_TARGET_SCOPE_PROMPT, LEGACY_ACTION_TARGET_SCOPE_PROMPT, LEGACY_UNSCOPED_TARGET_SCOPE_PROMPT,
    ):
        settings["decider"]["target_scope_prompt"] = TARGET_SCOPE_PROMPT
    if "domain_confidence" not in decider and "domain_filter_confidence" in decider:
        settings["decider"]["domain_confidence"] = decider["domain_filter_confidence"]
    for field, old in (("numeric_prompt", LEGACY_NUMERIC_INSTRUCTIONS), ("numeric_mode_prompt", LEGACY_NUMERIC_MODE_INSTRUCTIONS)):
        if settings["decider"][field] == old:
            settings["decider"][field] = DEFAULTS["decider"][field]
    for field in ("text_prompt", "message_prompt"):
        if settings["decider"][field] in LEGACY_PROMPTS:
            settings["decider"][field] = DEFAULTS["decider"][field]
    if "message_prompt" not in decider and "text_prompt" in decider and decider["text_prompt"].strip() not in (*LEGACY_PROMPTS, TEXT_INSTRUCTIONS, ""):
        settings["decider"]["message_prompt"] = settings["decider"]["text_prompt"]
    return settings


def validate_settings(data):
    settings = normalize_settings(data)
    for group in ("decider", "answer", "general"):
        config = settings[group]
        if group != "general" or config["enable"]:
            parsed = urlsplit(config["url"])
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError(f"{group}: enter a full HTTP(S) endpoint URL without embedded credentials.")
            if not config["model"]:
                raise ValueError(f"{group}: model ID is required.")
            if parsed.hostname == "openrouter.ai" and not config["api_key"]:
                raise ValueError(f"{group}: an OpenRouter API key is required.")
        _range(config, "timeout", 1, 600, group)
        _range(config, "max_input_chars", 100, 2000000, group, integer=True)
        if group != "decider":
            _range(config, "max_tokens", 1, 100000, group, integer=True)
            _range(config, "temperature", 0, 2, group)
    decider = settings["decider"]
    for field in ("parameter_probability_threshold", "action_area_probability_threshold", "action_entity_probability_threshold",
                  "query_target_probability_threshold", "text_probability_threshold"):
        _range(decider, field, 0, 1, "decider")
    for stage in ("service", "area", "domain", "query_type", "binary_answer", "target_scope", "numeric"):
        _range(decider, f"{stage}_confidence", 0, 1, "decider")
    if not isinstance(settings["history_enabled"], bool):
        raise TypeError("history_enabled must be a boolean.")
    _range(settings, "history_messages", 1, 100, "history", integer=True)
    general = settings["general"]
    _range(general, "mcp_max_rounds", 1, 20, "general", integer=True)
    _range(general, "mcp_max_calls", 1, 100, "general", integer=True)
    _range(general, "mcp_max_result_chars", 100, 1000000, "general", integer=True)
    if not isinstance(general["mcp_servers"], list):
        raise TypeError("general: MCP servers must be a list.")
    servers = []
    names = set()
    for raw in general["mcp_servers"]:
        if not isinstance(raw, dict):
            raise TypeError("general: each MCP server must be an object.")
        server = {"name": "", "url": "", "token": "", "enable": False, "timeout": 30, "tools": [], **raw}
        for field in ("name", "url", "token"):
            if not isinstance(server[field], str):
                raise TypeError(f"general: MCP {field} must be text.")
            server[field] = server[field].strip()
        if not server["name"] or server["name"] in names:
            raise ValueError("general: MCP server names must be nonempty and unique.")
        if not isinstance(server["enable"], bool):
            raise TypeError("general: MCP server enable must be a boolean.")
        parsed = urlsplit(server["url"])
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("general: MCP servers require HTTP(S) URLs without embedded credentials.")
        _range(server, "timeout", 1, 600, "MCP server")
        if not isinstance(server["tools"], list) or any(not isinstance(tool, str) or not tool.strip() for tool in server["tools"]):
            raise ValueError("general: allowed MCP tools must be a list of nonempty names.")
        server["tools"] = [tool.strip() for tool in server["tools"]]
        names.add(server["name"])
        servers.append(server)
    general["mcp_servers"] = servers
    for field in ("name", "unsure_reply", "error_reply"):
        if not isinstance(settings[field], str) or not settings[field].strip():
            raise ValueError(f"{field}: text must not be empty.")
    return settings


def _range(config, field, minimum, maximum, group, integer=False):
    value = config[field]
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(value) or not minimum <= value <= maximum
        or (integer and int(value) != value)):
        raise ValueError(f"{group}: {field} must be {'an integer' if integer else 'a number'} from {minimum} to {maximum}.")
    if integer:
        config[field] = int(value)
