import math
from copy import deepcopy
from urllib.parse import urlsplit

from .const import (
    ANSWER_PROMPT,
    AREA_PROMPT,
    DOMAIN_PROMPT,
    GENERAL_PROMPT,
    NUMERIC_MODE_PROMPT,
    SERVICE_PROMPT,
    TARGET_PROMPT,
)
from .numeric_values import INSTRUCTIONS

DEFAULTS = {
    "name": "GlowAssistant",
    "decider": {
        "url": "https://openrouter.ai/api/alpha/decisions",
        "model": "typesafe/jev-1.14", "api_key": "", "timeout": 60,
        "max_input_chars": 100000,
        "service_prompt": SERVICE_PROMPT, "domain_prompt": DOMAIN_PROMPT,
        "area_prompt": AREA_PROMPT, "target_prompt": TARGET_PROMPT,
        "numeric_prompt": INSTRUCTIONS, "numeric_mode_prompt": NUMERIC_MODE_PROMPT,
        "service_confidence": 0, "domain_confidence": 0, "area_confidence": 0,
        "target_confidence": 0, "numeric_confidence": 0,
        "temperature_min": 16, "temperature_max": 26, "temperature_levels": 9,
        "temperature_step": 1, "temperature_default_change": 0.5,
        "temperature_baseline": "target",
        "brightness_min": 0, "brightness_max": 100, "brightness_levels": 6,
        "brightness_step": 1, "brightness_default_change": 10,
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
    },
    "excluded_entities": [],
    "unsure_reply": "ich bin mir nicht sicher",
    "error_reply": "Ich konnte die Anfrage nicht erfolgreich abschließen.",
    "answer_error_reply": "Der Befehl wurde übermittelt, aber ich konnte keine Antwort formulieren.",
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
    for stage in ("service", "domain", "area", "target", "numeric"):
        _range(decider, f"{stage}_confidence", 0, 1, "decider")
    for parameter in ("temperature", "brightness"):
        lower = -100 if parameter == "temperature" else 0
        upper = 200 if parameter == "temperature" else 100
        for field in ("min", "max"):
            _range(decider, f"{parameter}_{field}", lower, upper, "decider")
        if decider[f"{parameter}_min"] >= decider[f"{parameter}_max"]:
            raise ValueError(f"decider: {parameter} minimum must be smaller than maximum.")
        _range(decider, f"{parameter}_levels", 2, 10, "decider", integer=True)
        _range(decider, f"{parameter}_step", 0.01, 100, "decider")
        _range(decider, f"{parameter}_default_change", 0.01, 100, "decider")
    if decider["temperature_baseline"] not in ("target", "measured"):
        raise ValueError("decider: unknown temperature baseline.")
    for field in ("name", "unsure_reply", "error_reply", "answer_error_reply"):
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
