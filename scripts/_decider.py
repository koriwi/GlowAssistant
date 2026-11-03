import json
import os
from builtins import BaseExceptionGroup
from urllib.parse import urlsplit

MODEL = "typesafe/jev-1.14"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
UNSURE = "unsure"


def setting(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing {name}; set it in .env or the environment.")
    return value


def decider_url():
    url = os.environ.get("DECIDER_URL", "").strip() or DECISIONS_URL
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("DECIDER_URL must be a full HTTP(S) endpoint URL.")
    return url


def decider_model():
    return os.environ.get("DECIDER_MODEL", "").strip() or MODEL


def decider_api_key():
    key = os.environ.get("DECIDER_API_KEY", "").strip()
    if not key and urlsplit(decider_url()).hostname == "openrouter.ai":
        key = setting("OPENROUTER_API_KEY")
    return key or None


def choice_question(options, instructions, unsure):
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {
            **{key: json.dumps(value, ensure_ascii=False) for key, value in options.items()},
            UNSURE: unsure,
        },
    }


async def call_decider(payload, api_key=None):
    import httpx

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            decider_url(),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            json=payload,
        )
        if response.is_error:
            raise ValueError(f"Decider HTTP {response.status_code}: {response.text[:500]}")
        return response.json()


def selection_result(response, options, question="target"):
    if not isinstance(response, dict):
        raise ValueError("Expected a JSON object from the decider.")
    if "error" in response:
        raise ValueError(f"Decider error: {response['error']}")
    try:
        answer = response["answers"][question]
        choice = answer["choice"]
        confidence = answer["confidence"]
        probabilities = answer["probabilities"]
        labels = set(options) | {UNSURE}
        valid = (
            answer["type"] == "choice"
            and choice in labels
            and isinstance(probabilities, dict)
            and set(probabilities) == labels
            and all(
                isinstance(score, (int, float)) and not isinstance(score, bool)
                and 0 <= score <= 1
                for score in [confidence, *probabilities.values()]
            )
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError("Invalid decider choice response or unknown candidate labels.")
    alternatives = sorted(probabilities, key=probabilities.get, reverse=True)
    return {
        "model": response.get("model", decider_model()),
        "candidate_count": len(options),
        "choice": choice,
        question: options.get(choice),
        "probability": probabilities[choice],
        "confidence": confidence,
        "alternatives": [
            {"choice": key, "probability": probabilities[key], question: options.get(key)}
            for key in alternatives if key != choice
        ][:3],
    }


def error_message(error):
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(error_message(child) for child in error.exceptions)
    if isinstance(error, ModuleNotFoundError):
        return "Dependencies missing; install requirements.txt first."
    return str(error) or type(error).__name__
