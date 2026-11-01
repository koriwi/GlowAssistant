import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import aiohttp
from dotenv import load_dotenv

from custom_components.glow_assistant.client import ModelClient, ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.settings import (
    DEFAULTS,
    normalize_settings,
    validate_settings,
)

CONFIDENCE_FIELDS = tuple(key for key in DEFAULTS["decider"] if key.endswith("_confidence"))


def confidence(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("Confidence must be between 0 and 1.")
    return value


def parser():
    result = argparse.ArgumentParser(description="Benchmark the shared HA engine locally; device actions are planned only.")
    result.add_argument("--query", "-q", required=True)
    result.add_argument("--confidence", type=confidence, help="Override every decider confidence threshold.")
    for field in CONFIDENCE_FIELDS:
        result.add_argument("--" + field.replace("_", "-"), type=confidence)
    result.add_argument("--language", help="Assist pipeline language to simulate; defaults to BENCHMARK_LANGUAGE or de.")
    result.add_argument("--json", action="store_true", help="Print complete decisions, trace and timings.")
    return result


def model_settings(exported, args, env):
    settings = normalize_settings(exported)
    for group, prefix in (("decider", "DECIDER"), ("answer", "ANSWER"), ("general", "GENERAL")):
        for field in ("url", "model", "api_key"):
            value = env.get(f"{prefix}_{field.upper()}", "").strip()
            if value:
                settings[group][field] = value
    decider = settings["decider"]
    if not decider["api_key"] and urlsplit(decider["url"]).hostname == "openrouter.ai":
        decider["api_key"] = env.get("OPENROUTER_API_KEY", "").strip()
    for field in CONFIDENCE_FIELDS:
        if args.confidence is not None:
            decider[field] = args.confidence
        specific = getattr(args, field)
        if specific is not None:
            decider[field] = specific
    return validate_settings(settings)


async def fetch_snapshot(session, env):
    url = env.get("HA_URL", "").strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Set HA_URL to your Home Assistant base URL, without credentials or query parameters.")
    token = env.get("HA_TOKEN", "").strip()
    if not token:
        raise ValueError("Set HA_TOKEN to a Home Assistant administrator token in .env.")
    entry_id = env.get("GLOW_ENTRY_ID", "").strip()
    try:
        async with session.get(
            url + "/api/glow_assistant/snapshot", headers={"Authorization": f"Bearer {token}"},
            params={"entry_id": entry_id} if entry_id else {},
            timeout=aiohttp.ClientTimeout(total=30), allow_redirects=False,
        ) as response:
            if response.status in (401, 403):
                raise ModelError("Snapshot access denied; check HA_TOKEN and administrator access.")
            if response.status == 404:
                raise ModelError("Snapshot endpoint or entry unavailable; update/restart GlowAssistant and check GLOW_ENTRY_ID.")
            if response.status != 200:
                raise ModelError(f"Snapshot endpoint returned HTTP {response.status}; check GLOW_ENTRY_ID if multiple entries exist.")
            data = await response.json()
    except TimeoutError as error:
        raise ModelError("Home Assistant snapshot request timed out.") from error
    except aiohttp.ClientError as error:
        raise ModelError("Home Assistant snapshot connection failed.") from error
    if (not isinstance(data, dict) or not isinstance(data.get("entry_id"), str)
        or not isinstance(data.get("snapshot"), dict)
        or not isinstance(data.get("settings"), dict)
        or not isinstance(data["snapshot"].get("entities"), list)
        or not isinstance(data["snapshot"].get("services"), dict)):
        raise ModelError("Snapshot endpoint returned an invalid snapshot or settings.")
    return data


async def run(args, env):
    started = perf_counter()
    trace = []

    def record(stage, arguments, result=None):
        trace.append({"stage": stage, "input": arguments} if result is None else {"stage": stage, "output": result})

    async def no_execution(action, arguments):
        raise RuntimeError("A plan-only benchmark must not execute device actions.")

    async with aiohttp.ClientSession() as session:
        exported = await fetch_snapshot(session, env)
        snapshot_ms = round((perf_counter() - started) * 1000)
        settings = model_settings(exported["settings"], args, env)
        language = args.language or env.get("BENCHMARK_LANGUAGE", "").strip() or "de"
        engine = AgentEngine(settings, ModelClient(session), no_execution, trace=record, dry_run=True)
        engine_started = perf_counter()
        try:
            result = await engine.process(args.query, language, exported["snapshot"])
        except ModelError as error:
            result = {"request": args.query, "error": str(error), "details": error.details}
        engine_ms = round((perf_counter() - engine_started) * 1000)
    return {
        "mode": "plan_only", "entry_id": exported["entry_id"], "language": language,
        "context": {"entities": len(exported["snapshot"]["entities"]), "actions": len(exported["snapshot"]["services"])},
        "result": result, "trace": trace,
        "timings_ms": {"snapshot": snapshot_ms, "engine": engine_ms, "total": round((perf_counter() - started) * 1000)},
    }


def print_report(report):
    result = report["result"]
    print("mode: plan only; no device actions")
    print(f"context: {report['context']['entities']} entities; {report['context']['actions']} actions")
    for field in ("route", "service", "domain"):
        if field in result:
            print(f"{field}: {result[field]}")
    if result.get("target"):
        target = result["target"]
        name = target.get("name") or ", ".join(target.get("names", []))
        identifier = target.get("entity_id") or target.get("area_id")
        print(f"target: {target['type']}; {name}" + (f" ({identifier})" if identifier else ""))
    for event in report["trace"]:
        output = event.get("output", {})
        if "duration_ms" not in output:
            continue
        decisions = output.get("decisions", {})
        if decisions and all(info.get("choice") in ("keep", "drop") for info in decisions.values()):
            kept = sum(info.get("kept", False) for info in decisions.values())
            detail = f"keep {kept}/{len(decisions)}"
        else:
            choices = []
            for name, info in decisions.items():
                value = info.get("effective_choice", info.get("score", info.get("choice")))
                score = info.get("confidence")
                choices.append(f"{name}={value}" + (f" ({score:.3f})" if isinstance(score, (int, float)) else ""))
            detail = ", ".join(choices) or output.get("status", "")
        print(f"{event['stage']}: {output['duration_ms']} ms; {detail}")
    if result.get("execution"):
        execution = result["execution"]
        print(f"execution: {execution['status']}; " + json.dumps(execution.get("arguments", {}), ensure_ascii=False))
    for field in ("answer", "error"):
        if field in result:
            print(f"{field}: {result[field]}")
    for failure in result.get("failures", []):
        print(f"failure [{failure['stage']}]: {failure['reason']}")
    timings = report["timings_ms"]
    print(f"total: {timings['total']} ms; snapshot: {timings['snapshot']} ms; engine: {timings['engine']} ms")


def main(argv=None):
    load_dotenv(Path(__file__).with_name(".env"))
    arguments = parser()
    args = arguments.parse_args(argv)
    if not args.query.strip():
        arguments.error("--query must not be empty.")
    try:
        report = asyncio.run(run(args, os.environ))
    except (ModelError, ValueError, OSError) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(report)
    return int("error" in report["result"])


if __name__ == "__main__":
    raise SystemExit(main())
