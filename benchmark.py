import argparse
import asyncio
import json
import os
import shutil
import sys
import textwrap
from itertools import zip_longest
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
    result.add_argument("--parameter-threshold", type=confidence,
                        help="Keep optional action parameters above this probability; independent of --confidence.")
    result.add_argument("--action-area-threshold", type=confidence, help="Probability threshold for selecting multiple action areas.")
    result.add_argument("--action-entity-threshold", type=confidence, help="Probability threshold for selecting action entities.")
    result.add_argument("--query-target-threshold", type=confidence,
                        help="Probability threshold for state-query entity selection; independent of --confidence.")
    for field in CONFIDENCE_FIELDS:
        flags = ["--" + field.replace("_", "-")]
        if field == "domain_confidence":
            flags.append("--domain-filter-confidence")
        result.add_argument(*flags, type=confidence)
    result.add_argument("--language", help="Assist pipeline language to simulate; defaults to BENCHMARK_LANGUAGE or de.")
    result.add_argument("--speaker-area", help="Optional speaker room, e.g. Küche; omitted means location unknown.")
    result.add_argument("--json", action="store_true", help="Print complete decisions, trace and timings.")
    result.add_argument("--show-answer-input", action="store_true", help="Print the exact smart-home answer request, including messages.")
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
    if args.parameter_threshold is not None:
        decider["parameter_probability_threshold"] = args.parameter_threshold
    if args.query_target_threshold is not None:
        decider["query_target_probability_threshold"] = args.query_target_threshold
    for kind in ("area", "entity"):
        value = getattr(args, f"action_{kind}_threshold")
        if value is not None:
            decider[f"action_{kind}_probability_threshold"] = value
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
            result = await engine.process(args.query, language, exported["snapshot"], speaker_area=args.speaker_area)
        except ModelError as error:
            result = {"request": args.query, "error": str(error), "details": error.details}
        engine_ms = round((perf_counter() - engine_started) * 1000)
    return {
        "mode": "plan_only", "entry_id": exported["entry_id"], "language": language,
        "context": {"entities": len(exported["snapshot"]["entities"]), "actions": len(exported["snapshot"]["services"])},
        "result": result, "trace": trace,
        "timings_ms": {"snapshot": snapshot_ms, "engine": engine_ms, "total": round((perf_counter() - started) * 1000)},
    }


def print_table(headers, rows, limits, right_columns=()):
    rows = [[str(value) for value in row] for row in rows]
    widths = [min(limit, max(len(row[index]) for row in [headers, *rows])) for index, limit in enumerate(limits)]

    def line(cells):
        return "  ".join(cell.rjust(width) if index in right_columns else cell.ljust(width)
                         for index, (cell, width) in enumerate(zip(cells, widths, strict=True))).rstrip()

    print(line(headers))
    print(line(["-" * width for width in widths]))
    for row in rows:
        cells = [textwrap.wrap(value, width=width) or [""] for value, width in zip(row, widths, strict=True)]
        for parts in zip_longest(*cells, fillvalue=""):
            print(line(parts))


def answer_input(report, readable=False):
    for event in report["trace"]:
        data = event.get("input", {})
        if event["stage"] == "answer" and "messages" in data:
            payload = {key: data[key] for key in ("model", "messages", "max_tokens", "temperature")}
            if readable:
                payload["messages"] = [{**message} for message in payload["messages"]]
                for message in payload["messages"]:
                    content = message.get("content")
                    if isinstance(content, str):
                        try:
                            decoded = json.loads(content)
                        except ValueError:
                            continue
                        if isinstance(decoded, (dict, list)):
                            message["content"] = decoded
            return payload
    return None


def print_report(report, show_answer_input=False):
    result = report["result"]
    columns = shutil.get_terminal_size((120, 24)).columns
    summary = [("mode", "plan only; no device actions"),
               ("context", f"{report['context']['entities']} entities; {report['context']['actions']} actions")]
    for field in ("route", "service", "domain"):
        if field in result:
            summary.append((field, result[field] if result[field] is not None else "-"))
    if "action_metadata" in result:
        summary.append(("speaker area", result.get("speaker_area") or "unknown"))
        scopes = {"entities": "Specific arealess entities",
                  "entities_implied_area": "Specific entities + implied area",
                  "entities_explicit_area": "Specific entities + explicit area",
                  "areas": "Explicit area(s); no specific entities",
                  "implied_area": "Implied area; no specific entities"}
        summary.append(("target scope", scopes.get(result.get("target_scope"), "Unsure")))
        if "selected_areas" in result:
            names = [area["name"] for area in result["selected_areas"]]
            summary.append(("area filter", (", ".join(names[:5]) or "none") +
                            (f" (+{len(names) - 5} more)" if len(names) > 5 else "")))
            if result["area_source"] == "speaker_context":
                summary.append(("area source", "HA speaker context"))
        if "action_target_filter" in result:
            counts = result["action_target_filter"]
            summary.append(("candidates", f"{counts['entities_after_domain']} in domain → {counts['entities_after_area']} after area filter"))
    if result.get("target"):
        target = result["target"]
        names = target.get("names", [])
        name = target.get("name") or ", ".join(names[:5]) + (f" (+{len(names) - 5} more)" if len(names) > 5 else "")
        identifier = target.get("entity_id") or target.get("area_id")
        summary.append(("target", f"{target['type']}; {name}" + (f" ({identifier})" if identifier else "")))
    if result.get("execution"):
        execution = result["execution"]
        summary.append(("execution", execution["status"]))
        summary.append(("arguments", json.dumps(execution.get("arguments", {}), ensure_ascii=False)))
    print_table(["Result", "Value"], summary, [14, max(24, columns - 16)])
    print()
    if "parameter_selection" in result:
        selection = result["parameter_selection"]
        selected = selection["selected_parameters"]
        if "parameter_capabilities" in result:
            capabilities = result["parameter_capabilities"]
            removed = list(capabilities["removed_fields"])
            detail = ", ".join(removed[:5]) + (f" (+{len(removed) - 5} more)" if len(removed) > 5 else "")
            print(f"Capability filter: {capabilities['fields_before']} → {capabilities['fields_after']} fields" +
                  (f"; removed: {detail}" if removed else ""))
        print("Required parameters: " + (", ".join(result["required_parameters"]) or "none"))
        print("Selected optional parameters: " + (", ".join(selected) or "none"))
        print(f"Probability threshold: {selection['probability_threshold']:g} (strictly above)")
        capabilities = result.get("parameter_capabilities", {})
        support = capabilities.get("field_support", {})
        total_targets = len(capabilities.get("target_entity_ids", []))
        rows = [(name, f"{probability:.4f}",
                 f"{len(support[name]['supported_entity_ids'])}/{total_targets}" if name in support else "-",
                 "keep" if name in selected else "not a parameter")
                for name, probability in selection["probabilities"].items() if name in selected or name == "none"]
        if rows:
            print_table(["Parameter", "Probability", "Targets", "Selection"], rows, [40, 12, 9, 18], right_columns=(1, 2))
        print()
    stages = [("snapshot", str(report["timings_ms"]["snapshot"]), "Read HA context and settings", "-")]
    for event in report["trace"]:
        output = event.get("output", {})
        if "duration_ms" not in output:
            continue
        decisions = output.get("decisions", {})
        if decisions and all(info.get("choice") in ("keep", "drop") for info in decisions.values()):
            kept = sum(info.get("kept", False) for info in decisions.values())
            detail = f"keep {kept}/{len(decisions)}"
        else:
            detail = ", ".join(f"{name}={info.get('effective_choice', info.get('score', info.get('choice')))}"
                               for name, info in decisions.items()) or output.get("status", "")
        if event["stage"] == "optional_parameters" and "selected_parameters" in output:
            detail = "selected: " + (", ".join(output["selected_parameters"]) or "none")
        if event["stage"] in ("action_areas", "action_entities") and "selected_names" in output:
            names = output["selected_names"]
            detail = "selected: " + (", ".join(names[:5]) or "none") + (f" (+{len(names) - 5} more)" if len(names) > 5 else "")
        if event["stage"] == "target" and "selected_entity_ids" in output:
            detail = f"selected {output['entities_after']}/{output['entities_before']} (one Choice)"
        if event["stage"] == "parameter_capabilities" and "fields_after" in output:
            detail = f"{output['fields_before']} → {output['fields_after']} fields; no model call"
        scores = [info["confidence"] for info in decisions.values() if isinstance(info.get("confidence"), (int, float))]
        score = "-"
        if scores:
            score = f"{min(scores):.3f}" if min(scores) == max(scores) else f"{min(scores):.3f}–{max(scores):.3f}"
        stage_label = "capabilities" if event["stage"] == "parameter_capabilities" else event["stage"]
        stages.append((stage_label, str(output["duration_ms"]), detail, score))
    timings = report["timings_ms"]
    stages.extend([("engine subtotal", str(timings["engine"]), "", "-"), ("total", str(timings["total"]), "", "-")])
    print_table(["Stage", "Time (ms)", "Decision / status", "Confidence"], stages,
                [20, 9, max(24, columns - 50), 12], right_columns=(1, 3))
    print()
    for field in ("answer", "error"):
        if field in result:
            print(f"{field}: {result[field]}")
    for failure in result.get("failures", []):
        print(f"failure [{failure['stage']}]: {failure['reason']}")
    if show_answer_input:
        data = answer_input(report, readable=True)
        print("\nAnswer model input:")
        print(json.dumps(data, ensure_ascii=False, indent=2) if data is not None else "Answer model was not called.")


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
        if args.show_answer_input:
            report["answer_input"] = answer_input(report, readable=True)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(report, show_answer_input=args.show_answer_input)
    return int("error" in report["result"])


if __name__ == "__main__":
    raise SystemExit(main())
