import argparse
import asyncio
import json
import os
import shutil
import sys
import textwrap
import traceback
from itertools import zip_longest
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import aiohttp
from dotenv import load_dotenv

from custom_components.glow_assistant.actions import ActionError
from custom_components.glow_assistant.client import ModelClient, ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.run_logs import RunRecord
from custom_components.glow_assistant.settings import (
    DEFAULTS,
    normalize_settings,
    validate_settings,
)

LOG_DIRECTORY = Path(__file__).parent / "runs"

CONFIDENCE_FIELDS = tuple(key for key in DEFAULTS["decider"] if key.endswith("_confidence"))


def confidence(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("Confidence must be between 0 and 1.")
    return value


def parser():
    result = argparse.ArgumentParser(description="Test and debug the shared HA engine locally; dry-run unless --execute is set.")
    result.add_argument("--query", "-q", required=True)
    result.add_argument("--execute", action="store_true", help="Execute selected HA actions and allow general-model MCP calls; default is dry-run.")
    result.add_argument("--log-runs", action="store_true", help="Save each run to runs/successful or runs/unsuccessful; disabled by default.")
    result.add_argument("--llm-answer", action="store_true", help="Generate smart-home replies with the answer LLM; disabled by default.")
    result.add_argument("--confidence", type=confidence, help="Override every decider confidence threshold.")
    result.add_argument("--parameter-threshold", type=confidence,
                        help="Relative probability threshold for optional parameters; independent of --confidence.")
    result.add_argument("--action-area-threshold", type=confidence, help="Relative probability threshold for multiple action areas.")
    result.add_argument("--action-entity-threshold", type=confidence, help="Relative probability threshold for action entities.")
    result.add_argument("--query-target-threshold", type=confidence,
                        help="Relative probability threshold for state-query entities; independent of --confidence.")
    result.add_argument("--text-threshold", type=confidence, help="Relative probability threshold for item/message words.")
    result.add_argument("--select-threshold", type=confidence, help="Relative probability threshold for multi-value select extraction; default 0.3.")
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


def model_settings(exported, args, env, record=None):
    settings = normalize_settings(exported)
    for group, prefix in (("decider", "DECIDER"), ("answer", "ANSWER"), ("general", "GENERAL")):
        for field in ("url", "model", "api_key"):
            value = env.get(f"{prefix}_{field.upper()}", "").strip()
            if value:
                settings[group][field] = value
    settings["answer"]["enable"] = args.llm_answer
    settings["log_runs"] = args.log_runs
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
    if args.text_threshold is not None:
        decider["text_probability_threshold"] = args.text_threshold
    if args.select_threshold is not None:
        decider["select_probability_threshold"] = args.select_threshold
    for kind in ("area", "entity"):
        value = getattr(args, f"action_{kind}_threshold")
        if value is not None:
            decider[f"action_{kind}_probability_threshold"] = value
    if record is not None:
        record.settings(settings)
    return validate_settings(settings)


def ha_connection(env):
    url = env.get("HA_URL", "").strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Set HA_URL to your Home Assistant base URL, without credentials or query parameters.")
    token = env.get("HA_TOKEN", "").strip()
    if not token:
        raise ValueError("Set HA_TOKEN to a Home Assistant administrator token in .env.")
    return url, token


async def fetch_snapshot(session, env):
    url, token = ha_connection(env)
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


async def execute_action(session, env, entry_id, action, arguments):
    url, token = ha_connection(env)
    try:
        async with session.post(
            url + "/api/glow_assistant/execute", headers={"Authorization": f"Bearer {token}"},
            params={"entry_id": entry_id}, json={"service": action, "arguments": arguments},
            timeout=aiohttp.ClientTimeout(total=60), allow_redirects=False,
        ) as response:
            if response.status in (401, 403):
                raise ActionError("Action access denied; check HA_TOKEN and administrator access.")
            if response.status == 404:
                raise ActionError("Action endpoint or entry unavailable; update/restart GlowAssistant and check GLOW_ENTRY_ID.")
            try:
                data = await response.json()
            except (ValueError, aiohttp.ContentTypeError) as error:
                raise ActionError("Action endpoint returned invalid data; outcome is unconfirmed.", command_sent=True) from error
            if response.status != 200:
                sent = data.get("command_sent", True) if isinstance(data, dict) else True
                message = data.get("error") if isinstance(data, dict) else None
                raise ActionError(message if isinstance(message, str) else f"Action endpoint returned HTTP {response.status}.",
                                  command_sent=sent is not False)
            if not isinstance(data, dict) or "result" not in data:
                raise ActionError("Action endpoint returned invalid data; outcome is unconfirmed.", command_sent=True)
            return data["result"]
    except (TimeoutError, aiohttp.ClientError) as error:
        raise ActionError("Home Assistant action request failed; outcome is unconfirmed. The action was not retried.",
                          command_sent=True) from error


async def run(args, env):
    started = perf_counter()
    language = args.language or env.get("BENCHMARK_LANGUAGE", "").strip() or "de"
    record = RunRecord("local", args.query, language, dry_run=not args.execute, enabled=args.log_runs,
                       capture_models=args.json, secrets=[env.get("HA_TOKEN", "")])
    record.data.update(speaker_area=args.speaker_area, local_flags=vars(args))
    engine = None
    result = None
    snapshot_ms = 0
    engine_ms = 0
    try:
        async with aiohttp.ClientSession() as session:
            exported = await fetch_snapshot(session, env)
            snapshot_ms = round((perf_counter() - started) * 1000)
            record.data["entry_id"] = exported["entry_id"]
            record.snapshot(exported["snapshot"])
            settings = model_settings(exported["settings"], args, env, record=record)
            client = ModelClient(session, observer=record.model_call if record.capture_models else None)

            async def execute(action, arguments):
                return await execute_action(session, env, exported["entry_id"], action, arguments)

            engine = AgentEngine(settings, client, trace=record.trace, dry_run=not args.execute,
                                 execute=execute if args.execute else None)
            record.data["history"] = engine.history
            engine_started = perf_counter()
            try:
                result = await engine.process(args.query, language, exported["snapshot"], speaker_area=args.speaker_area)
            finally:
                engine_ms = round((perf_counter() - engine_started) * 1000)
    except Exception as error:
        result = {**getattr(engine, "result", {}), "request": args.query, "error": str(error),
                  "details": getattr(error, "details", {})}
        record.data["exception"] = {"type": type(error).__name__, "traceback": traceback.format_exc()}
        if engine is None:
            snapshot_ms = round((perf_counter() - started) * 1000)
    finally:
        if result is None:
            result = {**getattr(engine, "result", {}), "request": args.query, "error": "The run was interrupted."}
        report = await record.finish(result, LOG_DIRECTORY, timings_ms={
            "snapshot": snapshot_ms, "engine": engine_ms, "total": round((perf_counter() - started) * 1000),
        })
    return report


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
    summary = [("mode", "live; actions enabled" if report.get("mode") == "live" else "plan only; no device actions"),
               ("context", f"{report['context']['entities']} entities; {report['context']['actions']} actions")]
    for field in ("route", "service", "domain"):
        if field in result:
            summary.append((field, result[field] if result[field] is not None else "-"))
    if "execution" in result:
        execution = result["execution"]
        summary.append(("execution", f"{execution['status']}; {len(execution['calls'])} call(s)"))
    if "answer_source" in result:
        summary.append(("answer mode", result["answer_source"]))
    if "state_property" in result:
        summary.append(("state property", result["state_property"]))
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
                  (f"; unsupported for selected targets: {detail}" if removed else ""))
        print("Required parameters: " + (", ".join(result["required_parameters"]) or "none"))
        print("Selected optional parameters: " + (", ".join(selected) or "none"))
        threshold = selection["probability_threshold"]
        cutoff = selection["effective_probability_threshold"]
        option_count = selection["real_option_count"]
        print(f"Probability cutoff: {threshold:g} / max(1, {option_count}) = {cutoff:.6g}; keep p >= cutoff")
        capabilities = result.get("parameter_capabilities", {})
        support = capabilities.get("field_support", {})
        total_targets = len(capabilities.get("target_entity_ids", []))
        rows = [(name, f"{probability:.4f}",
                 f"{len(support[name]['supported_entity_ids'])}/{total_targets}" if name in support else "-",
                 "not a parameter" if name == "none" else "keep" if name in selected
                 else f"drop: p < {cutoff:.6g}")
                for name, probability in selection["probabilities"].items()]
        if rows:
            print_table(["Parameter", "Probability", "Targets", "Decision / reason"], rows, [40, 12, 9, 28], right_columns=(1, 2))
        print()
    if result.get("parameters"):
        print_table(["Parameter", "Value"], [(name, json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value)
                                             for name, value in result["parameters"].items()], [32, max(30, columns - 36)])
        print()
    stages = [("snapshot", str(report["timings_ms"]["snapshot"]), "Read HA context and settings", "-")]
    for event in report["trace"]:
        output = event.get("output", {})
        if "duration_ms" not in output:
            continue
        decisions = output.get("decisions", {})
        detail = ", ".join(f"{name}={info.get('effective_choice', info.get('choice'))}"
                           for name, info in decisions.items()) or output.get("status", "")
        if event["stage"].split(":", 1)[0] in ("text", "numeric", "select") and "value" in output:
            detail = (f"{output['parameter']}: skipped (no text)" if output.get("status") == "skipped"
                      else f"{output['parameter']}={output['value']}")
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
        if field in result and not (field == "answer" and result.get("error") == result[field]):
            print(f"{field}: {result[field]}")
    if report.get("log_file"):
        print(f"run log: {report['log_file']}")
    if report.get("log_error"):
        print(f"run log error: {report['log_error']}")
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
        print(f"local error: {error}", file=sys.stderr)
        return 1
    if args.json:
        if args.show_answer_input:
            report["answer_input"] = answer_input(report, readable=True)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(report, show_answer_input=args.show_answer_input)
    return int(not report.get("success", "error" not in report["result"]))


if __name__ == "__main__":
    raise SystemExit(main())
