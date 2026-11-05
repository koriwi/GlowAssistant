import argparse
import asyncio
import copy
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4

import aiohttp
from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_components.glow_assistant.client import ModelClient
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.optional_parameters import action_fields
from custom_components.glow_assistant.run_logs import redact
from custom_components.glow_assistant.validation import finite_number
from scripts import ROOT
from scripts.local import confidence, model_settings, print_table
from scripts.local import parser as local_parser

DATA = ROOT / "tests" / "decisions"
SET_FIELDS = {"entity_ids", "area_ids", "parameters", "state_properties"}
CHECK_FIELDS = SET_FIELDS | {"route", "service", "service_choice", "domain", "target_scope", "target_type", "query_type", "state_answer", "select_values"}


def levenshtein(left, right):
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, 1):
        current = [index]
        for other, value in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[other] + 1, previous[other - 1] + (char != value)))
        previous = current
    return previous[-1]


def value_score(expected, actual, scale=1):
    if isinstance(expected, str):
        left, right = (" ".join(value.casefold().split()) for value in (expected, actual if isinstance(actual, str) else ""))
        distance = levenshtein(left, right)
        return {"kind": "text", "score": 1 - distance / max(1, len(left), len(right)), "distance": distance}
    if not finite_number(expected) or not finite_number(scale) or scale <= 0:
        raise ValueError("Numeric scores need a finite expected value and a positive scale.")
    error = abs(expected - actual) if finite_number(actual) else None
    return {"kind": "number", "score": max(0, 1 - error / scale) if error is not None else 0,
            "absolute_error": error, "scale": scale}


def load_cases(path, snapshot):
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    ids = set()
    entity_ids = {entity["entity_id"] for entity in snapshot["entities"]}
    area_ids = {entity["area_id"] for entity in snapshot["entities"] if entity.get("area_id")}
    for case in cases:
        if not case.get("id") or case["id"] in ids or not case.get("query", "").strip() or not case.get("category"):
            raise ValueError("Each case needs a unique id, category and non-empty query.")
        ids.add(case["id"])
        expectations = case["expect"] if isinstance(case["expect"], list) else [case["expect"]]
        if not expectations:
            raise ValueError(f"{case['id']}: no expected decisions.")
        for expected in expectations:
            unknown = set(expected) - CHECK_FIELDS - {"score_values"}
            if unknown or not set(expected) & CHECK_FIELDS:
                raise ValueError(f"{case['id']}: invalid expected fields: {sorted(unknown)}")
            if not set(expected.get("entity_ids", [])) <= entity_ids or not set(expected.get("area_ids", [])) <= area_ids:
                raise ValueError(f"{case['id']}: expected entity or area missing from snapshot.")
            service = expected.get("service")
            if service and service != "query_state" and service not in snapshot["services"]:
                raise ValueError(f"{case['id']}: expected action missing from snapshot.")
            for spec in expected.get("score_values", {}).values():
                value_score(spec["value"], None, spec.get("scale", 1)) if isinstance(spec, dict) else value_score(spec, None)
    return cases


def actual_decisions(result, snapshot):
    target = result.get("target") or {}
    ids = target.get("entity_ids", [target["entity_id"]] if target.get("entity_id") else [])
    areas = target.get("area_ids", [target["area_id"]] if target.get("area_id") else [])
    if target.get("type") in ("area", "areas"):
        ids = [entity["entity_id"] for entity in snapshot["entities"]
               if entity.get("area_id") in areas and entity["domain"] == result["domain"]]
    parameters = sorted(set(result.get("required_parameters", [])) | set(result.get("parameter_selection", {}).get("selected_parameters", [])))
    failures = result.get("failures", [])
    # Extraction quality is scored separately, including empty or uncertain values.
    extraction_stop = bool(failures) and all(failure["stage"].split(":", 1)[0] in ("text", "numeric") for failure in failures)
    return {
        "route": "smarthome" if extraction_stop else result.get("route"),
        "service": result.get("service"), "domain": result.get("domain"),
        "service_choice": result.get("decisions", {}).get("service", {}).get("service", {}).get("choice"),
        "target_scope": result.get("target_scope"), "target_type": target.get("type"),
        "entity_ids": sorted(ids), "area_ids": sorted(areas or [area["area_id"] for area in result.get("selected_areas", [])]),
        "parameters": parameters, "query_type": result.get("query_type"), "state_answer": result.get("state_answer"),
        "state_properties": result.get("state_properties", []),
        "select_values": {name: value["value"] for name, value in result.get("parameter_extractions", {}).items() if value["type"] == "select"},
        "extraction_only_stop": extraction_stop,
    }


def evaluate(case, result, snapshot, error=None):
    actual = actual_decisions(result, snapshot)
    expectations = case["expect"] if isinstance(case["expect"], list) else [case["expect"]]
    comparisons = []
    for expected in expectations:
        checks = {}
        for name, value in expected.items():
            if name == "score_values":
                continue
            found = actual[name]
            if name in SET_FIELDS:
                matches = set(value) == set(found)
            elif name == "select_values":
                matches = set(value) == set(found) and all(
                    set(item) == set(found[key]) if isinstance(item, list) and isinstance(found[key], list) else item == found[key]
                    for key, item in value.items())
            else:
                matches = value == found
            checks[name] = {"passed": matches, "expected": value, "actual": found}
        comparisons.append((expected, checks))
    expected, checks = min(comparisons, key=lambda item: sum(not check["passed"] for check in item[1].values()))
    scores = {}
    fields = action_fields(snapshot["services"].get(result.get("service"), {}).get("fields", {}))
    for name, spec in expected.get("score_values", {}).items():
        extraction = result.get("parameter_extractions", {}).get(name, {})
        value = extraction.get("value", result.get("parameters", {}).get(name))
        selector = fields.get(name, {}).get("selector", {}).get("number", {})
        scale = selector.get("max", 1) - selector.get("min", 0)
        wanted = spec.get("value") if isinstance(spec, dict) else spec
        if isinstance(spec, dict):
            scale = spec.get("scale", scale)
        scores[name] = {"expected": wanted, "actual": value, **value_score(wanted, value, scale)}
    passed = error is None and all(check["passed"] for check in checks.values())
    return {"passed": passed, "checks": checks, "actual": actual, "scores": scores,
            "error": error, "expected": expected}


async def run_case(case, snapshot, settings, session):
    config = copy.deepcopy(settings)
    config["history_enabled"] = bool(case.get("history"))
    trace, calls = [], []

    def observe(stage, arguments, output=None):
        trace.append(copy.deepcopy({"stage": stage, "input" if output is None else "output": arguments if output is None else output}))

    client = ModelClient(session, observer=calls.append)
    engine = AgentEngine(config, client, trace=observe, history=case.get("history"), decisions_only=True)
    started = perf_counter()
    error = None
    try:
        result = await engine.process(case["query"], case.get("language", "de"), snapshot, speaker_area=case.get("speaker_area"))
    except Exception as failure:
        result = engine.result
        error = f"{type(failure).__name__}: {failure}"
    return {"id": case["id"], "category": case["category"], "query": case["query"], "speaker_area": case.get("speaker_area"),
            **evaluate(case, result, snapshot, error), "duration_ms": round((perf_counter() - started) * 1000),
            "result": result, "trace": trace, "model_calls": calls}


def summarize(results):
    scores = [score for result in results for score in result["scores"].values()]
    return {"cases": len(results), "passed": sum(result["passed"] for result in results),
            "failed": sum(not result["passed"] for result in results), "errors": sum(result["error"] is not None for result in results),
            "text_score": mean(score["score"] for score in scores if score["kind"] == "text") if any(score["kind"] == "text" for score in scores) else None,
            "number_score": mean(score["score"] for score in scores if score["kind"] == "number") if any(score["kind"] == "number" for score in scores) else None,
            "mean_ms": mean(result["duration_ms"] for result in results)}


def parser():
    result = argparse.ArgumentParser(description="Run real decider tests against a fixed HA fixture. No HA, chat-model or device calls.")
    result.add_argument("--cases", type=Path, default=DATA / "cases.jsonl")
    result.add_argument("--snapshot", type=Path, default=DATA / "snapshot.json")
    result.add_argument("--settings", type=Path, help="Optional settings/export/run JSON; otherwise use current code defaults.")
    result.add_argument("--category", action="append", help="Run only this category; repeat for several categories.")
    result.add_argument("--case", action="append", dest="case_ids", help="Run only this case ID; repeat for several cases.")
    result.add_argument("--limit", type=int, help="Maximum number of cases to run.")
    result.add_argument("--confidence", type=confidence, help="Override all confidence gates; relative thresholds stay unchanged.")
    result.add_argument("--query-target-threshold", type=confidence, help="Override the relative threshold for state-query entity selection.")
    result.add_argument("--list", action="store_true", help="List cases without making any API calls.")
    result.add_argument("--output", type=Path, help="JSON report path; default benchmarks/decisions/TIMESTAMP-ID.json.")
    return result


def percent(value):
    return "-" if value is None else f"{value * 100:.1f}"


async def run(cases, snapshot, settings):
    results = []
    width = max(20, shutil.get_terminal_size((140, 24)).columns - 76)
    print(f"model: {settings['decider']['model']}; {len(cases)} cases; decisions only", flush=True)
    print(f"{'Result':<6}  {'Case':<28}  {'Checks':>6}  {'Text %':>6}  {'Num %':>6}  {'ms':>6}  Query", flush=True)
    async with aiohttp.ClientSession() as session:
        for case in cases:
            result = await run_case(case, snapshot, settings, session)
            results.append(result)
            summary = summarize([result])
            checks = f"{sum(check['passed'] for check in result['checks'].values())}/{len(result['checks'])}"
            status = "ERROR" if result["error"] else "PASS" if result["passed"] else "FAIL"
            print(f"{status:<6}  {case['id']:<28}  {checks:>6}  {percent(summary['text_score']):>6}  "
                  f"{percent(summary['number_score']):>6}  {result['duration_ms']:>6}  {case['query'][:width]}", flush=True)
            for name, check in result["checks"].items():
                if not check["passed"]:
                    print(f"  {name}: expected {check['expected']!r}; got {check['actual']!r}", flush=True)
            if result["error"]:
                print(f"  {result['error']}", flush=True)
            for name, score in result["scores"].items():
                print(f"  {name}: {score['actual']!r}; expected {score['expected']!r}; score {percent(score['score'])}%", flush=True)
    return results


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive.")
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        cases = load_cases(args.cases, snapshot)
        if args.category:
            cases = [case for case in cases if case["category"] in args.category]
        if args.case_ids:
            unknown = set(args.case_ids) - {case["id"] for case in cases}
            if unknown:
                raise ValueError(f"Unknown or excluded case IDs: {sorted(unknown)}")
            cases = [case for case in cases if case["id"] in args.case_ids]
        if args.limit:
            cases = cases[:args.limit]
        if not cases:
            raise ValueError("No cases match the selected filters.")
        if args.list:
            print_table(["Case", "Category", "Speaker area", "Query"],
                        [(case["id"], case["category"], case.get("speaker_area") or "-", case["query"]) for case in cases], [32, 20, 15, 75])
            return 0
        if args.output and args.output.exists():
            raise ValueError(f"Report already exists: {args.output}")
        load_dotenv(ROOT / ".env")
        saved = json.loads(args.settings.read_text(encoding="utf-8")) if args.settings else {}
        exported = copy.deepcopy(saved.get("settings", saved))
        exported.setdefault("general", {})["enable"] = False
        config_args = local_parser().parse_args(["-q", "decision tests"])
        config_args.confidence = args.confidence
        config_args.query_target_threshold = args.query_target_threshold
        settings = model_settings(exported, config_args, os.environ)
        started = datetime.now(UTC)
        results = asyncio.run(run(cases, snapshot, settings))
        summary = summarize(results)
        report = {"schema_version": 1, "started_at": started.isoformat(), "finished_at": datetime.now(UTC).isoformat(),
                  "integration_version": json.loads((ROOT / "custom_components/glow_assistant/manifest.json").read_text())["version"],
                  "model": settings["decider"]["model"], "settings": settings,
                  "snapshot": snapshot, "cases": cases, "summary": summary, "results": results,
                  "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                  "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest()}
        output = args.output or ROOT / "benchmarks" / "decisions" / f"{started.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex[:8]}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(redact(report, [settings["decider"]["api_key"]]), stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
        print(f"\nDecisions: {summary['passed']}/{summary['cases']} passed; {summary['errors']} errors.")
        print(f"Extraction: text {percent(summary['text_score'])}%; numbers {percent(summary['number_score'])}%; mean {summary['mean_ms']:.0f} ms/case.")
        print(f"report: {output}")
        return int(summary["failed"] > 0)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"decision tests error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
