import argparse
import copy
import io
import json
import unittest
from contextlib import asynccontextmanager, redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import benchmark
from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.settings import normalize_settings
from test_integration_engine import SNAPSHOT, ScriptedModels


def settings():
    return normalize_settings({
        "decider": {"url": "http://decider:8321/v1/systemone", "model": "decider"},
        "answer": {"url": "http://answer:8322/v1/chat/completions", "model": "answer"},
    })


class BenchmarkSettingsTests(unittest.TestCase):
    def test_environment_connections_and_specific_thresholds_override_exported_settings(self):
        args = benchmark.parser().parse_args(["--query", "test", "--confidence", "0.3", "--target-filter-confidence", "0.7"])
        exported = settings()
        exported["decider"]["target_filter_prompt"] = "Custom rules from HA"
        result = benchmark.model_settings(exported, args, {
            "DECIDER_URL": "http://localhost:8321/v1/systemone", "DECIDER_MODEL": "local-model",
            "DECIDER_API_KEY": "local-key", "ANSWER_API_KEY": "answer-key",
        })
        self.assertEqual(result["decider"]["url"], "http://localhost:8321/v1/systemone")
        self.assertEqual(result["decider"]["model"], "local-model")
        self.assertEqual(result["decider"]["api_key"], "local-key")
        self.assertEqual(result["answer"]["api_key"], "answer-key")
        self.assertEqual(result["decider"]["target_filter_prompt"], "Custom rules from HA")
        for field in benchmark.CONFIDENCE_FIELDS:
            self.assertEqual(result["decider"][field], 0.7 if field == "target_filter_confidence" else 0.3)
        self.assertEqual(exported["decider"]["target_filter_confidence"], 0.5)

    def test_legacy_openrouter_key_is_only_used_for_openrouter_decider(self):
        args = benchmark.parser().parse_args(["--query", "test"])
        local = benchmark.model_settings(settings(), args, {"OPENROUTER_API_KEY": "cloud-key"})
        self.assertEqual(local["decider"]["api_key"], "")
        exported = settings()
        exported["decider"]["url"] = "https://openrouter.ai/api/alpha/decisions"
        cloud = benchmark.model_settings(exported, args, {"OPENROUTER_API_KEY": "cloud-key"})
        self.assertEqual(cloud["decider"]["api_key"], "cloud-key")
        self.assertEqual(cloud["answer"]["api_key"], "")

    def test_invalid_confidence_values_are_rejected(self):
        for value in ("-0.1", "1.1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                benchmark.confidence(value)


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def exported(self):
        return {"entry_id": "one", "snapshot": copy.deepcopy(SNAPSHOT), "settings": settings()}

    async def test_benchmark_imports_the_same_engine_and_sends_identical_decider_requests(self):
        self.assertIs(benchmark.AgentEngine, AgentEngine)
        choices = [{"service": "query_state"}, {"area": "all_areas"},
                   {"domain_light": "keep", "domain_sensor": "drop"}, {"targets": ["light.desk"]}]
        local, addon = ScriptedModels(choices), ScriptedModels(choices)
        args = benchmark.parser().parse_args(["--query", "Is the desk lamp on?", "--language", "en"])
        session = AsyncMock()
        with (patch("benchmark.fetch_snapshot", new=AsyncMock(return_value=self.exported())),
              patch("benchmark.aiohttp.ClientSession", return_value=session),
              patch("benchmark.ModelClient", return_value=local)):
            report = await benchmark.run(args, {})
        execute = AsyncMock()
        await AgentEngine(settings(), addon, execute).process(args.query, "en", SNAPSHOT)
        self.assertEqual(local.requests, addon.requests)
        self.assertEqual(report["mode"], "plan_only")
        self.assertEqual(report["result"]["state_info"][0]["entity_id"], "light.desk")
        self.assertEqual({event["stage"] for event in report["trace"]}, {"service", "area", "domain_filter", "target", "answer"})
        self.assertIn("engine", report["timings_ms"])
        execute.assert_not_awaited()

    async def test_action_benchmark_marks_planned_arguments_and_keeps_credentials_out_of_report(self):
        args = benchmark.parser().parse_args(["--query", "Turn on the office lights"])
        models = ScriptedModels([{"service": "light.turn_on"}, {"area": "area:office"}, {"target_scope": "area"}])
        with (patch("benchmark.fetch_snapshot", new=AsyncMock(return_value=self.exported())),
              patch("benchmark.aiohttp.ClientSession", return_value=AsyncMock()),
              patch("benchmark.ModelClient", return_value=models)):
            report = await benchmark.run(args, {"DECIDER_API_KEY": "private-key"})
        self.assertEqual(report["result"]["execution"], {
            "status": "planned", "arguments": {"entity_id": ["light.desk", "light.ceiling"]},
        })
        self.assertNotIn("private-key", json.dumps(report))
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.print_report(report)
        self.assertIn("execution: planned", output.getvalue())
        self.assertIn("target_scope:", output.getvalue())

    async def test_model_failure_keeps_stage_diagnostics(self):
        args = benchmark.parser().parse_args(["--query", "test"])
        client = SimpleNamespace(decide=AsyncMock(side_effect=ModelError("Model request timed out.")))
        with (patch("benchmark.fetch_snapshot", new=AsyncMock(return_value=self.exported())),
              patch("benchmark.aiohttp.ClientSession", return_value=AsyncMock()),
              patch("benchmark.ModelClient", return_value=client)):
            report = await benchmark.run(args, {})
        self.assertEqual(report["result"]["error"], "Model request timed out.")
        self.assertTrue(any(event["stage"] == "failure" for event in report["trace"]))

    async def test_snapshot_request_uses_administrator_token_without_redirects(self):
        response = SimpleNamespace(status=200, json=AsyncMock(return_value=self.exported()))

        @asynccontextmanager
        async def request(*args, **kwargs):
            yield response

        session = SimpleNamespace(get=Mock(side_effect=request))
        exported = await benchmark.fetch_snapshot(session, {"HA_URL": "http://ha:8123/", "HA_TOKEN": "ha-secret", "GLOW_ENTRY_ID": "one"})
        self.assertEqual(exported["entry_id"], "one")
        self.assertEqual(session.get.call_args.args, ("http://ha:8123/api/glow_assistant/snapshot",))
        self.assertEqual(session.get.call_args.kwargs["headers"], {"Authorization": "Bearer ha-secret"})
        self.assertEqual(session.get.call_args.kwargs["params"], {"entry_id": "one"})
        self.assertFalse(session.get.call_args.kwargs["allow_redirects"])

    async def test_snapshot_errors_never_include_token_or_echo_server_response(self):
        for status in (302, 401, 403, 404, 500):
            @asynccontextmanager
            async def request(*args, **kwargs):
                yield SimpleNamespace(status=status)

            with self.subTest(status=status), self.assertRaises(ModelError) as error:
                await benchmark.fetch_snapshot(SimpleNamespace(get=request), {"HA_URL": "http://ha:8123", "HA_TOKEN": "private-token"})
            self.assertNotIn("private-token", str(error.exception))


if __name__ == "__main__":
    unittest.main()
