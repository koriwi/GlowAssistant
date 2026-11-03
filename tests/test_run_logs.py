import copy
import json
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from custom_components.glow_assistant.client import ModelClient, ModelError
from custom_components.glow_assistant.engine import AgentEngine
from custom_components.glow_assistant.run_logs import RunRecord, redact, successful
from custom_components.glow_assistant.settings import normalize_settings
from scripts import local as benchmark
from tests.test_integration_engine import SNAPSHOT, ScriptedModels


def settings(log_runs=False):
    return normalize_settings({"log_runs": log_runs, "decider": {"url": "http://local/v1/systemone", "model": "native"}})


def completed_result():
    return {"request": "test", "route": "smarthome", "answer": "ok", "decisions": {}, "timings_ms": {}, "usage": {}}


def session_for(data):
    @asynccontextmanager
    async def post(*args, **kwargs):
        yield SimpleNamespace(status=200, json=AsyncMock(return_value=data))
    return SimpleNamespace(post=Mock(side_effect=post))


class RunLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_logging_off_does_not_create_any_folder_or_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing"
            record = RunRecord("benchmark", "test", "de", enabled=False)
            record.settings(settings())
            report = await record.finish(completed_result(), path)
            self.assertFalse(path.exists())
            self.assertNotIn("log_file", report)
            self.assertNotIn("log_error", report)
            self.assertTrue(report["success"])

    async def test_success_and_failure_use_distinct_folders_and_unique_valid_json_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for result, expected in ((completed_result(), "successful"),
                                     ({**completed_result(), "route": "unsure"}, "unsuccessful"),
                                     ({**completed_result(), "route": "unsupported", "error": "not yet supported"}, "unsuccessful"),
                                     ({**completed_result(), "answer_error": "timeout"}, "unsuccessful")):
                record = RunRecord("home_assistant", "test", "de", enabled=True)
                record.settings(settings(True))
                record.snapshot(SNAPSHOT)
                record.trace("service", {"model": "native"})
                record.trace("service", {}, {"status": "completed", "duration_ms": 12})
                report = await record.finish(result, directory, entry_id="one", conversation_id="conversation")
                path = Path(report["log_file"])
                self.assertEqual(path.parent.name, expected)
                saved = json.loads(path.read_text())
                self.assertEqual(saved, report)
                self.assertEqual(saved["schema_version"], 1)
                self.assertEqual(saved["entry_id"], "one")
                self.assertEqual(saved["snapshot"]["entities"], SNAPSHOT["entities"])
                self.assertEqual(saved["settings"]["decider"]["service_prompt"], settings()["decider"]["service_prompt"])
                self.assertIn("integration_version", saved)
                if os.name == "posix":
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            files = list(Path(directory).rglob("*.json"))
            self.assertEqual(len(files), 4)
            self.assertEqual(len({path.name for path in files}), 4)
            self.assertFalse(list(Path(directory).rglob("*.tmp")))

    async def test_model_requests_and_full_responses_are_captured_without_credentials(self):
        config = settings(True)
        config["decider"]["api_key"] = "test-model-secret"
        config["general"]["mcp_servers"] = [{"name": "Tools", "url": "http://tools/mcp?token=test-mcp-secret", "token": "test-mcp-secret"}]
        response = {"answers": {"route": {"type": "choice", "choice": "ok", "confidence": 1, "probabilities": {"ok": 1}}},
                    "usage": {"input_tokens": 10}, "provider_details": {"reason": "all good"}}
        record = RunRecord("benchmark", "test", "de", enabled=True, secrets=["test-ha-secret"])
        record.settings(config)
        record.trace("service", {"model": "native"})
        client = ModelClient(session_for(response), observer=record.model_call)
        state = {"prompt": "test", "history": [{"role": "user", "content": "before"}]}
        questions = {"route": {"type": "choice", "instructions": "Exact prompt", "criteria": {"ok": "Okay"}}}
        await client.decide(config["decider"], state, questions)
        state["history"].append({"role": "assistant", "content": "later"})
        questions["route"]["instructions"] = "changed"
        record.data["exception"] = {"message": "Bearer test-ha-secret; test-model-secret; test-mcp-secret"}
        with tempfile.TemporaryDirectory() as directory:
            report = await record.finish(completed_result(), directory)
            text = Path(report["log_file"]).read_text()
        for secret in ("test-model-secret", "test-mcp-secret", "test-ha-secret"):
            self.assertNotIn(secret, text)
        call = report["model_calls"][0]
        self.assertEqual(call["stage"], "service")
        self.assertEqual(call["request"]["questions"]["route"]["instructions"], "Exact prompt")
        self.assertEqual(call["request"]["state"]["history"], [{"role": "user", "content": "before"}])
        self.assertEqual(call["response"], response)
        self.assertEqual(call["status"], "response_received")
        self.assertNotIn("api_key", call["settings"])
        self.assertNotIn("headers", call)

    async def test_json_provider_errors_are_preserved_in_logs_without_echoing_credentials(self):
        config = settings(True)
        config["decider"]["api_key"] = "test-error-secret"
        record = RunRecord("benchmark", "test", "de", enabled=True)
        record.settings(config)
        body = {"error": {"message": "Rejected test-error-secret", "code": "invalid_request"}}

        @asynccontextmanager
        async def post(*args, **kwargs):
            yield SimpleNamespace(status=400, json=AsyncMock(return_value=body))

        client = ModelClient(SimpleNamespace(post=post), observer=record.model_call)
        with self.assertRaisesRegex(ModelError, "HTTP 400"):
            await client.decide(config["decider"], {"prompt": "test"}, {})
        with tempfile.TemporaryDirectory() as directory:
            report = await record.finish({"request": "test", "error": "HTTP 400"}, directory)
        call = report["model_calls"][0]
        self.assertEqual(call["http_status"], 400)
        self.assertEqual(call["response"]["error"]["code"], "invalid_request")
        self.assertNotIn("test-error-secret", json.dumps(report))
        self.assertFalse(report["success"])

    async def test_model_failure_preserves_request_error_details_and_partial_engine_result(self):
        config = settings(True)
        config["decider"]["max_input_chars"] = 100
        record = RunRecord("benchmark", "test", "de", enabled=True)
        record.settings(config)
        session = Mock()
        engine = AgentEngine(config, ModelClient(session, observer=record.model_call), trace=record.trace, dry_run=True)
        with self.assertRaises(ModelError):
            await engine.process("test", "de", SNAPSHOT)
        with tempfile.TemporaryDirectory() as directory:
            report = await record.finish(engine.result, directory)
            saved = json.loads(Path(report["log_file"]).read_text())
        self.assertFalse(saved["success"])
        self.assertIn("service", saved["result"]["timings_ms"])
        self.assertEqual(saved["model_calls"][0]["status"], "error")
        self.assertEqual(saved["model_calls"][0]["details"]["max_input_chars"], 100)
        self.assertIn("questions", saved["model_calls"][0]["request"])
        session.post.assert_not_called()

    async def test_log_write_failure_does_not_change_the_result_or_retry_a_completed_action(self):
        execute = AsyncMock(return_value=None)
        record = RunRecord("home_assistant", "test", "de", enabled=True)
        config = settings(True)
        record.settings(config)
        client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}])
        result = await AgentEngine(config, client, execute=execute, trace=record.trace).process("Stop", "en", SNAPSHOT, speaker_area="Office")
        with patch("custom_components.glow_assistant.run_logs._write", side_effect=OSError("disk full")):
            report = await record.finish(result, "/unused")
        self.assertEqual(result["execution"]["status"], "completed")
        self.assertEqual(report["result"]["answer"], "Okay")
        self.assertTrue(report["success"])
        self.assertEqual(report["log_error"], "disk full")
        execute.assert_awaited_once()
        self.assertNotIn("log_file", report)

    async def test_record_preparation_failure_does_not_replace_a_successful_result(self):
        record = RunRecord("home_assistant", "test", "de", enabled=True)
        record.settings(settings(True))
        result = completed_result()
        with patch("custom_components.glow_assistant.run_logs.redact", side_effect=ValueError("bad record")):
            report = await record.finish(result, "/unused")
        self.assertIs(report["result"], result)
        self.assertTrue(report["success"])
        self.assertIn("Could not prepare", report["log_error"])

    async def test_snapshot_failure_and_unsure_runs_are_saved_by_benchmark_when_enabled(self):
        args = benchmark.parser().parse_args(["-q", "test", "--log-runs"])
        with tempfile.TemporaryDirectory() as directory:
            with (patch("scripts.local.LOG_DIRECTORY", Path(directory)),
                  patch("scripts.local.fetch_snapshot", new=AsyncMock(side_effect=ModelError("Snapshot access denied."))),
                  patch("scripts.local.aiohttp.ClientSession", return_value=AsyncMock())):
                failed = await benchmark.run(args, {"HA_TOKEN": "test-ha-secret"})
            self.assertFalse(failed["success"])
            self.assertEqual(Path(failed["log_file"]).parent.name, "unsuccessful")
            self.assertIn("Snapshot access denied", failed["result"]["error"])
            self.assertIn("traceback", failed["exception"])
            client = ScriptedModels([{"service": "light.turn_off"}], confidence=0.1)
            with (patch("scripts.local.LOG_DIRECTORY", Path(directory)),
                  patch("scripts.local.fetch_snapshot", new=AsyncMock(return_value={"entry_id": "one", "settings": settings(), "snapshot": SNAPSHOT})),
                  patch("scripts.local.aiohttp.ClientSession", return_value=AsyncMock()), patch("scripts.local.ModelClient", return_value=client)):
                unsure = await benchmark.run(args, {})
            self.assertEqual(unsure["result"]["route"], "unsure")
            self.assertFalse(unsure["success"])
            self.assertEqual(len(list((Path(directory) / "unsuccessful").glob("*.json"))), 2)

    async def test_dry_run_plan_is_logged_as_success_and_ha_logging_does_not_implicitly_enable_benchmark_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for enabled in (False, True):
                client = ScriptedModels([{"service": "light.turn_off"}, {"target_scope": "implied_area"}])
                args = benchmark.parser().parse_args(["-q", "Stop", "--speaker-area", "Office", *(["--log-runs"] if enabled else [])])
                exported = {"entry_id": "one", "settings": settings(True), "snapshot": copy.deepcopy(SNAPSHOT)}
                with (patch("scripts.local.LOG_DIRECTORY", Path(directory)),
                      patch("scripts.local.fetch_snapshot", new=AsyncMock(return_value=exported)),
                      patch("scripts.local.aiohttp.ClientSession", return_value=AsyncMock()), patch("scripts.local.ModelClient", return_value=client)):
                    report = await benchmark.run(args, {})
                self.assertTrue(report["success"])
                self.assertEqual(report["result"]["execution"]["status"], "planned")
                self.assertEqual("log_file" in report, enabled)
            self.assertEqual(len(list(Path(directory).rglob("*.json"))), 1)

    async def test_executed_local_run_logs_live_mode_and_returned_action_data(self):
        from tests.test_actions import ActionExecutionTests

        fixture = ActionExecutionTests()
        args = benchmark.parser().parse_args(["-q", "Read the shopping list", "--execute", "--log-runs"])
        response = {"todo.shopping": {"items": [{"summary": "Milk", "status": "needs_action"}]}}
        exported = {"entry_id": "one", "settings": settings(), "snapshot": fixture.todo_snapshot()}
        with tempfile.TemporaryDirectory() as directory:
            with (patch("scripts.local.LOG_DIRECTORY", Path(directory)),
                  patch("scripts.local.fetch_snapshot", new=AsyncMock(return_value=exported)),
                  patch("scripts.local.aiohttp.ClientSession", return_value=AsyncMock()),
                  patch("scripts.local.ModelClient", return_value=ScriptedModels(fixture.todo_choices())),
                  patch("scripts.local.execute_action", new=AsyncMock(return_value=response)) as execute):
                report = await benchmark.run(args, {})
            saved = json.loads(Path(report["log_file"]).read_text())
            self.assertEqual(Path(report["log_file"]).parent.name, "successful")
            self.assertEqual(saved["mode"], "live")
            self.assertEqual(saved["source"], "local")
            self.assertTrue(saved["local_flags"]["execute"])
            self.assertEqual(saved["result"]["execution"]["calls"][0]["result"], response)
            execute.assert_awaited_once()


class RunOutcomeTests(unittest.TestCase):
    def test_success_means_complete_decisions_and_no_operation_or_answer_failure(self):
        self.assertTrue(successful(completed_result()))
        for status, dry_run, expected in (("planned", True, True), ("planned", False, False), ("completed", False, True),
                                          ("error", False, False), ("partial", False, False)):
            result = {**completed_result(), "execution": {"status": status, "dry_run": dry_run}}
            self.assertEqual(successful(result), expected)
        for patch_values in ({"route": "unsure"}, {"route": "unsupported"}, {"error": "bad"}, {"error": ""}, {"answer_error": "bad"}, {"answer_error": ""},
                             {"failures": [{"stage": "answer"}]}):
            self.assertFalse(successful({**completed_result(), **patch_values}))

    def test_redaction_handles_nested_headers_url_credentials_and_nonfinite_error_values(self):
        data = {"api_key": "key", "token": "token", "headers": {"Authorization": "Bearer other"},
                "nested": [{"url": "http://user:password@host/v1?api_key=query-secret", "error": "Bearer secret"}],
                "invalid_probability": float("nan")}
        safe = redact(data)
        text = json.dumps(safe, allow_nan=False)
        self.assertNotIn("password", text)
        self.assertNotIn("query-secret", text)
        self.assertNotIn("Bearer other", text)
        self.assertNotIn("Bearer secret", text)
        self.assertEqual(safe["invalid_probability"], "nan")
