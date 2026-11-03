import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from scripts.area_filter_probe import CASES, SOURCE_AREA, matrix_cases, probe, run


def answer(payload, choice, confidence=0.9):
    stage, question = next(iter(payload["questions"].items()))
    return {"answers": {stage: {
        "type": "choice", "choice": choice, "confidence": confidence,
        "probabilities": {key: int(key == choice) for key in question["criteria"]},
    }}}


class AreaFilterProbeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output_dir = Path(directory.name)

    def test_each_query_runs_in_both_languages_with_and_without_location(self):
        cases = list(matrix_cases())
        self.assertEqual(len(cases), len(CASES) * 4)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        for index in range(0, len(cases), 4):
            self.assertEqual(
                {(case["language"], case["user_location_known"]) for case in cases[index:index + 4]},
                {("DE", False), ("DE", True), ("EN", False), ("EN", True)},
            )

    async def classify(self, selected, score, expected=False, confidence=0.5):
        case = {
            "prompt": "Are any windows open?", "language": "EN", "user_location_known": True,
            "source_area": SOURCE_AREA, "expected": expected,
        }
        decider = AsyncMock(side_effect=lambda payload, key: answer(payload, selected, score))
        with patch("scripts.area_filter_probe.call_decider", decider):
            result = await probe(case, confidence)
        decider.assert_awaited_once()
        return result, decider

    async def test_prompt_language_and_source_area_match_the_future_integration_state(self):
        result, decider = await self.classify("filter_by_area", 0.9, expected=True)
        self.assertTrue(result["passed"])
        self.assertEqual(decider.await_args.args[0]["state"], {
            "prompt": "Are any windows open?", "language": "en", "source_area": SOURCE_AREA,
        })

    async def test_unknown_location_is_null_and_query_is_unchanged_for_each_language(self):
        cases = list(matrix_cases())[:4]
        decider = AsyncMock(side_effect=lambda payload, key: answer(payload, "filter_by_area"))
        with patch("scripts.area_filter_probe.call_decider", decider):
            for case in cases:
                await probe(case)
        for case, call in zip(cases, decider.await_args_list, strict=True):
            state = call.args[0]["state"]
            self.assertEqual(state["prompt"], case["prompt"])
            self.assertEqual(state["language"], case["language"].lower())
            self.assertEqual(state["source_area"], SOURCE_AREA if case["user_location_known"] else None)
            self.assertEqual(set(state), {"prompt", "language", "source_area"})

    async def test_unsure_and_low_confidence_are_failures_not_no_filter(self):
        for choice, score in (("unsure", 0.9), ("skip_area", 0.49), ("filter_by_area", 0.49)):
            with self.subTest(choice=choice, score=score):
                result, _ = await self.classify(choice, score)
                self.assertIsNone(result["selected"])
                self.assertFalse(result["passed"])
        result, _ = await self.classify("skip_area", 0.5)
        self.assertIs(result["selected"], False)
        self.assertTrue(result["passed"])

    async def test_cli_continues_after_failed_call_without_retry_and_never_connects_to_ha(self):
        cases = list(matrix_cases())[:2]
        decider = AsyncMock(side_effect=[
            ValueError("Decider HTTP 503"),
            answer({"questions": {"filter": {"criteria": {"filter_by_area": "", "skip_area": "", "unsure": ""}}}},
                   "filter_by_area"),
        ])
        output = io.StringIO()
        with (
            patch("scripts.area_filter_probe.matrix_cases", return_value=iter(cases)),
            patch("scripts.area_filter_probe.call_decider", decider),
            patch("scripts.area_filter_probe.load_dotenv"),
            patch("scripts.area_filter_probe.decider_api_key", return_value=None),
            patch("scripts.local.fetch_snapshot", new=AsyncMock()) as inspector,
            patch("scripts.local.execute_action", new=AsyncMock()) as executor,
            redirect_stdout(output),
        ):
            self.assertEqual(await run(["--json", "--output-dir", str(self.output_dir)]), 1)
        self.assertEqual(decider.await_count, len(cases))
        inspector.assert_not_awaited()
        executor.assert_not_awaited()
        result = json.loads(output.getvalue())
        self.assertEqual(result["summary"]["passed"], 1)
        self.assertEqual(result["summary"]["failed"], 1)
        self.assertIn("503", result["results"][0]["error"])
        self.assertEqual(json.loads(Path(result["output_file"]).read_text()), result)

    async def test_successful_matrix_prints_each_case_and_summary(self):
        cases = list(matrix_cases())[:2]
        decider = AsyncMock(side_effect=lambda payload, key: answer(payload, "filter_by_area"))
        output = io.StringIO()
        with (
            patch("scripts.area_filter_probe.matrix_cases", return_value=iter(cases)),
            patch("scripts.area_filter_probe.call_decider", decider),
            patch("scripts.area_filter_probe.load_dotenv"),
            patch("scripts.area_filter_probe.decider_api_key", return_value=None),
            redirect_stdout(output),
        ):
            self.assertEqual(await run(["--output-dir", str(self.output_dir)]), 0)
        self.assertEqual(output.getvalue().count("PASS"), 2)
        self.assertIn("2/2 passed (100.0%)", output.getvalue())
        self.assertIn("saved:", output.getvalue())

    async def test_saved_runs_identify_prompt_model_suite_and_distinguish_threshold_failures(self):
        case = next(matrix_cases())
        instructions = "Decide whether the targets have a room restriction."
        prompt_file = self.output_dir / "prompt.txt"
        prompt_file.write_text(instructions, encoding="utf-8")
        decider = AsyncMock(side_effect=lambda payload, key: answer(payload, "filter_by_area", 0.1))
        with (
            patch("scripts.area_filter_probe.matrix_cases", side_effect=lambda: iter([case])),
            patch("scripts.area_filter_probe.call_decider", decider),
            patch("scripts.area_filter_probe.load_dotenv"),
            patch("scripts.area_filter_probe.decider_api_key", return_value="secret-key"),
            patch("scripts.area_filter_probe.decider_model", return_value="local-model"),
            patch("scripts.area_filter_probe.decider_url", return_value="http://user:password@localhost:8321/v1/systemone?key=private"),
            redirect_stdout(io.StringIO()),
        ):
            for _ in range(2):
                self.assertEqual(await run([
                    "--output-dir", str(self.output_dir), "--instructions-file", str(prompt_file),
                    "--label", "d3-mini-Q8", "--confidence", "0.25",
                ]), 1)
        files = list(self.output_dir.glob("*.json"))
        self.assertEqual(len(files), 2)
        for file in files:
            report = json.loads(file.read_text())
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["model"], "local-model")
            self.assertEqual(report["label"], "d3-mini-Q8")
            self.assertEqual(report["instructions"], instructions)
            self.assertEqual(report["instructions_sha256"], hashlib.sha256(instructions.encode()).hexdigest())
            self.assertEqual(report["endpoint"], {
                "scheme": "http", "host": "localhost", "port": 8321, "path": "/v1/systemone",
            })
            self.assertEqual(report["summary"]["raw_accuracy"], 1)
            self.assertEqual(report["summary"]["accuracy"], 0)
            self.assertEqual(report["summary"]["below_confidence"], 1)
            self.assertEqual(report["summary"]["model_unsure"], 0)
            self.assertEqual(report["summary"]["errors"], 0)
            self.assertEqual(report["results"][0]["request"]["questions"]["filter"]["instructions"], instructions)
            for secret in ("secret-key", "password", "private"):
                self.assertNotIn(secret, file.read_text())
        self.assertEqual(json.loads(files[0].read_text())["suite_sha256"],
                         json.loads(files[1].read_text())["suite_sha256"])

    async def test_malformed_response_counts_as_failed_case(self):
        with patch("scripts.area_filter_probe.call_decider", new=AsyncMock(return_value={"answers": {}})):
            result = await probe(next(matrix_cases()))
        self.assertFalse(result["passed"])
        self.assertIsNone(result["selected"])
        self.assertIn("Invalid decider", result["error"])
        self.assertEqual(result["response"], {"answers": {}})


if __name__ == "__main__":
    unittest.main()
