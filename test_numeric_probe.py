import asyncio
import copy
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

from numeric_probe import decode_probe, probe_request, run


RESPONSE = {
    "answers": {
        "numeric_score": {
            "type": "score", "score": 1.25, "confidence": 0.6,
            "probabilities": {"0": 0, "1": 0.75, "2": 0.25},
        },
        "numeric_choice": {
            "type": "choice", "choice": "level_1", "confidence": 0.6,
            "probabilities": {"level_0": 0, "level_1": 0.75, "level_2": 0.25},
        },
    },
    "usage": {"cost": 0.00001},
}


class NumericProbeTests(unittest.TestCase):
    def test_absolute_and_relative_prompts_and_current_are_passed_to_model(self):
        for prompt, current in (("set brightness to 55%", None), ("increase by 15 percentage points", 40)):
            with self.subTest(prompt=prompt):
                anchors, payload = probe_request(prompt, "brightness", current, 0, 100, 6, "%")
                self.assertEqual(anchors, [0, 20, 40, 60, 80, 100])
                self.assertEqual(payload["state"]["prompt"], prompt)
                self.assertEqual(payload["state"]["current"], current)
                self.assertEqual(set(payload["questions"]), {"numeric_score", "numeric_choice"})

    def test_default_step_and_instructions_can_be_experimented_with(self):
        _, payload = probe_request("make it warmer", "temperature", 20, 15, 30, 7, "°C",
                                   method="score", default_step=0.5, instructions="Experimental wording")
        self.assertEqual(payload["state"]["default_step"], 0.5)
        self.assertEqual(list(payload["questions"]), ["numeric_score"])
        self.assertEqual(payload["questions"]["numeric_score"]["instructions"], "Experimental wording")

    def test_score_is_scaled_and_choice_and_weighted_mean_remain_separate(self):
        result = decode_probe(RESPONSE, [0, 50, 100], expected=60)
        self.assertEqual(result["score"]["value"], 62.5)
        self.assertEqual(result["score"]["weighted_value"], 62.5)
        self.assertEqual(result["choice"]["value"], 50)
        self.assertEqual(result["choice"]["weighted_value"], 62.5)
        self.assertEqual(result["score"]["absolute_error"], 2.5)
        self.assertEqual(result["choice"]["absolute_error"], 10)

    def test_decimal_temperature_and_negative_ranges(self):
        self.assertEqual(decode_probe(RESPONSE, [15, 20, 25], "score")["score"]["value"], 21.25)
        self.assertEqual(decode_probe(RESPONSE, [-10, 0, 10], "score")["score"]["value"], 2.5)

    def test_probability_mean_can_hide_opposing_peaks(self):
        data = copy.deepcopy(RESPONSE)
        data["answers"]["numeric_score"].update(score=1, confidence=0,
                                                probabilities={"0": 0.5, "1": 0, "2": 0.5})
        result = decode_probe(data, [0, 50, 100], "score")["score"]
        self.assertEqual(result["value"], 50)
        self.assertEqual(result["confidence"], 0)
        self.assertEqual(result["probabilities"][1]["probability"], 0)

    def test_invalid_ranges_levels_and_values_are_rejected(self):
        for minimum, maximum, levels, current in (
            (0, 0, 6, 0), (100, 0, 6, 40), (0, 100, 11, 40),
            (0, 100, 1, 40), (0, 100, 6, 101), (0, float("inf"), 6, 40),
            (0, 100, 6, float("nan")),
        ):
            with self.subTest(values=(minimum, maximum, levels, current)), self.assertRaises(ValueError):
                probe_request("set it", "brightness", current, minimum, maximum, levels, "%")

    def test_malformed_numeric_outputs_are_rejected(self):
        for change in (
            {"score": 3}, {"score": float("nan")}, {"confidence": True},
            {"probabilities": {"0": 0, "1": 0, "2": 0}},
            {"probabilities": {"0": 1}}, {"type": "choice"},
        ):
            data = copy.deepcopy(RESPONSE)
            data["answers"]["numeric_score"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                decode_probe(data, [0, 50, 100])
        with self.assertRaises(ValueError):
            decode_probe({"error": "provider failed"}, [0, 50, 100])

    def test_cli_makes_one_decider_call_without_ha_and_keeps_expected_out_of_payload(self):
        decider = AsyncMock(return_value=RESPONSE)
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["numeric_probe.py", "increase by 15 percentage points",
                                       "--current", "40", "--levels", "3", "--expected", "55", "--json"]),
            patch("numeric_probe.load_dotenv"),
            patch("numeric_probe.setting", return_value="test-key"),
            patch("numeric_probe.call_decider", new=decider),
            patch("prototype.inspect_mcp", new=AsyncMock()) as inspector,
            patch("prototype.execute_mcp", new=AsyncMock()) as executor,
            redirect_stdout(output),
        ):
            self.assertEqual(asyncio.run(run()), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["expected"], 55)
        self.assertEqual(result["estimates"]["score"]["value"], 62.5)
        decider.assert_awaited_once()
        self.assertNotIn("expected", decider.await_args.args[0]["state"])
        inspector.assert_not_awaited()
        executor.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
