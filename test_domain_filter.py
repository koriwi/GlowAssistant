import copy
import unittest

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.domain_filter import decode_domains


def response_for(choices):
    return {"answers": {
        f"domain_{domain}": {
            "type": "choice", "choice": choice, "confidence": confidence,
            "probabilities": {"keep": 0.9 if choice == "keep" else 0.1,
                              "drop": 0.1 if choice == "keep" else 0.9},
        }
        for domain, (choice, confidence) in choices.items()
    }}


class DomainFilterTests(unittest.TestCase):
    def test_multiple_kept_domains_and_uncertain_drops_are_retained(self):
        domains = ["binary_sensor", "sensor", "light", "cover"]
        response = response_for({"binary_sensor": ("keep", 0.8), "sensor": ("keep", 0.2),
                                 "light": ("drop", 0.9), "cover": ("drop", 0.4)})
        result = decode_domains(response, domains, 0.5)
        self.assertEqual(result["kept_domains"], ["binary_sensor", "sensor", "cover"])
        self.assertEqual(result["dropped_domains"], ["light"])
        self.assertIsNone(result["fallback"])

    def test_drop_at_threshold_is_uncertain_and_kept(self):
        result = decode_domains(response_for({"sensor": ("drop", 0.5)}), ["sensor"], 0.5)
        self.assertEqual(result["kept_domains"], ["sensor"])
        self.assertIsNone(result["fallback"])

    def test_all_drop_falls_back_to_all_domains(self):
        result = decode_domains(response_for({"sensor": ("drop", 0.9), "light": ("drop", 0.9)}), ["sensor", "light"], 0.5)
        self.assertEqual(result["kept_domains"], ["sensor", "light"])
        self.assertEqual(result["dropped_domains"], [])
        self.assertEqual(result["fallback"], "all_domains")

    def test_missing_or_unadvertised_domain_decisions_are_rejected(self):
        for response in (None, {}, {"answers": {}}, response_for({"unknown": ("keep", 1)})):
            with self.subTest(response=response), self.assertRaises(ModelError):
                decode_domains(response, ["sensor"], 0.5)

    def test_invalid_decisions_are_rejected(self):
        valid = response_for({"sensor": ("keep", 0.9)})
        for change in (
            {"choice": "unsure"}, {"type": "score"}, {"confidence": True},
            {"confidence": float("nan")}, {"confidence": 1.1},
            {"probabilities": {"keep": 1}}, {"probabilities": {"keep": 1, "drop": -1}},
        ):
            response = copy.deepcopy(valid)
            response["answers"]["domain_sensor"].update(change)
            with self.subTest(change=change), self.assertRaises(ModelError):
                decode_domains(response, ["sensor"], 0.5)
