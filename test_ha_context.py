import importlib.util
import json
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class ComputedNameType(Enum):
    COMPUTED_NAME = 0


class EntityContextTests(unittest.TestCase):
    def snapshot(self, entry_aliases, area_aliases, registered=True):
        state = SimpleNamespace(
            entity_id="light.office", name="Office light", domain="light", state="on", attributes={},
        )
        entry = SimpleNamespace(aliases=entry_aliases, device_id=None, area_id="office") if registered else None
        area = SimpleNamespace(id="office", name="Office", aliases=area_aliases)
        registry = SimpleNamespace(async_get=lambda _: entry)
        helpers = SimpleNamespace(
            area_registry=SimpleNamespace(async_get=lambda _: SimpleNamespace(async_get_area=lambda _: area)),
            device_registry=SimpleNamespace(async_get=lambda _: None),
            entity_registry=SimpleNamespace(async_get=lambda _: registry),
        )
        dependencies = {
            "homeassistant.components.homeassistant": SimpleNamespace(async_should_expose=lambda *args: True),
            "homeassistant.exceptions": SimpleNamespace(HomeAssistantError=Exception),
            "homeassistant.helpers": helpers,
            "homeassistant.helpers.service": SimpleNamespace(async_get_all_descriptions=Mock()),
        }
        path = Path(__file__).parent / "custom_components/glow_assistant/ha_context.py"
        spec = importlib.util.spec_from_file_location("custom_components.glow_assistant._context_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependencies):
            spec.loader.exec_module(module)
        hass = SimpleNamespace(states=SimpleNamespace(async_all=lambda: [state]))
        return module.entity_context(hass, set())

    def test_mixed_computed_markers_and_text_aliases_remain_json_serializable(self):
        result = self.snapshot(
            ["Work light", ComputedNameType.COMPUTED_NAME, "Desk lamp"],
            ["Study", ComputedNameType.COMPUTED_NAME],
        )
        self.assertEqual(result[0]["name"], "Office light")
        self.assertEqual(result[0]["aliases"], ["Desk lamp", "Work light"])
        self.assertEqual(result[0]["area_aliases"], ["Study"])
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_computed_only_aliases_keep_the_state_name(self):
        result = self.snapshot([ComputedNameType.COMPUTED_NAME], [])
        self.assertEqual(result[0]["name"], "Office light")
        self.assertEqual(result[0]["aliases"], [])

    def test_entities_without_registry_entries_have_no_aliases(self):
        result = self.snapshot([], [], registered=False)
        self.assertEqual(result[0]["aliases"], [])
        self.assertEqual(result[0]["area_aliases"], [])


if __name__ == "__main__":
    unittest.main()
