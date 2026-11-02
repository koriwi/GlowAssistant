import importlib.util
import json
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


class ComputedNameType(Enum):
    COMPUTED_NAME = 0


class SupportsResponse(Enum):
    NONE = "none"
    OPTIONAL = "optional"
    ONLY = "only"


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
        module = self.load_context(helpers)
        hass = SimpleNamespace(states=SimpleNamespace(async_all=lambda: [state]))
        return module.entity_context(hass, set())

    def load_context(self, helpers):
        dependencies = {
            "homeassistant.components.homeassistant": SimpleNamespace(async_should_expose=lambda *args: True),
            "homeassistant.core": SimpleNamespace(SupportsResponse=SupportsResponse),
            "homeassistant.exceptions": SimpleNamespace(HomeAssistantError=Exception),
            "homeassistant.helpers": helpers,
            "homeassistant.helpers.service": SimpleNamespace(async_get_all_descriptions=Mock()),
        }
        path = Path(__file__).parent / "custom_components/glow_assistant/ha_context.py"
        spec = importlib.util.spec_from_file_location("custom_components.glow_assistant._context_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependencies):
            spec.loader.exec_module(module)
        return module

    def test_speaker_area_uses_device_room_and_returns_none_for_missing_context(self):
        devices = {"speaker": SimpleNamespace(area_id="kitchen"),
                   "no_room": SimpleNamespace(area_id=None), "deleted_room": SimpleNamespace(area_id="deleted")}
        areas = {"kitchen": SimpleNamespace(name="Küche")}
        helpers = SimpleNamespace(
            device_registry=SimpleNamespace(async_get=lambda _: SimpleNamespace(async_get=devices.get)),
            area_registry=SimpleNamespace(async_get=lambda _: SimpleNamespace(async_get_area=areas.get)),
            entity_registry=Mock(),
        )
        module = self.load_context(helpers)
        for device_id, expected in ((None, None), ("missing", None), ("no_room", None),
                                    ("deleted_room", None), ("speaker", "Küche")):
            with self.subTest(device_id=device_id):
                self.assertEqual(module.speaker_area(Mock(), device_id), expected)

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


class HAExecutionTests(unittest.IsolatedAsyncioTestCase):
    def context_module(self):
        return EntityContextTests().load_context(SimpleNamespace(area_registry=Mock(), device_registry=Mock(), entity_registry=Mock()))

    async def test_each_service_call_rechecks_exposure_and_preserves_user_context(self):
        from custom_components.glow_assistant.actions import ActionError

        module = self.context_module()
        hass = SimpleNamespace(services=SimpleNamespace(has_service=Mock(return_value=True), async_call=AsyncMock(),
                                                       supports_response=Mock(return_value=SupportsResponse.NONE)))
        context = object()
        arguments = {"entity_id": ["light.allowed"], "brightness_pct": 50}
        with patch.object(module, "entity_context", side_effect=[[{"entity_id": "light.allowed", "domain": "light"}], []]) as exposed:
            await module.async_execute(hass, {"light.excluded"}, context, "light.turn_on", arguments)
            with self.assertRaises(ActionError) as raised:
                await module.async_execute(hass, {"light.excluded"}, context, "light.turn_on", arguments)
        self.assertFalse(raised.exception.command_sent)
        self.assertEqual(exposed.call_count, 2)
        hass.services.async_call.assert_awaited_once_with("light", "turn_on", arguments, blocking=True, context=context, return_response=False)

    async def test_collects_mandatory_and_optional_response_data(self):
        module = self.context_module()
        data = {"todo.shopping": {"items": [{"summary": "Zwiebeln", "status": "needs_action"}]}}
        for support in (SupportsResponse.ONLY, SupportsResponse.OPTIONAL):
            with self.subTest(support=support):
                hass = SimpleNamespace(services=SimpleNamespace(
                    has_service=Mock(return_value=True), supports_response=Mock(return_value=support),
                    async_call=AsyncMock(return_value=data),
                ))
                arguments = {"entity_id": ["todo.shopping"]}
                context = object()
                with patch.object(module, "entity_context", return_value=[{"entity_id": "todo.shopping", "domain": "todo"}]):
                    response = await module.async_execute(hass, set(), context, "todo.get_items", arguments)
                self.assertEqual(response, data)
                hass.services.async_call.assert_awaited_once_with(
                    "todo", "get_items", arguments, blocking=True, context=context, return_response=True,
                )

    async def test_invalid_or_broad_targets_and_unavailable_services_are_never_sent(self):
        from custom_components.glow_assistant.actions import ActionError

        module = self.context_module()
        hass = SimpleNamespace(services=SimpleNamespace(has_service=Mock(return_value=True), async_call=AsyncMock()))
        with patch.object(module, "entity_context", return_value=[{"entity_id": "light.allowed", "domain": "light"}]):
            for arguments in ({}, {"entity_id": []}, {"entity_id": "light.allowed"}, {"entity_id": ["light.hidden"]},
                              {"entity_id": ["light.allowed"], "area_id": "all"}):
                with self.subTest(arguments=arguments), self.assertRaises(ActionError) as raised:
                    await module.async_execute(hass, set(), None, "light.turn_on", arguments)
                self.assertFalse(raised.exception.command_sent)
            hass.services.has_service.return_value = False
            with self.assertRaises(ActionError):
                await module.async_execute(hass, set(), None, "light.turn_on", {"entity_id": ["light.allowed"]})
        hass.services.async_call.assert_not_awaited()

    async def test_backend_failure_is_unconfirmed_and_not_retried(self):
        from custom_components.glow_assistant.actions import ActionError

        module = self.context_module()
        hass = SimpleNamespace(services=SimpleNamespace(has_service=Mock(return_value=True),
                                                       supports_response=Mock(return_value=SupportsResponse.NONE),
                                                       async_call=AsyncMock(side_effect=module.HomeAssistantError("backend rejected"))))
        with patch.object(module, "entity_context", return_value=[{"entity_id": "light.allowed", "domain": "light"}]):
            with self.assertRaises(ActionError) as raised:
                await module.async_execute(hass, set(), None, "light.turn_on", {"entity_id": ["light.allowed"]})
        self.assertTrue(raised.exception.command_sent)
        self.assertIn("backend rejected", str(raised.exception.__cause__))
        hass.services.async_call.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
