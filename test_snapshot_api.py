import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


class View:
    def json(self, data, status=200):
        return SimpleNamespace(data=data, status=status)


class Request(dict):
    def __init__(self, admin=True, entry_id=None):
        super().__init__(hass_user=SimpleNamespace(is_admin=admin))
        self.query = {"entry_id": entry_id} if entry_id else {}


class SnapshotAPITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.loaded = object()
        self.snapshot = AsyncMock(return_value={"entities": [], "services": {}})
        path = Path(__file__).parent / "custom_components/glow_assistant/snapshot_api.py"
        spec = importlib.util.spec_from_file_location("custom_components.glow_assistant._snapshot_api_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "homeassistant.components.http": SimpleNamespace(HomeAssistantView=View),
            "homeassistant.config_entries": SimpleNamespace(ConfigEntryState=SimpleNamespace(LOADED=self.loaded)),
            "custom_components.glow_assistant.ha_context": SimpleNamespace(async_snapshot=self.snapshot),
        }):
            spec.loader.exec_module(module)
        self.hass = SimpleNamespace(config_entries=SimpleNamespace(async_entries=Mock()))
        self.view = module.SnapshotView(self.hass)

    def entry(self, entry_id="one", data=None, loaded=True):
        return SimpleNamespace(entry_id=entry_id, title=entry_id, data=data or {}, state=self.loaded if loaded else object())

    async def test_only_authenticated_administrators_can_read_the_snapshot(self):
        self.assertTrue(self.view.requires_auth)
        response = await self.view.get(Request(admin=False))
        self.assertEqual(response.status, 403)
        self.hass.config_entries.async_entries.assert_not_called()
        self.snapshot.assert_not_awaited()

    async def test_snapshot_reuses_ha_context_and_redacts_credentials_without_mutating_entry_data(self):
        data = {"excluded_entities": ["light.private"], "decider": {"api_key": "decider-secret"},
                "answer": {"api_key": "answer-secret"}, "general": {"api_key": "general-secret", "mcp_servers": [
                    {"name": "Tools", "enable": True, "url": "http://tools/mcp", "token": "mcp-secret"},
                ]}}
        original = copy.deepcopy(data)
        self.hass.config_entries.async_entries.return_value = [self.entry(data=data)]
        response = await self.view.get(Request())
        self.assertEqual(response.status, 200)
        self.assertEqual(response.data["entry_id"], "one")
        self.snapshot.assert_awaited_once_with(self.hass, {"light.private"})
        self.assertIs(response.data["snapshot"], self.snapshot.return_value)
        for group in ("decider", "answer", "general"):
            self.assertEqual(response.data["settings"][group]["api_key"], "")
        self.assertEqual(response.data["settings"]["general"]["mcp_servers"][0]["token"], "")
        self.assertEqual(data, original)

    async def test_multiple_entries_need_an_explicit_choice_and_unloaded_entries_are_not_exported(self):
        self.hass.config_entries.async_entries.return_value = [self.entry("one"), self.entry("two"), self.entry("off", loaded=False)]
        response = await self.view.get(Request())
        self.assertEqual(response.status, 400)
        self.snapshot.assert_not_awaited()
        response = await self.view.get(Request(entry_id="two"))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.data["entry_id"], "two")
        response = await self.view.get(Request(entry_id="off"))
        self.assertEqual(response.status, 404)
