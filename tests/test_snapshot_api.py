import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from custom_components.glow_assistant.actions import ActionError


class View:
    def json(self, data, status=200):
        return SimpleNamespace(data=data, status=status)


class Request(dict):
    def __init__(self, admin=True, entry_id=None, data=None):
        super().__init__(hass_user=SimpleNamespace(is_admin=admin, id="local-user"))
        self.query = {"entry_id": entry_id} if entry_id else {}
        self.json = AsyncMock(return_value=data)


class SnapshotAPITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.loaded = object()
        self.snapshot = AsyncMock(return_value={"entities": [], "services": {}})
        self.execute = AsyncMock(return_value=None)
        path = Path(__file__).parents[1] / "custom_components/glow_assistant/snapshot_api.py"
        spec = importlib.util.spec_from_file_location("custom_components.glow_assistant._snapshot_api_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "homeassistant.components.http": SimpleNamespace(HomeAssistantView=View),
            "homeassistant.config_entries": SimpleNamespace(ConfigEntryState=SimpleNamespace(LOADED=self.loaded)),
            "homeassistant.core": SimpleNamespace(Context=lambda **kwargs: SimpleNamespace(**kwargs)),
            "custom_components.glow_assistant.ha_context": SimpleNamespace(async_snapshot=self.snapshot, async_execute=self.execute),
        }):
            spec.loader.exec_module(module)
        self.hass = SimpleNamespace(config_entries=SimpleNamespace(async_entries=Mock()))
        self.view = module.SnapshotView(self.hass)
        self.execute_view = module.ExecuteView(self.hass)

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

    async def test_execute_requires_administrator_and_one_loaded_entry(self):
        self.assertTrue(self.execute_view.requires_auth)
        request = Request(admin=False)
        self.assertEqual((await self.execute_view.post(request)).status, 403)
        request.json.assert_not_awaited()
        self.hass.config_entries.async_entries.return_value = [self.entry("one"), self.entry("two")]
        self.assertEqual((await self.execute_view.post(Request())).status, 400)
        self.assertEqual((await self.execute_view.post(Request(entry_id="missing"))).status, 404)
        self.execute.assert_not_awaited()

    async def test_execute_uses_selected_entries_current_exclusions_and_user_context(self):
        entry = self.entry("two", {"excluded_entities": ["todo.private"]})
        self.hass.config_entries.async_entries.return_value = [self.entry("one"), entry]
        data = {"service": "todo.get_items", "arguments": {"entity_id": ["todo.shopping"]}}
        returned = {"todo.shopping": {"items": [{"summary": "Milk"}]}}
        self.execute.return_value = returned
        response = await self.execute_view.post(Request(entry_id="two", data=data))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.data, {"result": returned})
        args = self.execute.call_args.args
        self.assertIs(args[0], self.hass)
        self.assertEqual(args[1], {"todo.private"})
        self.assertEqual(args[2].user_id, "local-user")
        self.assertEqual(args[3:], ("todo.get_items", data["arguments"]))
        entry.data["excluded_entities"].append("todo.shopping")
        await self.execute_view.post(Request(entry_id="two", data=data))
        self.assertEqual(self.execute.call_args.args[1], {"todo.private", "todo.shopping"})

    async def test_execute_rejects_invalid_json_and_targets_before_dispatch(self):
        self.hass.config_entries.async_entries.return_value = [self.entry()]
        for data in (None, [], {}, {"service": "light.turn_on", "arguments": {}},
                     {"service": "invalid", "arguments": {"entity_id": ["light.a"]}},
                     {"service": "light.turn_on", "arguments": {"entity_id": "light.a"}},
                     {"service": "light.turn_on", "arguments": {"entity_id": [{}]}}):
            with self.subTest(data=data):
                response = await self.execute_view.post(Request(data=data))
                self.assertEqual(response.status, 400)
                self.assertFalse(response.data["command_sent"])
        request = Request()
        request.json.side_effect = ValueError("Invalid JSON")
        self.assertEqual((await self.execute_view.post(request)).status, 400)
        self.execute.assert_not_awaited()

    async def test_execute_preserves_outcome_on_failure_without_retrying(self):
        self.hass.config_entries.async_entries.return_value = [self.entry()]
        for sent in (False, True):
            with self.subTest(sent=sent):
                self.execute.reset_mock()
                self.execute.side_effect = ActionError("Rejected action", command_sent=sent)
                response = await self.execute_view.post(Request(data={"service": "light.turn_off", "arguments": {"entity_id": ["light.a"]}}))
                self.assertEqual(response.status, 502 if sent else 400)
                self.assertIs(response.data["command_sent"], sent)
                self.execute.assert_awaited_once()

    async def test_get_never_executes(self):
        self.assertEqual((await self.execute_view.get(Request())).status, 405)
        self.execute.assert_not_awaited()
