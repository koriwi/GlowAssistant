from copy import deepcopy
from http import HTTPStatus

from homeassistant.components.http import HomeAssistantView
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context

from .actions import ActionError
from .const import DOMAIN
from .ha_context import async_execute, async_snapshot
from .settings import normalize_settings


class SnapshotView(HomeAssistantView):
    url = "/api/glow_assistant/snapshot"
    name = "api:glow_assistant:snapshot"
    requires_auth = True

    def __init__(self, hass):
        self.hass = hass

    def entry(self, request):
        if not request["hass_user"].is_admin:
            return None, self.json({"error": "An administrator token is required."}, HTTPStatus.FORBIDDEN)
        entries = [entry for entry in self.hass.config_entries.async_entries(DOMAIN)
                   if entry.state is ConfigEntryState.LOADED]
        entry_id = request.query.get("entry_id")
        if entry_id:
            entries = [entry for entry in entries if entry.entry_id == entry_id]
        if not entries:
            return None, self.json({"error": "No matching loaded GlowAssistant entry."}, HTTPStatus.NOT_FOUND)
        if len(entries) != 1:
            return None, self.json({"error": "Multiple GlowAssistant entries; set GLOW_ENTRY_ID.",
                              "entries": [{"entry_id": entry.entry_id, "name": entry.title} for entry in entries]},
                             HTTPStatus.BAD_REQUEST)
        return entries[0], None

    async def get(self, request):
        entry, error = self.entry(request)
        if error is not None:
            return error
        settings = normalize_settings(deepcopy(dict(entry.data)))
        snapshot = await async_snapshot(self.hass, set(settings["excluded_entities"]))
        for group in ("decider", "answer", "general"):
            settings[group]["api_key"] = ""
        for server in settings["general"]["mcp_servers"]:
            server["token"] = ""
        return self.json({"entry_id": entry.entry_id, "snapshot": snapshot, "settings": settings})


class ExecuteView(SnapshotView):
    url = "/api/glow_assistant/execute"
    name = "api:glow_assistant:execute"

    async def post(self, request):
        entry, error = self.entry(request)
        if error is not None:
            return error
        try:
            data = await request.json()
        except ValueError:
            return self.json({"error": "Invalid JSON.", "command_sent": False}, HTTPStatus.BAD_REQUEST)
        service = data.get("service") if isinstance(data, dict) else None
        arguments = data.get("arguments") if isinstance(data, dict) else None
        targets = arguments.get("entity_id") if isinstance(arguments, dict) else None
        if (not isinstance(service, str) or service.count(".") != 1 or not all(service.split("."))
                or not isinstance(targets, list) or not targets or not all(isinstance(target, str) for target in targets)):
            return self.json({"error": "A service and explicit entity_id list are required.", "command_sent": False}, HTTPStatus.BAD_REQUEST)
        try:
            result = await async_execute(self.hass, set(entry.data.get("excluded_entities", [])),
                                         Context(user_id=request["hass_user"].id), service, arguments)
        except ActionError as error:
            return self.json({"error": str(error), "command_sent": error.command_sent},
                             HTTPStatus.BAD_GATEWAY if error.command_sent else HTTPStatus.BAD_REQUEST)
        return self.json({"result": result})

    async def get(self, request):
        return self.json({"error": "Use POST to execute an action."}, HTTPStatus.METHOD_NOT_ALLOWED)
