from copy import deepcopy
from http import HTTPStatus

from homeassistant.components.http import HomeAssistantView
from homeassistant.config_entries import ConfigEntryState

from .const import DOMAIN
from .ha_context import async_snapshot
from .settings import normalize_settings


class SnapshotView(HomeAssistantView):
    url = "/api/glow_assistant/snapshot"
    name = "api:glow_assistant:snapshot"
    requires_auth = True

    def __init__(self, hass):
        self.hass = hass

    async def get(self, request):
        if not request["hass_user"].is_admin:
            return self.json({"error": "An administrator token is required."}, HTTPStatus.FORBIDDEN)
        entries = [entry for entry in self.hass.config_entries.async_entries(DOMAIN)
                   if entry.state is ConfigEntryState.LOADED]
        entry_id = request.query.get("entry_id")
        if entry_id:
            entries = [entry for entry in entries if entry.entry_id == entry_id]
        if not entries:
            return self.json({"error": "No matching loaded GlowAssistant entry."}, HTTPStatus.NOT_FOUND)
        if len(entries) != 1:
            return self.json({"error": "Multiple GlowAssistant entries; set GLOW_ENTRY_ID.",
                              "entries": [{"entry_id": entry.entry_id, "name": entry.title} for entry in entries]},
                             HTTPStatus.BAD_REQUEST)
        entry = entries[0]
        settings = normalize_settings(deepcopy(dict(entry.data)))
        snapshot = await async_snapshot(self.hass, set(settings["excluded_entities"]))
        for group in ("decider", "answer", "general"):
            settings[group]["api_key"] = ""
        for server in settings["general"]["mcp_servers"]:
            server["token"] = ""
        return self.json({"entry_id": entry.entry_id, "snapshot": snapshot, "settings": settings})
