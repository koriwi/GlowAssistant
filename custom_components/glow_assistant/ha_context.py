from homeassistant.components.homeassistant import async_should_expose
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import area_registry, device_registry, entity_registry
from homeassistant.helpers.service import async_get_all_descriptions

from .engine import ActionError


def entity_context(hass, excluded):
    areas = area_registry.async_get(hass)
    devices = device_registry.async_get(hass)
    registry = entity_registry.async_get(hass)
    entities = []
    for state in hass.states.async_all():
        if state.entity_id in excluded or not async_should_expose(hass, "conversation", state.entity_id):
            continue
        entry = registry.async_get(state.entity_id)
        device = devices.async_get(entry.device_id) if entry and entry.device_id else None
        area_id = (entry.area_id if entry else None) or (device.area_id if device else None)
        area = areas.async_get_area(area_id) if area_id else None
        attributes = {
            key: value for key, value in state.attributes.items()
            if key not in ("friendly_name", "icon", "entity_picture", "assumed_state")
        }
        if state.domain == "climate":
            attributes.setdefault("temperature_unit", hass.config.units.temperature_unit)
        entities.append({
            "entity_id": state.entity_id, "name": state.name,
            "aliases": sorted(alias for alias in entry.aliases if isinstance(alias, str)) if entry else [],
            "domain": state.domain, "state": state.state,
            "area_id": area.id if area else None, "area_name": area.name if area else None,
            "area_aliases": sorted(alias for alias in area.aliases if isinstance(alias, str)) if area else [],
            "attributes": attributes,
        })
    return entities


async def async_snapshot(hass, excluded):
    entities = entity_context(hass, excluded)
    domains = {entity["domain"] for entity in entities}
    descriptions = await async_get_all_descriptions(hass)
    services = {
        f"{domain}.{name}": description
        for domain, entries in descriptions.items() if domain in domains
        for name, description in entries.items()
        if description.get("target") is not None and hass.services.has_service(domain, name)
    }
    return {"entities": entities, "services": services}


async def async_execute(hass, excluded, context, action, arguments):
    domain, service = action.split(".", 1)
    allowed = {entity["entity_id"] for entity in entity_context(hass, excluded) if entity["domain"] == domain}
    if not arguments["entity_id"] or not set(arguments["entity_id"]) <= allowed:
        raise ActionError("Selected entities are no longer exposed or available.")
    # Explicit IDs keep area calls inside the exposed, non-excluded scope
    try:
        await hass.services.async_call(domain, service, arguments, blocking=True, context=context)
    except (HomeAssistantError, TimeoutError) as error:
        raise ActionError("Action failed; outcome unconfirmed.") from error
