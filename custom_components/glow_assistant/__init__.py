from .const import DOMAIN


def _config_schema(config):
    # Keep local engine imports independent of HA.
    from homeassistant.helpers import config_validation as cv

    return cv.config_entry_only_config_schema(DOMAIN)(config)


CONFIG_SCHEMA = _config_schema


async def async_setup(hass, config):
    from .snapshot_api import ExecuteView, SnapshotView

    hass.http.register_view(SnapshotView(hass))
    hass.http.register_view(ExecuteView(hass))
    return True


async def async_setup_entry(hass, entry):
    from homeassistant.const import Platform

    entry.async_on_unload(entry.add_update_listener(_async_reload))
    await hass.config_entries.async_forward_entry_setups(entry, [Platform.CONVERSATION])
    return True


async def _async_reload(hass, entry):
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass, entry):
    from homeassistant.const import Platform

    return await hass.config_entries.async_unload_platforms(entry, [Platform.CONVERSATION])
