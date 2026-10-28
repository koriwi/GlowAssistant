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
