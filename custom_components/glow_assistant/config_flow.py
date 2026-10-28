import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import section
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    NumberSelector,
    SelectSelector,
    TextSelector,
)

from .client import ModelClient, ModelError
from .const import DOMAIN
from .engine import choice_question, read_choice
from .settings import DEFAULTS, normalize_settings, validate_settings


def _number(minimum, maximum, step=1):
    return NumberSelector({"min": minimum, "max": maximum, "step": step, "mode": "box"})


def settings_schema(values):
    schema = {vol.Required("name", default=values["name"]): TextSelector()}
    for group in ("decider", "answer", "general"):
        fields = {}
        for field, value in values[group].items():
            if field in ("url", "model", "api_key"):
                selector = TextSelector({"type": "password" if field == "api_key" else "text"})
            elif field == "enable":
                selector = BooleanSelector()
            elif field.endswith("_prompt") or field == "prompt":
                selector = TextSelector({"multiline": True})
            elif field == "timeout":
                selector = _number(1, 600)
            elif field == "max_input_chars":
                selector = _number(100, 2000000)
            elif field == "max_tokens":
                selector = _number(1, 100000)
            elif field == "temperature":
                selector = _number(0, 2, 0.1)
            elif field.endswith("_confidence"):
                selector = _number(0, 1, 0.01)
            elif field.endswith("_levels"):
                selector = _number(2, 10)
            elif field == "temperature_baseline":
                selector = SelectSelector({"options": ["target", "measured"], "translation_key": "temperature_baseline"})
            elif field.endswith(("_step", "_default_change")):
                selector = _number(0.01, 100, 0.01)
            else:
                selector = _number(-100 if field.startswith("temperature_") else 0,
                                   200 if field.startswith("temperature_") else 100, 0.1)
            fields[vol.Optional(field, default=value)] = selector
        fields[vol.Optional("test_connection", default=False)] = BooleanSelector()
        schema[vol.Required(group)] = section(vol.Schema(fields), {"collapsed": False})
    schema[vol.Optional("excluded_entities", default=values["excluded_entities"])] = EntitySelector({"multiple": True})
    for field in ("unsure_reply", "error_reply", "answer_error_reply"):
        schema[vol.Required(field, default=values[field])] = TextSelector()
    return vol.Schema(schema)


async def async_test_models(hass, data, raw):
    if not any(raw.get(group, {}).get("test_connection") for group in ("decider", "answer", "general")):
        return
    client = ModelClient(async_get_clientsession(hass))
    for group in ("decider", "answer", "general"):
        if not raw.get(group, {}).get("test_connection"):
            continue
        if group == "general" and not data[group]["enable"]:
            continue
        if group == "decider":
            options = {"ok": "The endpoint is working."}
            response = await client.decide(data[group], {"prompt": "Select ok."}, {
                "connection": choice_question(options, "Select ok to confirm this connection."),
            })
            read_choice(response, "connection", options, 0)
        else:
            await client.chat(data[group], [{"role": "user", "content": "Reply with OK."}])


class _SettingsForm:
    async def _async_form(self, step, user_input, current):
        errors = {}
        values = normalize_settings(current)
        if user_input is not None:
            values = normalize_settings(user_input)
            try:
                settings = validate_settings(values)
                await async_test_models(self.hass, settings, user_input)
            except (ValueError, ModelError) as error:
                errors["base"] = "invalid_settings"
                detail = str(error)
            else:
                for group in ("decider", "answer", "general"):
                    settings[group].pop("test_connection", None)
                return settings, None
        else:
            detail = ""
        return None, self.async_show_form(
            step_id=step, data_schema=settings_schema(values), errors=errors,
            description_placeholders={"error": detail},
        )


class GlowConfigFlow(_SettingsForm, config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        settings, form = await self._async_form("user", user_input, DEFAULTS)
        return form if form is not None else self.async_create_entry(title=settings["name"], data=settings)

    async def async_step_reconfigure(self, user_input=None):
        settings, form = await self._async_form("reconfigure", user_input, self._get_reconfigure_entry().data)
        if form is not None:
            return form
        return self.async_update_reload_and_abort(
            self._get_reconfigure_entry(), data_updates=settings, title=settings["name"],
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return GlowOptionsFlow()


class GlowOptionsFlow(_SettingsForm, config_entries.OptionsFlow):
    async def async_step_init(self, user_input=None):
        settings, form = await self._async_form("init", user_input, self.config_entry.data)
        if form is not None:
            return form
        self.hass.config_entries.async_update_entry(self.config_entry, data=settings, title=settings["name"])
        return self.async_create_entry(title="", data={})
