import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

from custom_components.glow_assistant.settings import normalize_settings


@unittest.skipUnless(importlib.util.find_spec("homeassistant"), "Home Assistant is not installed; run the HA CI suite")
class HomeAssistantContractTests(unittest.IsolatedAsyncioTestCase):
    def settings(self):
        return normalize_settings({
            "decider": {"url": "http://server:8321/v1/systemone"},
            "answer": {"url": "http://server:8322/v1/chat/completions", "model": "small-model"},
        })

    async def test_config_flow_renders_three_sections_and_creates_entry(self):
        from custom_components.glow_assistant.config_flow import GlowConfigFlow

        flow = GlowConfigFlow()
        flow.hass = Mock()
        form = await flow.async_step_user()
        for key, block in form["data_schema"].schema.items():
            if key.schema not in ("decider", "answer", "general"):
                continue
            api_key_selector = next(selector for field, selector in block.schema.schema.items() if field.schema == "api_key")
            self.assertEqual(api_key_selector.config["type"], "password")
            if key.schema == "decider":
                prompt = next(selector for field, selector in block.schema.schema.items() if field.schema == "text_prompt")
                confidence = next(selector for field, selector in block.schema.schema.items() if field.schema == "text_confidence")
                self.assertTrue(prompt.config["multiline"])
                self.assertEqual(confidence.config["min"], 0)
                self.assertEqual(confidence.config["max"], 1)
        validated = form["data_schema"](self.settings())
        self.assertEqual(set(validated) & {"decider", "answer", "general"}, {"decider", "answer", "general"})
        result = await flow.async_step_user(validated)
        self.assertEqual(result["type"], "create_entry")
        self.assertEqual(result["data"]["answer"]["model"], "small-model")
        self.assertFalse(result["data"]["general"]["enable"])
        self.assertNotIn("test_connection", result["data"]["decider"])

    async def test_invalid_form_preserves_values_and_returns_error(self):
        from custom_components.glow_assistant.config_flow import GlowConfigFlow

        flow = GlowConfigFlow()
        flow.hass = Mock()
        settings = self.settings()
        settings["answer"]["model"] = ""
        result = await flow.async_step_user(settings)
        self.assertEqual(result["type"], "form")
        self.assertEqual(result["errors"], {"base": "invalid_settings"})
        self.assertIn("answer", result["description_placeholders"]["error"])

    async def test_options_update_existing_data_and_preserve_prompts(self):
        from custom_components.glow_assistant.config_flow import GlowOptionsFlow

        data = self.settings()
        data["decider"]["target_prompt"] = "My custom target instructions"
        entry = SimpleNamespace(data=data)
        flow = GlowOptionsFlow()
        flow.hass = Mock()
        with patch.object(GlowOptionsFlow, "config_entry", new_callable=PropertyMock, return_value=entry):
            form = await flow.async_step_init()
            updated = form["data_schema"](data)
            updated["general"]["enable"] = False
            result = await flow.async_step_init(updated)
        self.assertEqual(result["type"], "create_entry")
        saved = flow.hass.config_entries.async_update_entry.call_args.kwargs["data"]
        self.assertEqual(saved["decider"]["target_prompt"], "My custom target instructions")
        self.assertEqual(saved["answer"]["model"], "small-model")

    async def test_reconfigure_updates_same_entry(self):
        from custom_components.glow_assistant.config_flow import GlowConfigFlow

        entry = SimpleNamespace(data=self.settings())
        flow = GlowConfigFlow()
        flow.hass = Mock()
        with patch.object(flow, "_get_reconfigure_entry", return_value=entry), \
                patch.object(flow, "async_update_reload_and_abort", return_value={"type": "abort"}) as update:
            await flow.async_step_reconfigure(self.settings())
        self.assertIs(update.call_args.args[0], entry)
        self.assertEqual(update.call_args.kwargs["data_updates"]["answer"]["model"], "small-model")

    async def test_exposure_exclusions_aliases_and_device_area(self):
        from custom_components.glow_assistant import ha_context

        states = [SimpleNamespace(entity_id=f"light.{name}", domain="light", name=name, state="on", attributes={})
                  for name in ("allowed", "excluded", "hidden")]
        hass = Mock()
        hass.states.async_all.return_value = states
        registry = SimpleNamespace(async_get=lambda _: SimpleNamespace(device_id="device", area_id=None, aliases={"Alias"}))
        devices = SimpleNamespace(async_get=lambda _: SimpleNamespace(area_id="office"))
        areas = SimpleNamespace(async_get_area=lambda _: SimpleNamespace(id="office", name="Office", aliases={"Study"}))
        with patch.object(ha_context.entity_registry, "async_get", return_value=registry), \
                patch.object(ha_context.device_registry, "async_get", return_value=devices), \
                patch.object(ha_context.area_registry, "async_get", return_value=areas), \
                patch.object(ha_context, "async_should_expose", side_effect=lambda h, assistant, entity_id: entity_id != "light.hidden"):
            entities = ha_context.entity_context(hass, {"light.excluded"})
        self.assertEqual([item["entity_id"] for item in entities], ["light.allowed"])
        self.assertEqual(entities[0]["area_id"], "office")
        self.assertEqual(entities[0]["aliases"], ["Alias"])
        self.assertEqual(entities[0]["area_aliases"], ["Study"])

    async def test_direct_calls_preserve_context_and_recheck_exposure(self):
        from custom_components.glow_assistant import ha_context
        from custom_components.glow_assistant.engine import ActionError

        hass = SimpleNamespace(services=SimpleNamespace(async_call=AsyncMock()))
        context = object()
        with patch.object(ha_context, "entity_context", return_value=[{"entity_id": "light.allowed", "domain": "light"}]):
            await ha_context.async_execute(hass, set(), context, "light.turn_off", {"entity_id": ["light.allowed"]})
            with self.assertRaises(ActionError):
                await ha_context.async_execute(hass, set(), context, "light.turn_off", {"entity_id": ["light.hidden"]})
        hass.services.async_call.assert_awaited_once_with(
            "light", "turn_off", {"entity_id": ["light.allowed"]}, blocking=True, context=context,
        )

    async def test_conversation_returns_speech_and_preserves_conversation_id(self):
        from custom_components.glow_assistant.conversation import GlowConversationEntity

        entry = SimpleNamespace(data=self.settings(), entry_id="test-entry")
        agent = GlowConversationEntity(entry)
        agent.hass = Mock()
        chat_log = Mock(conversation_id="conversation-123")
        user_input = SimpleNamespace(text="test", language="de", agent_id="conversation.glow", context=object())
        with patch("custom_components.glow_assistant.conversation.async_snapshot", new_callable=AsyncMock, return_value={}), \
                patch("custom_components.glow_assistant.conversation.async_get_clientsession", return_value=Mock()), \
                patch("custom_components.glow_assistant.conversation.AgentEngine") as engine:
            engine.return_value.process = AsyncMock(return_value={
                "answer": "ich bin mir nicht sicher", "route": "unsure", "service": None, "timings_ms": {}, "usage": {},
            })
            result = await agent._async_handle_message(user_input, chat_log)
        self.assertEqual(result.conversation_id, "conversation-123")
        self.assertEqual(result.response.speech["plain"]["speech"], "ich bin mir nicht sicher")
        chat_log.async_add_assistant_content_without_tools.assert_called_once()

    async def test_query_answer_failure_is_logged_with_its_cause(self):
        from custom_components.glow_assistant.conversation import GlowConversationEntity

        entry = SimpleNamespace(data=self.settings(), entry_id="test-entry")
        agent = GlowConversationEntity(entry)
        agent.hass = Mock()
        chat_log = Mock(conversation_id="conversation-123")
        user_input = SimpleNamespace(text="ist das licht im buero an?", language="de", agent_id="conversation.glow", context=object())
        with patch("custom_components.glow_assistant.conversation.async_snapshot", new_callable=AsyncMock, return_value={}), \
                patch("custom_components.glow_assistant.conversation.async_get_clientsession", return_value=Mock()), \
                patch("custom_components.glow_assistant.conversation.AgentEngine") as engine, \
                self.assertLogs("custom_components.glow_assistant.conversation", level="WARNING") as logs:
            engine.return_value.process = AsyncMock(return_value={
                "answer": self.settings()["error_reply"], "answer_error": "Model endpoint returned HTTP 401.",
                "route": "smarthome", "service": "query_state", "timings_ms": {}, "usage": {},
            })
            result = await agent._async_handle_message(user_input, chat_log)
        self.assertIn("Smart-home answer generation failed: Model endpoint returned HTTP 401.", logs.output[0])
        self.assertEqual(result.response.speech["plain"]["speech"], self.settings()["error_reply"])

    async def test_native_chat_log_accepts_external_decider_calls_and_results(self):
        from homeassistant.components.conversation.chat_log import ChatLog

        from custom_components.glow_assistant.chat_log_trace import ChatLogTrace

        chat_log = ChatLog(hass=SimpleNamespace(data={}), conversation_id="conversation-trace")
        trace = ChatLogTrace(chat_log, "conversation.glow")
        trace("service", {"model": "decider"})
        trace("service", {}, {"duration_ms": 42, "decisions": {"service": {"choice": "query_state", "confidence": 0.9}}})
        call_message, result_message = chat_log.content[-2:]
        self.assertEqual(call_message.role, "assistant")
        self.assertTrue(call_message.tool_calls[0].external)
        self.assertEqual(result_message.role, "tool_result")
        self.assertEqual(result_message.tool_call_id, call_message.tool_calls[0].id)
        self.assertEqual(result_message.tool_result["duration_ms"], 42)

    async def test_native_chat_log_accepts_visible_failure_summary_without_adding_it_to_history(self):
        from homeassistant.components.conversation.chat_log import (
            ChatLog,
            UserContent,
        )

        from custom_components.glow_assistant.chat_log_trace import ChatLogTrace
        from custom_components.glow_assistant.debug import model_failure
        from custom_components.glow_assistant.history import conversation_history

        chat_log = ChatLog(hass=SimpleNamespace(data={}), conversation_id="failure-trace")
        chat_log.async_add_user_content(UserContent(content="test"))
        trace = ChatLogTrace(chat_log, "conversation.glow")
        trace.failure(model_failure("answer", "Model reply reached its output token limit."))
        bubble, details = chat_log.content[-2:]
        self.assertIn("Increase Maximum output tokens", bubble.content)
        self.assertEqual(details.tool_result["stage"], "answer")
        self.assertEqual(conversation_history(chat_log.content, 10), [])

    async def test_context_failure_reaches_debug_chat_and_keeps_the_configured_spoken_reply(self):
        from custom_components.glow_assistant.conversation import GlowConversationEntity

        agent = GlowConversationEntity(SimpleNamespace(data=self.settings(), entry_id="test-entry"))
        agent.hass = Mock()
        chat_log = Mock(conversation_id="conversation-error")
        user_input = SimpleNamespace(text="test", language="de", agent_id="conversation.glow", context=object())
        with patch("custom_components.glow_assistant.conversation.async_snapshot", new_callable=AsyncMock,
                   side_effect=TypeError("private exception details")), \
                self.assertLogs("custom_components.glow_assistant.conversation", level="ERROR"):
            result = await agent._async_handle_message(user_input, chat_log)
        bubble, details, spoken = [call.args[0] for call in chat_log.async_add_assistant_content_without_tools.call_args_list]
        self.assertIn("Read Home Assistant devices", bubble.content)
        self.assertIn("unexpected integration error", bubble.content)
        self.assertNotIn("private exception details", bubble.content)
        self.assertEqual(details.tool_result["stage"], "context")
        self.assertEqual(spoken.content, self.settings()["error_reply"])
        self.assertEqual(result.response.speech["plain"]["speech"], self.settings()["error_reply"])

    async def test_mcp_servers_use_structured_fields_and_mask_the_token(self):
        from custom_components.glow_assistant.config_flow import settings_schema

        schema = settings_schema(self.settings())
        general = next(value for key, value in schema.schema.items() if key.schema == "general")
        selector = next(value for key, value in general.schema.schema.items() if key.schema == "mcp_servers")
        self.assertTrue(selector.config["multiple"])
        self.assertEqual(selector.config["fields"]["token"]["selector"]["text"]["type"], "password")
        self.assertIn("tools", selector.config["fields"])
        values = self.settings()
        values.update(history_enabled=True, history_messages=3)
        values["general"]["mcp_servers"] = [{"name": "Search", "url": "http://server:8000/mcp", "enable": True}]
        validated = schema(values)
        self.assertEqual(validated["history_messages"], 3)
        self.assertEqual(validated["general"]["mcp_servers"][0]["name"], "Search")

    async def test_handler_supplies_only_prior_plain_messages_to_the_engine(self):
        from homeassistant.components.conversation.chat_log import (
            AssistantContent,
            UserContent,
        )

        from custom_components.glow_assistant.conversation import GlowConversationEntity

        settings = self.settings()
        settings.update(history_enabled=True, history_messages=2)
        agent = GlowConversationEntity(SimpleNamespace(data=settings, entry_id="test-history"))
        agent.hass = Mock()
        content = [UserContent(content="Turn on the kitchen music"),
                   AssistantContent(agent_id="conversation.glow", content="The kitchen music is playing."),
                   UserContent(content="now stop it")]
        chat_log = SimpleNamespace(content=content, conversation_id="history-test",
                                   async_add_assistant_content_without_tools=Mock())
        user_input = SimpleNamespace(text="now stop it", language="en", agent_id="conversation.glow", context=object())
        with patch("custom_components.glow_assistant.conversation.async_snapshot", new_callable=AsyncMock, return_value={}), \
                patch("custom_components.glow_assistant.conversation.async_get_clientsession", return_value=Mock()), \
                patch("custom_components.glow_assistant.conversation.AgentEngine") as engine:
            engine.return_value.process = AsyncMock(return_value={
                "answer": "Stopped.", "route": "smarthome", "service": "media_player.media_stop", "timings_ms": {}, "usage": {},
            })
            await agent._async_handle_message(user_input, chat_log)
        self.assertEqual(engine.call_args.kwargs["history"], [
            {"role": "user", "content": "Turn on the kitchen music"},
            {"role": "assistant", "content": "The kitchen music is playing."},
        ])

    async def test_general_connection_test_discovers_mcp_tools_without_execution(self):
        from custom_components.glow_assistant.config_flow import async_test_models

        settings = self.settings()
        settings["general"].update(enable=True, mcp_servers=[{"name": "Search", "enable": True}])
        with patch("custom_components.glow_assistant.config_flow.async_get_clientsession", return_value=Mock()), \
                patch("custom_components.glow_assistant.config_flow.ModelClient") as model, \
                patch("custom_components.glow_assistant.config_flow.MCPTools") as tools:
            model.return_value.chat = AsyncMock(return_value=("OK", {}))
            tools.return_value.__aenter__ = AsyncMock(return_value=SimpleNamespace())
            tools.return_value.__aexit__ = AsyncMock(return_value=False)
            tools.return_value.call = AsyncMock()
            await async_test_models(Mock(), settings, {"general": {"test_connection": True}})
        model.return_value.chat.assert_awaited_once()
        tools.return_value.__aenter__.assert_awaited_once()
        tools.return_value.call.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
