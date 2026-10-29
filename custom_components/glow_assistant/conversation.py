import logging

from homeassistant.components import conversation
from homeassistant.const import MATCH_ALL
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .chat_log_trace import ChatLogTrace
from .client import ModelClient, ModelError
from .debug import failure_details, model_failure
from .engine import AgentEngine
from .ha_context import async_execute, async_snapshot
from .history import conversation_history
from .settings import normalize_settings

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry, async_add_entities):
    async_add_entities([GlowConversationEntity(entry)])


class GlowConversationEntity(conversation.ConversationEntity, conversation.AbstractConversationAgent):
    _attr_has_entity_name = True
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry):
        self.entry = entry
        self.settings = normalize_settings(entry.data)
        self._attr_name = self.settings["name"]
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self):
        return MATCH_ALL

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self):
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(self, user_input, chat_log):
        excluded = set(self.settings["excluded_entities"])
        trace = ChatLogTrace(chat_log, user_input.agent_id)

        async def execute(action, arguments):
            await async_execute(self.hass, excluded, user_input.context, action, arguments)

        try:
            snapshot = await async_snapshot(self.hass, excluded)
            client = ModelClient(async_get_clientsession(self.hass))
            history = conversation_history(chat_log.content, self.settings["history_messages"]) if self.settings["history_enabled"] else []
            result = await AgentEngine(self.settings, client, execute, trace=trace, history=history).process(
                user_input.text, user_input.language, snapshot,
            )
            text = result["answer"]
            if result.get("answer_error"):
                _LOGGER.warning("Smart-home answer generation failed: %s", result["answer_error"])
            _LOGGER.debug("route=%s service=%s timings_ms=%s usage=%s",
                          result["route"], result["service"], result["timings_ms"], result["usage"])
        except ModelError as error:
            _LOGGER.warning("Model request failed: %s", error)
            if not trace.failure_reported:
                trace.failure(model_failure(trace.stage, error))
            text = self.settings["error_reply"]
        except Exception:
            _LOGGER.exception("Conversation processing failed")
            trace.failure(failure_details(trace.stage, "An unexpected integration error stopped the request.",
                                          "Check the Home Assistant logs for the error details. Include them when reporting this issue."))
            text = self.settings["error_reply"]
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(agent_id=user_input.agent_id, content=text),
        )
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(text)
        return conversation.ConversationResult(
            response=response, conversation_id=chat_log.conversation_id, continue_conversation=False,
        )
