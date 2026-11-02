import json
import logging
import traceback
from time import perf_counter

from homeassistant.components import conversation
from homeassistant.const import MATCH_ALL
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .actions import device_outcome
from .chat_log_trace import ChatLogTrace
from .client import ModelClient, ModelError
from .debug import failure_details, model_failure
from .engine import AgentEngine
from .ha_context import async_execute, async_snapshot, speaker_area
from .history import conversation_history
from .run_logs import RunRecord
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
        record = RunRecord("home_assistant", user_input.text, user_input.language, enabled=self.settings["log_runs"],
                           capture=self.settings["log_runs"], trace=trace)
        record.settings(self.settings)
        record.data.update(entry_id=self.entry.entry_id, conversation_id=chat_log.conversation_id,
                           agent_id=user_input.agent_id)
        engine = None
        result = None
        started = perf_counter()
        snapshot_ms = 0
        engine_started = None

        async def execute(action, arguments):
            return await async_execute(self.hass, set(self.entry.data.get("excluded_entities", [])), user_input.context, action, arguments)

        try:
            snapshot = await async_snapshot(self.hass, excluded)
            snapshot_ms = round((perf_counter() - started) * 1000)
            record.snapshot(snapshot)
            client = ModelClient(async_get_clientsession(self.hass), observer=record.model_call if record.capture_models else None)
            history = conversation_history(chat_log.content, self.settings["history_messages"]) if self.settings["history_enabled"] else []
            location = speaker_area(self.hass, getattr(user_input, "device_id", None))
            record.data.update(speaker_area=location, history=history)
            engine = AgentEngine(self.settings, client, trace=record.trace, history=history, execute=execute)
            engine_started = perf_counter()
            result = await engine.process(user_input.text, user_input.language, snapshot, speaker_area=location)
            text = result["answer"]
            if "action_metadata" in result:
                _LOGGER.debug("Selected action metadata: %s", json.dumps(result["action_metadata"], ensure_ascii=False, default=str))
                _LOGGER.info("Target scope: %s; speaker area: %s", result.get("target_scope"), result.get("speaker_area"))
                _LOGGER.debug("Parameter detection: %s; extracted values: %s",
                              result.get("parameter_selection"), result.get("parameters"))
            if result.get("answer_error"):
                _LOGGER.warning("Smart-home answer generation failed: %s", result["answer_error"])
            _LOGGER.debug("route=%s service=%s timings_ms=%s usage=%s",
                          result["route"], result["service"], result["timings_ms"], result["usage"])
        except ModelError as error:
            _LOGGER.warning("Model request failed: %s", error)
            partial = getattr(engine, "result", {})
            partial = partial if isinstance(partial, dict) else {}
            outcome = device_outcome(partial.get("execution"))
            text = self.settings["answer_error_reply" if outcome == "completed" else "error_reply"]
            result = {**partial, "request": user_input.text, "route": "error", "error": str(error), "answer": text}
            record.data["exception"] = {"type": type(error).__name__, "traceback": traceback.format_exc()}
            if not trace.failure_reported:
                details = {**model_failure(trace.stage, error), "device_command": outcome}
                record.trace("failure", details)
                record.trace("failure", {}, details)
        except Exception as error:
            _LOGGER.exception("Conversation processing failed")
            partial = getattr(engine, "result", {})
            partial = partial if isinstance(partial, dict) else {}
            outcome = device_outcome(partial.get("execution"))
            text = self.settings["answer_error_reply" if outcome == "completed" else "error_reply"]
            result = {**partial, "request": user_input.text, "route": "error", "error": str(error), "answer": text}
            record.data["exception"] = {"type": type(error).__name__, "traceback": traceback.format_exc()}
            details = failure_details(trace.stage, "An unexpected integration error stopped the request.",
                                      "Check the Home Assistant logs for the error details. Include them when reporting this issue.",
                                      device_command=outcome)
            record.trace("failure", details)
            record.trace("failure", {}, details)
        finally:
            if result is None:
                result = {**getattr(engine, "result", {}), "request": user_input.text, "error": "The run was interrupted."}
            report = await record.finish(result, self.hass.config.path("glow_assistant", "runs"), timings_ms={
                "snapshot": snapshot_ms if engine_started is not None else round((perf_counter() - started) * 1000),
                "engine": round((perf_counter() - engine_started) * 1000) if engine_started is not None else 0,
                "total": round((perf_counter() - started) * 1000),
            })
            if report.get("log_error"):
                _LOGGER.warning("Could not save run log: %s", report["log_error"])
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(agent_id=user_input.agent_id, content=text),
        )
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(text)
        return conversation.ConversationResult(
            response=response, conversation_id=chat_log.conversation_id, continue_conversation=False,
        )
