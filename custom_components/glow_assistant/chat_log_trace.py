from homeassistant.components.conversation.chat_log import (
    AssistantContent,
    ToolResultContent,
)
from homeassistant.helpers.llm import ToolInput

from .debug import failure_text

STAGE_NAMES = {
    "service": "glow_assistant_select_service_and_domain",
    "area": "glow_assistant_select_area",
    "target": "glow_assistant_select_target",
    "numeric": "glow_assistant_extract_numeric_value",
    "text": "glow_assistant_extract_text",
    "execution": "glow_assistant_execute_action",
    "answer": "glow_assistant_generate_answer",
    "general": "glow_assistant_general_answer",
    "mcp": "glow_assistant_mcp_tool",
    "mcp_connect": "glow_assistant_connect_mcp_servers",
    "failure": "glow_assistant_failure_summary",
}


class ChatLogTrace:
    def __init__(self, chat_log, agent_id):
        self.chat_log = chat_log
        self.agent_id = agent_id
        self.calls = {}
        self.stage = "context"
        self.failure_reported = False
        self.device_command = "not_sent"

    def failure(self, details):
        for stage in list(self.calls):
            self(stage, {}, {"status": "error", "error": details["reason"]})
        details = {"device_command": self.device_command, **details}
        self("failure", details)
        self("failure", {}, details)

    def __call__(self, stage, arguments, result=None):
        if result is None:
            if stage == "failure":
                self.failure_reported = True
            else:
                self.stage = stage
            if stage == "execution":
                self.device_command = "unconfirmed"
            call = ToolInput(tool_name=STAGE_NAMES[stage], tool_args=arguments, external=True)
            self.calls[stage] = call
            self.chat_log.async_add_assistant_content_without_tools(
                AssistantContent(agent_id=self.agent_id, tool_calls=[call],
                                 content=failure_text(arguments) if stage == "failure" else None),
            )
        else:
            if stage == "execution" and result.get("status") == "completed":
                self.device_command = "completed"
            call = self.calls.pop(stage)
            self.chat_log.async_add_assistant_content_without_tools(ToolResultContent(
                agent_id=self.agent_id, tool_call_id=call.id,
                tool_name=call.tool_name, tool_result=result,
            ))
