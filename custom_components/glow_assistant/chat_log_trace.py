from homeassistant.components.conversation.chat_log import (
    AssistantContent,
    ToolResultContent,
)
from homeassistant.helpers.llm import ToolInput

STAGE_NAMES = {
    "service": "glow_assistant_select_service_and_domain",
    "area": "glow_assistant_select_area",
    "target": "glow_assistant_select_target",
    "numeric": "glow_assistant_extract_numeric_value",
    "execution": "glow_assistant_execute_action",
    "answer": "glow_assistant_generate_answer",
    "general": "glow_assistant_general_answer",
    "mcp": "glow_assistant_mcp_tool",
    "mcp_connect": "glow_assistant_connect_mcp_servers",
}


class ChatLogTrace:
    def __init__(self, chat_log, agent_id):
        self.chat_log = chat_log
        self.agent_id = agent_id
        self.calls = {}

    def __call__(self, stage, arguments, result=None):
        if result is None:
            call = ToolInput(tool_name=STAGE_NAMES[stage], tool_args=arguments, external=True)
            self.calls[stage] = call
            self.chat_log.async_add_assistant_content_without_tools(
                AssistantContent(agent_id=self.agent_id, tool_calls=[call]),
            )
        else:
            call = self.calls.pop(stage)
            self.chat_log.async_add_assistant_content_without_tools(ToolResultContent(
                agent_id=self.agent_id, tool_call_id=call.id,
                tool_name=call.tool_name, tool_result=result,
            ))
