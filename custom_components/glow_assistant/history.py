import json

from .const import HISTORY_INSTRUCTIONS


def conversation_history(content, limit):
    messages = [
        {"role": message.role, "content": message.content}
        for message in content
        if message.role in ("user", "assistant")
        and isinstance(message.content, str) and message.content.strip()
        and not (message.role == "assistant" and getattr(message, "tool_calls", None))
    ]
    # HA already added the current user input
    if messages and messages[-1]["role"] == "user":
        messages.pop()
    return messages[-limit:]


def with_history(messages, history):
    if not history:
        return messages
    context = json.dumps(history, ensure_ascii=False)
    return [{**messages[0], "content": f"{messages[0]['content']}\n{HISTORY_INSTRUCTIONS}\nHistory: {context}"}, *messages[1:]]
