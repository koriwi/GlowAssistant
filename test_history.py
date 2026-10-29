import json
import unittest
from types import SimpleNamespace

from custom_components.glow_assistant.history import conversation_history, with_history


def message(role, content=None, **extra):
    return SimpleNamespace(role=role, content=content, **extra)


class HistoryTests(unittest.TestCase):
    def test_history_contains_only_prior_user_inputs_and_final_assistant_text(self):
        content = [
            message("system", "private system prompt"),
            message("user", "Make the office warmer"),
            message("assistant", None, thinking_content="private reasoning", tool_calls=["private tool"]),
            message("assistant", "Intermediate tool preface", tool_calls=["private tool"]),
            message("tool_result", "private tool result"),
            message("assistant", "The office target is now 22 degrees"),
            message("user", "a little bit more"),
        ]
        self.assertEqual(conversation_history(content, 10), [
            {"role": "user", "content": "Make the office warmer"},
            {"role": "assistant", "content": "The office target is now 22 degrees"},
        ])

    def test_rolling_window_counts_messages_without_idle_expiry(self):
        content = [message("user", "old question", created="2001-01-01"),
                   message("assistant", "old answer", created="2001-01-01"),
                   message("user", "new question"), message("assistant", "new answer"),
                   message("user", "current request")]
        self.assertEqual(len(conversation_history(content, 100)), 4)
        self.assertEqual(conversation_history(content, 3), [
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "new answer"},
        ])

    def test_repeated_user_text_is_not_removed_from_prior_turns(self):
        content = [message("user", "more"), message("assistant", "done"), message("user", "more")]
        self.assertEqual(conversation_history(content, 2)[0]["content"], "more")

    def test_history_context_preserves_chat_roles_with_an_odd_window_size(self):
        history = [{"role": "assistant", "content": "It is 330 metres tall."}]
        messages = [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "And its age?"}]
        result = with_history(messages, history)
        self.assertEqual([item["role"] for item in result], ["system", "user"])
        self.assertIn(json.dumps(history, ensure_ascii=False), result[0]["content"])
        self.assertEqual(result[-1], messages[-1])
        self.assertEqual(messages[0]["content"], "Answer briefly.")

    def test_empty_history_does_not_change_the_working_message_path(self):
        messages = [{"role": "user", "content": "Hello"}]
        self.assertIs(with_history(messages, []), messages)


if __name__ == "__main__":
    unittest.main()
