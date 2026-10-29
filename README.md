# GlowAssistant

A Home Assistant conversation agent. Install it with HACS.

GlowAssistant supports state queries, actions that need only a target, brightness changes, and thermostat temperature changes.
It can add shopping or to-do items, send messages to notify entities, and announce messages on Assist satellites.
It uses Home Assistant's exposed entities. You can exclude additional entities in the integration settings.

Minimum Home Assistant version: **2026.4.2**.

## Install

1. Add `https://github.com/koriwi/GlowAssistant` to **HACS → Custom repositories**. Select the **Integration** category.
2. Download **GlowAssistant**.
3. Restart Home Assistant.
4. Open **Settings → Devices & services → Add integration → GlowAssistant**.
5. Configure the **Decider** and **Smart-home answer** models.
6. Select GlowAssistant as the conversation agent under **Settings → Voice assistants**.

Use the Assist pipeline to configure speech recognition and text-to-speech.

## Configure the models

Set the endpoint URL, model ID, and API key for each model.
Use a native decision model, such as Jev, for **Decider**.
Use chat models for the other sections.
Each model section also contains its prompts and limits.
Local servers without authentication do not need an API key.

| Provider | Decider endpoint | Answer and general endpoints |
| --- | --- | --- |
| OpenRouter | `https://openrouter.ai/api/alpha/decisions` | `https://openrouter.ai/api/v1/chat/completions` |
| llama.cpp | `http://SERVER:8321/v1/systemone` | `http://SERVER:8322/v1/chat/completions` |

Replace `SERVER` with the model server address.
Set the port for each server.
Use addresses that Home Assistant can reach.

To answer questions about other topics, enable the **General model**.
When you disable this model, these questions receive the configured **Unsure reply**.

## MCP tools

Add HTTP MCP servers under **General model → MCP servers**.
Enable each server that you want to use.
Set its URL and bearer token, if required.
Use a General model with tool-call support.
For stdio servers, supply your own HTTP proxy.

## Conversation history

Select **Enable conversation history** in the integration settings.
Set **Previous messages to include** to limit the rolling window.
All model steps receive these user inputs and final assistant replies.
History applies to the current Assist conversation. It has no additional idle expiry.

## Debug

Open **Settings → Voice assistants → your assistant → Debug**.
Expand a **Result for …** entry to see the choices, confidence, time, usage, and errors.

## TODO

- [x] Add MCP server support for requests that do not concern the smart home.
- [x] Add optional conversation history. Resolve requests such as “a little bit more” and “now stop it”.
