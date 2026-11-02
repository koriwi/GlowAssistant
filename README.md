# GlowAssistant

A Home Assistant conversation agent. Install it with HACS.

GlowAssistant answers state queries and calls Home Assistant actions. It extracts text and number parameters. Other parameter formats report “not yet supported.” The local CLI plans actions by default; `--execute` runs them.
It uses Home Assistant's exposed entities. You can exclude additional entities in the integration settings.

Minimum Home Assistant version: **2026.4.2**.

## Install

1. Add `https://github.com/koriwi/GlowAssistant` to **HACS → Custom repositories**. Select the **Integration** category.
2. Download **GlowAssistant**.
3. Restart Home Assistant.
4. Open **Settings → Devices & services → Add integration → GlowAssistant**.
5. Configure the **Decider**. Enable and configure **Smart-home answer** only if you want LLM replies.
6. Select GlowAssistant as the conversation agent under **Settings → Voice assistants**.

Use the Assist pipeline to configure speech recognition and text-to-speech.

## Configure the models

Set the endpoint URL, model ID, and API key for each model.
Use a native decision model, such as Jev, for **Decider**.
Use chat models for the optional answer and general sections.
Each model section also contains its prompts and limits.
Local servers without authentication do not need an API key.

| Provider | Decider endpoint | Answer and general endpoints |
| --- | --- | --- |
| OpenRouter | `https://openrouter.ai/api/alpha/decisions` | `https://openrouter.ai/api/v1/chat/completions` |
| llama.cpp | `http://SERVER:8321/v1/systemone` | `http://SERVER:8322/v1/chat/completions` |

Replace `SERVER` with the model server address.
Set the port for each server.
Use addresses that Home Assistant can reach.

Smart-home replies use no answer LLM by default. Commands reply `ok`. Yes/no queries use a fixed reply. Other queries list entity names and values; one decision selects the requested state property, such as brightness or temperature.
Enable **Smart-home answer → Enable LLM answer generation** to use the answer model.

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

## Run logs

Enable **Save run logs to disk** in the integration settings. Logs are off by default.
Files go to `<HA config>/glow_assistant/runs/successful/` or `unsuccessful/`.
Each JSON record includes the request, states, settings, model inputs/outputs, decisions and action results. Credentials are redacted. Files are not automatically deleted.

## Local testing

`local.py` uses the same engine as the integration. It reports decision times and prints the selected action's field metadata.
Add `--log-runs` to save local records under `runs/successful/` or `runs/unsuccessful/`. Actions run only with `--execute`.
See [local setup](local.md) for the integration update and `.env` settings. `benchmark.py` remains a compatibility entry point.

```bash
python local.py --query "Ist die Lampe im Büro an?"
python local.py --query "Wie hell ist die Lampe?" --llm-answer
python local.py --query "Was steht auf der Einkaufsliste?" --execute
```

## TODO

- [x] Add MCP server support for requests that do not concern the smart home.
- [x] Add optional conversation history. Resolve requests such as “a little bit more” and “now stop it”.
