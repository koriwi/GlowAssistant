# GlowAssistant

Home Assistant conversation agent, installable as a HACS custom integration. The local MCP prototype remains available for development. See [plan.md](plan.md).

## Home Assistant integration

Targets **Home Assistant 2026.4.2**; HACS minimum and HA test dependencies are pinned to that version. The integration is in `custom_components/glow_assistant` and uses HA's direct actions and state/area/entity registries. It does not need an HA token or HA MCP endpoint.

Install from a checkout by copying `custom_components/glow_assistant` into your HA configuration's `custom_components` directory, then restart HA. Once these files are on GitHub, you can instead add `https://github.com/koriwi/GlowAssistant` under **HACS → Custom repositories**, category **Integration**, download it, and restart HA.

1. Open **Settings → Devices & services → Add integration → GlowAssistant**.
2. Fill in the **Decider** and **Smart-home answer** models. Each block contains its endpoint, model ID, API key, prompts, and limits. Model-only connection tests are optional checkboxes in those same blocks.
3. Optionally enable and configure the **General model**. When disabled, unrelated requests return `ich bin mir nicht sicher`.
4. Choose the new conversation entity as the conversation agent for an Assist pipeline under **Settings → Voice assistants**. STT/TTS stay configured in the Assist pipeline.

Reopen **Configure** on the integration to edit the same grouped settings. Saving updates the existing entry and reloads the integration; a full HA restart is not required for those changes.

Endpoint examples:

| Model block | OpenRouter | llama.cpp |
|---|---|---|
| Decider | `https://openrouter.ai/api/alpha/decisions` | `http://MODEL-SERVER:8321/v1/systemone` |
| Smart-home answer / General | `https://openrouter.ai/api/v1/chat/completions` | `http://MODEL-SERVER:8322/v1/chat/completions` |

Use the actual host, port, and model ID for each server. **`localhost` means the machine/container running Home Assistant**, which may differ from your development machine. Each model has its own API-key field; the integration does not read the prototype's `.env` or inherit keys between models. Local unauthenticated servers can leave the key blank.

Prompts are editable plain instructions. Request/context data is injected separately, with no Jinja or other templating. Clear a prompt field to restore its built-in default. Choice IDs and response validation remain code-managed. Confidence thresholds default to zero, preserving the prototype's behavior until configured.

The decider uses **registered HA action names**, such as `light.turn_off`. The action determines its domain; the domain question is used for state queries. Brightness requests use a separate classifier candidate for `light.turn_on` with `brightness_pct`, and climate requests use `climate.set_temperature`. Execution currently supports target-only entity actions plus these two numeric parameters; other required parameter types are not advertised.

HA's existing exposed-entity settings determine which entities are available, with an additional optional exclusion list. Area selection filters the context before target selection. Room-wide calls expand into explicit exposed, non-excluded entity IDs, and exposure is checked again before execution. This avoids affecting hidden or excluded devices in the same room. The original HA conversation context is attached to direct action calls.

Temperature defaults remain **16–26, 9 anchors, whole-degree execution**, with the existing thermostat target as the relative baseline. Limits use HA's temperature unit; installations using Fahrenheit should adjust the defaults. Device min/max and step attributes further constrain temperature execution. Relative room-wide changes require a common known baseline; unresolved requests return the fixed unsure reply without a chat-model call.

Input limits count **characters**, including the request instructions and candidate metadata; they are not estimated token counts. Oversized requests are rejected rather than silently truncated. Chat output limits count tokens. Model calls and actions are never automatically retried. If answer generation fails after an action completes, the separate configurable reply states that the command was sent but its answer could not be generated.

MCP tools for the general model, conversation memory, color control, and extraction of other action parameters remain future work. HA maintains the conversation transcript, but previous turns are not yet sent to the models.

## Integration development checks

Offline checks, including the unchanged prototype suite:

```bash
python -m unittest -v
```

To also run the HA API/config-flow tests, use Python 3.14.2+ and install `requirements-ha-test.txt` alongside `requirements.txt`. HA-specific tests are explicitly skipped when HA is unavailable. GitHub Actions installs HA, runs the tests, and runs hassfest. No tests execute real device actions or contact model servers.

## Local prototype

For isolated brightness/temperature extraction tests, see [numeric_probe.md](numeric_probe.md). That command does not execute HA tools or change the agent path.

## Setup

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` if it does not already exist, then fill in the HA token and decider configuration. Existing environment variables override `.env`.

## Decider server

Both `prototype.py` and `numeric_probe.py` use the same settings. `DECIDER_URL` is the full endpoint URL; no path is appended automatically.

For your local llama.cpp server:

```dotenv
DECIDER_URL=http://localhost:8321/v1/systemone
DECIDER_MODEL=your-local-model-id
DECIDER_API_KEY=
```

Find the model ID with `curl http://localhost:8321/v1/models`. Local servers without authentication do not need a key.

For OpenRouter:

```dotenv
DECIDER_URL=https://openrouter.ai/api/alpha/decisions
DECIDER_MODEL=typesafe/jev-1.14
OPENROUTER_API_KEY=your-openrouter-key
```

`DECIDER_API_KEY` supports other authenticated providers and takes precedence over `OPENROUTER_API_KEY`. The legacy key is used only for the `openrouter.ai` host. Omitting the new settings preserves the existing OpenRouter URL and model.

The server must support the native `state`/`questions` → `answers` decision protocol. llama.cpp provides this at [`/v1/systemone`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api) for native decision models. An OpenAI-compatible `/v1/chat/completions` endpoint alone does not implement this protocol.

## Uncertain and general requests

The service decision distinguishes two fallback choices:

- `unsure`: ambiguous or unsupported smart-home request. Returns exactly `ich bin mir nicht sicher`, with no answer-model or device call. This also applies to unresolved domain/area/target choices and skipped numeric operations, even without a configured answer endpoint.
- `not-smarthome-related`: general knowledge or conversation, such as `wie hoch ist der eiffelturm?`. Skips area/target selection and device execution, and sends the original request to a separately configured smart model.

Configure the general model independently, using a full chat-completions endpoint:

```dotenv
GENERAL_URL=https://openrouter.ai/api/v1/chat/completions
GENERAL_MODEL=your-provider/smart-model-id
GENERAL_API_KEY=your-openrouter-key
```

For local llama.cpp, use your server's URL, for example `http://localhost:8323/v1/chat/completions`, its model ID, and an optional key. `OPENROUTER_API_KEY` is a fallback only for the `openrouter.ai` host. An unconfigured general endpoint produces an explicit `answer_error` for unrelated requests; it is not required for smart-home requests.

General replies are generated directly as concise, speech-ready text in the user's language. They bypass the small `ANSWER_MODEL` and have no tools or MCP access in this version. JSON includes `route` (`smarthome`, `unsure`, or `not-smarthome-related`); general requests add `usage.general` and `timings_ms.general`. Mixed smart-home/general requests and unavailable smart-home operations use `unsure`.

## Spoken answers

Optionally generate a short reply for TTS after a resolved smart-home state lookup or command execution. This uses a separate chat model; it does not synthesize audio. Leave `ANSWER_URL` and `ANSWER_MODEL` blank to disable this stage; fixed `unsure` replies and general-model routing still apply.

For a local llama.cpp server running a text/chat model:

```dotenv
ANSWER_URL=http://localhost:8322/v1/chat/completions
ANSWER_MODEL=your-local-chat-model-id
ANSWER_API_KEY=
```

Use your answer server's port. This can be independent of the decider server on port 8321. For OpenRouter:

```dotenv
ANSWER_URL=https://openrouter.ai/api/v1/chat/completions
ANSWER_MODEL=your-provider/model-id
ANSWER_API_KEY=your-openrouter-key
```

These full endpoint URLs use the chat-completions format supported by [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) and [OpenRouter](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion). `ANSWER_API_KEY` is optional for unauthenticated servers; `OPENROUTER_API_KEY` is a fallback only for the `openrouter.ai` host. The decider's model and key are independent.

Run `prototype.py` as usual. Output adds `answer: …`; `--json` adds `answer`, `usage.answer`, and `timings_ms.answer`. The model receives the original request, final service/domain/area/target, and selected state data or execution status/arguments/errors. It receives no entity/tool catalog or classifier scores. Instructions request one or two sentences in the user's language and distinguish completed, skipped, and failed commands.

Configuration is validated before contacting HA. Generation failures produce `answer_error` and a nonzero exit code while preserving the device execution result. No action or answer request is automatically retried; do not rerun a relative action just to regenerate its reply.

## Select a service and target

```bash
python prototype.py "Mach die Stehlampe im Wohnzimmer an"
python prototype.py "Mach alle Lichter im Wohnzimmer aus"
python prototype.py "Wie warm ist es im Wohnzimmer?"
python prototype.py "Mach es etwas wärmer im Wohnzimmer"
python prototype.py "Stell die Stehlampe im Wohnzimmer auf 60 Prozent"
```

Fetches MCP tools and Assist context once per script run and keeps that snapshot in memory. The decider calls run in this order:

1. Select an advertised MCP tool or `query_state`, plus an entity domain.
2. Select an area from names/aliases extracted from the cached context.
3. Select an entity or area target from candidates filtered by both domain and selected area.
4. For temperature/brightness setters, extract a numeric value before MCP execution.

The area call receives the prompt and area names/aliases only. `no_area` keeps all candidates from the selected domain when the request identifies no specific room. `unsure` represents unknown, ambiguous, or multiple requested areas and stops target selection. A known area with no entities in the selected domain also stops selection. No cache is written to disk.

The service/domain request sends only the user prompt, advertised tool definitions, and available domain choices. The entity catalog is kept locally for the subsequent target and numeric stages.

Unqualified warmer/colder requests default to heating/AC (`climate`). Lighting temperature is selected only when the prompt explicitly refers to lighting, a lamp, color temperature, warm/cold white, or Kelvin.

Example summary:

```text
service: assist__HassTurnOff
domain: light
area: Wohnzimmer
target type: area
target: Wohnzimmer
execution: completed
arguments: {"domain": ["light"], "area": "Wohnzimmer"}
```

The `service` is the actual advertised MCP tool name, rather than an inferred HA service such as `light.turn_off`. If the tool or domain is `unsure`, target selection is skipped and the target summary shows `none`. A target decision of `unsure` is displayed as `unsure`. `not-smarthome-related` is an additional service choice, used only to route unrelated requests.

Execution supports tools whose parameters are limited to target fields (`name`, `area`, `floor`, `domain`, `device_class`). Entity calls use a known name/alias, the area where available, and a domain filter where supported. Area calls send the canonical area name and supported domain filter. Snapshot-local choice IDs are never sent as HA entity IDs. Duplicate entity names within the available scope are skipped.

Missing/ambiguous targets, unsupported target schemas, and other additional parameters such as a list item are skipped with a reason. Calls are not automatically retried. `completed` means MCP reported completion; it does not independently verify the resulting device state. MCP errors and transport failures produce a nonzero exit code, and full tool responses are included in `--json` output.

## Numeric settings

Supported setters are `HassClimateSetTemperature` for room temperature and `HassLightSet` for brightness. The fourth request uses the same Score prompt/scaling as the standalone probe and checks whether the request is absolute, relative, or unsupported. Its mapped value is passed to MCP and printed in the summary; `--json` includes numeric scores, baseline, usage, and timing.

Constants in `prototype.py`:

| Parameter | Min | Max | Levels | Unit | Execution step |
| --- | --- | --- | --- | --- | --- |
| Temperature | 16 | 26 | 9 | °C | 1 °C |
| Brightness | 0 | 100 | 6 | % | 1% |

Relative temperature changes start from the thermostat's existing target (`temperature`). The prompt interprets vague warmer/cooler requests as ±0.5 °C. Brightness prefers `brightness_pct` when available, otherwise converts raw HA brightness from 0–255 to percent; off lights use zero. The MCP brightness argument is rounded to an integer.

`TEMPERATURE_STEP = 1.0` matches the step reported for the user's thermostat. The main path rounds temperature estimates to whole degrees and sends integers, so `21.025` becomes `21`. Relative changes smaller than one step move at least one step in the inferred direction instead of rounding back to the existing setting. A relative estimate with no change is skipped. Values are checked against the configured range after rounding. Raw numeric estimates remain available under `decisions.numeric.estimate` in JSON; the standalone probe retains its unrounded output.

Relative area changes require a known common setting across member devices; differing or missing values are skipped. Absolute area settings can apply a single new value across those devices. Light color and color-temperature changes remain unsupported. Numeric extraction is approximate: the configured range and tool schema constrain the output, and the model is instructed to classify requests outside that range as unsupported.

## State queries

State questions such as "is the lamp on?" or "how warm is the living room?" can select `query_state`. This uses the already fetched states and attributes and still selects a domain and target. The summary shows `service: none (query state)` and the selected entity's state and attributes, including units where provided. For example:

```text
service: none (query state)
domain: sensor
area: Wohnzimmer
target type: entity
target: Temperatur
state: 21.5 °C
```

Area queries list the states of each entity in the selected area and domain. Climate entities display their state plus measurements such as `current_temperature` and `temperature`. Unknown/unavailable states are shown explicitly; the summary suppresses their potentially stale attributes. If no target is identified, it reports that no state data is available. Values come directly from the cached context without another MCP or model request.

JSON contains `service: null`, the `query_state` choice in the first decision, and a `state_info` list with the matching entries and their raw state/attribute data. `query_state` remains distinct from an unsupported request (`unsure`). Single measurements should select their reporting entity; questions about a group can select an area.

For full scores, alternatives, selected metadata, per-request usage/cost, and timings:

```bash
python prototype.py "Mach alle Lichter im Wohnzimmer aus" --json
```

Set `DECIDER_MODEL` to try another compatible decision model. The fallback is `MODEL` in `prototype.py`, currently `typesafe/jev-1.14`.

Target choices get temporary IDs such as `entity_001` and `area_001`. These IDs belong to that classification request; they are not HA registry IDs. Area aliases are grouped with their canonical name. Entities outside the selected domain or selected area are excluded; requests with `no_area` retain the full domain scope. The final target choice still distinguishes a specific device from a room-wide target.

Specific device requests select an entity. Room-wide requests such as "turn off the lights in the living room" select an area. `unsure` covers no match, unresolved ambiguity, or multiple targets that cannot be represented by one area. Identical entity entries remain separate candidates; the classifier is instructed to choose `unsure` when it cannot distinguish the target.

An area target identifies the room. The separately selected `domain`, such as `light`, limits the call to that category when the tool supports a domain parameter.

For the API format, see [OpenRouter's Jev tutorial](https://openrouter.ai/docs/guides/community/jev-tutorial).

## Inspect the built-in MCP server

```bash
python prototype.py --inspect
```

Prints available tools/resources and the raw Assist context. Only the known Assist context resource or `GetLiveContext` tool is read; no device actions or OpenRouter requests run.

To save the response locally:

```bash
python prototype.py --inspect > inspection.json
```

`inspection.json` is gitignored because it contains home metadata.

## Verification

```bash
python -m unittest -v
```

Tests cover parsing, service/state-query and target decisions, measurements, domain filtering, one MCP context fetch per run, target argument construction, execution/error handling, and the CLI flow with mocked service calls. Tests do not operate real devices. Real model accuracy and device execution still need live trials.
