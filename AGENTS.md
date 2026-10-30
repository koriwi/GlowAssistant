# Collaboration preferences

- Keep responses short, information-dense, and easy to scan. No walls of text.
- The user wants close control over scope and decisions. Do only the work explicitly requested; do not start unsolicited coding, refactors, or features.
- For broad or ambiguous requests, briefly outline the proposed scope and clarify consequential choices before implementing. Resolve routine details within agreed scope without repeated approval requests.
- Prefer simple, readable solutions. Avoid premature optimization, speculative abstractions, and unnecessary dependencies.
- Add comments only when the reason or behavior would otherwise be unclear. Keep them short; fragments are fine. Do not narrate obvious code.

# Repository

- HACS integration: `custom_components/glow_assistant`, domain `glow_assistant`, targeting the user's HA 2026.4.2. `hacs.json` is at the repository root. Repository URL: https://github.com/koriwi/GlowAssistant. Keep the HA minimum and test dependency aligned with this version; check with the user before raising it.
- HA 2026.4.2 uses `voluptuous` config schemas; do not use newer `probatio` or `vol.Secret` APIs. API-key fields use password text selectors.
- HA config/options UI has three expanded model sections (decider, smart-home answer, general), with prompts, connections, numeric settings and request limits kept in the relevant model section. General has an enable checkbox; disabled general requests use the fixed unsure reply. Prompt text is not templated; blank restores the default.
- Integration uses direct HA actions and registry/state snapshots, never HA MCP or a long-lived HA token. No domain classifier: actions filter by their service domain; state queries select from all exposed domains in the chosen area. Query target metadata includes entity domains and area domain lists. Area queries pass all member states with entity IDs and domains to the answer model. It uses HA's exposed-entity settings minus optional exclusions and rechecks exposure before execution. Area actions expand to explicit exposed entity IDs.
- Assist debugging uses native `ChatLog` external `ToolInput` calls and `ToolResultContent` entries (`chat_log_trace.py`), visible in the upper conversation area. Engine trace callbacks report stage choices, confidence/thresholds, timing, usage and failures without extra model/device calls. Do not require frontend changes for this display or include credentials/system prompts in trace metadata.
- Pure integration engine/client/settings tests run offline. `test_ha_integration.py` needs HA (Python 3.14.2+, `requirements-ha-test.txt`) and otherwise skips explicitly. GitHub Actions runs HA tests and hassfest. Do not claim those checks ran locally if HA could not be installed.
- The integration is self-contained; its numeric helper is an isolated copy of the prototype helper. Keep probe/debugging behavior out of the HA path.
- Free-text extraction is limited to `todo.add_item.item`, `notify.send_message.message`, and `assist_satellite.announce.message`. One decider request classifies each word as keep/drop; keep only confidence strictly above `text_confidence` (default 0.5), then join in original order. Match the working probe: instructions once in `state.task`, short word questions, and item-specific criteria. Item and message prompts and the threshold live under Decider. Keep task instructions out of debug records. Empty or invalid selections must not execute. Keep `text_probe.py` independent; no string casting or additional text actions.
- General-model MCP uses Streamable HTTP only (`mcp==1.26.0`, matching HA 2026.4.2). Users supply their own proxy for stdio. Servers, bearer tokens, allowlists, timeouts and tool limits belong under General model. Only the enabled general route connects; discovery is read-only, and no tool is automatically retried after a failure. Native debug records include connection, model rounds and tool calls.
- Optional history uses HA's existing `ChatLog`, filtered to prior user inputs and final assistant text. Forward the last N messages to every model step. No extra expiry, target-reuse mechanism, extra decider call, tool-result history or reasoning history. Use fresh HA state values for numeric execution. History is scoped to the current Assist conversation and defaults to disabled.
- Local Python 3.11+ prototype; eventual goal is a Home Assistant conversation agent integration.
- Install dependencies: `python -m pip install -r requirements.txt` in a virtual environment.
- Inspect built-in HA MCP: `python prototype.py --inspect`.
- Classify a prompt: `python prototype.py "your request"`.
- Classification fetches MCP tools/context once per run, then selects service/domain → area → compatible target. No disk cache.
- Area selection receives only prompt and area names/aliases. `no_area` retains all domain candidates; `unsure` stops target selection. Known areas filter target candidates by domain and canonical area name.
- Service/domain selection receives the user prompt, tool definitions and domain choices, without the entity catalog. Target and numeric stages use the cached entity data.
- Unqualified warmer/colder requests mean heating/AC (`climate`). Light color temperature requires an explicit lighting reference.
- Non-query commands execute target-only MCP tools and supported temperature/brightness setters automatically. Other extra values and ambiguous/unsupported targets are skipped. Never run live device actions as an incidental test.
- Numeric setters use a fourth decider request with Score. Temperature constants: 16–26 °C, 9 levels. Relative temperature changes use the existing thermostat target (`temperature`), not the measured `current_temperature`.
- Temperature execution currently uses `TEMPERATURE_STEP = 1.0`, matching the user's reported thermostat step. Send whole-degree integers and preserve the direction of subdegree relative changes; the isolated probe keeps raw estimates.
- Shared numeric request/decoding logic lives in `numeric_values.py`. The standalone probe remains available for isolated experiments.
- State questions can select `query_state`, using cached states/attributes with no service. Domain and target selection still run; `unsure` remains separate.
- State-query output displays cached values/attributes; area queries list matching entities in the selected domain. JSON includes `state_info`.
- Default output is a short summary including the area filter; `--json` includes all stages' scores, alternatives, usage and timings.
- Optional `tts_answer.py` generates spoken text after state lookup/execution via independent `ANSWER_URL`, `ANSWER_MODEL`, `ANSWER_API_KEY` chat-completions settings. Blank URL/model disables it. Send only final selections and selected state/execution facts, not the entity/tool catalog or classifier scores. Answer failures preserve the execution outcome and never retry the action.
- In the local prototype, selection fallback `unsure` returns exactly `ich bin mir nicht sicher` without a chat-model call. Only the first service question also offers `not-smarthome-related`, routing the original request directly to the smart chat model via independent `GENERAL_URL`, `GENERAL_MODEL`, `GENERAL_API_KEY`. Later unresolved choices and skipped operations stay `unsure`. Prototype general replies have no MCP/tools and bypass the small answer model.
- Run offline checks: `python -m unittest -v`.
- Isolated numeric experiment: `python numeric_probe.py "Set brightness to 55%" --expected 55`. See `numeric_probe.md`; keep probe CLI/debugging behavior separate from normal execution.
- Classifier choices use snapshot-local IDs for entities and areas, not HA registry IDs. Room-wide requests select areas; specific device requests select entities.
- `.env` contains credentials. Never print its values or commit it.
- Both commands use `DECIDER_URL` (full native decision endpoint), `DECIDER_MODEL`, and optional `DECIDER_API_KEY`. Defaults preserve OpenRouter; its legacy key is used only for the `openrouter.ai` host. llama.cpp uses `/v1/systemone`, not the chat completions API.
- Scope and implementation checklist: `plan.md`.
