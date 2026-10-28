# GlowAssistant plan

## Goal

Build a Home Assistant conversation agent integration in small steps. Develop and evaluate the core logic locally first, without HA deployments or restarts.

## First milestone: service and target selection prototype

One Python script accepts a transcribed voice prompt, fetches tools and entity context from HA's built-in MCP server, and asks Jev through OpenRouter which MCP tool/domain and entity or area the user meant.

```bash
python prototype.py "turn on the desk lamp"
```

The prototype selects a tool or state query, domain, and target. State queries display cached values; other requests execute target-only MCP tools. Audio capture and speech recognition remain outside this milestone.

## Agreed choices

- Use HA's **built-in MCP server**, despite a third-party server also being available.
- Endpoint: `https://assistant.gosewis.ch/api/mcp`.
- Authenticate with a **long-lived HA token**.
- Default to **Jev through OpenRouter**. Configure the full native decision endpoint, model ID, and optional key through `DECIDER_URL`, `DECIDER_MODEL`, and `DECIDER_API_KEY`; local llama.cpp can use `http://localhost:8321/v1/systemone`.
- Fetch MCP tools/context once per run and reuse that in-memory snapshot for both classification stages. No disk cache.
- First select an advertised MCP tool and entity domain in one OpenRouter request. Then select a compatible entity or area in a second request.
- Include `query_state` for current-state/measurement questions. It needs no service and reuses cached states/attributes, while still selecting a domain and target.
- Display selected entity states/attributes with available units for `query_state`; area queries list matching entities in the selected domain.
- Execute target-only tools and supported temperature/brightness setters. Numeric setters use a third Jev Score request. Other extra values are skipped; no automatic retries.
- Temperature range/levels are constants: 16–26 °C, 9 levels. Relative changes use the existing thermostat target setting.
- Temperature execution uses a 1 °C step constant based on the user's device metadata; raw fractional estimates are rounded before MCP execution.
- Report the actual MCP tool name, such as `assist__HassTurnOff`, plus domain, rather than mapping it to an underlying HA service name.
- Include an explicit `other` option.
- Select context entries using temporary choice IDs; exact HA entity IDs are not required for this prototype.
- Include one choice per distinct area, grouping its aliases. Select an area for room-wide requests and an entity for specific device requests.
- Keep the implementation simple: one script, no agent framework or speculative abstractions.

## Local configuration

Load credentials from a gitignored `.env` file. Provide a tracked `.env.example` with placeholders.

```dotenv
HA_MCP_URL=https://assistant.gosewis.ch/api/mcp
HA_TOKEN=...
OPENROUTER_API_KEY=...
DECIDER_URL=https://openrouter.ai/api/alpha/decisions
DECIDER_MODEL=typesafe/jev-1.14
DECIDER_API_KEY=
```

The generic key is optional for unauthenticated local servers. The existing OpenRouter key is used only when the configured endpoint host is `openrouter.ai`. Both commands share these settings.

## Implementation steps

Current status: the user verified live MCP inspection and supplied `inspect.txt`: 151 entities across 17 areas, no registry IDs, and two entries with identical names/domain/area. Service/domain selection followed by compatible entity/area selection is implemented and tested offline. Live model behavior and accuracy evaluation remain pending.

### 1. Inspect MCP entity context

- [x] Connect and discover available read-only tools/resources.
- [x] Retrieve the entity context exposed by the built-in server.
- [x] Inspect the actual response for entity IDs, friendly names, domains, areas, and aliases.

Do not assume MCP exposes every HA entity or every metadata field. If the response lacks enough information to identify entities reliably, document the gap and agree on the next step before changing data sources.

### 2. Prepare choices

- [x] Create one option per available entity using its identifying metadata.
- [x] Add one option per area with canonical name, aliases, and available domains.
- [x] Preserve a deterministic mapping from each option to its target within the snapshot.
- [x] Add `other`: no single entity or area target can represent the request.

For this baseline, `other` includes no match, unresolved ambiguity, and multiple targets that cannot be represented by one area. Room-wide requests such as "turn off the lights in the living room" select the area instead of one lamp or `other`. Use available metadata; do not invent missing aliases or locations.

### 3. Classify the service and domain

- [x] Build tool choices from the cached MCP discovery response and domain choices from entity context.
- [x] Add a distinct `query_state` choice for information available in cached states/attributes.
- [x] Ask Jev for the MCP tool and domain in the first request.
- [x] Validate both selections, including explicit tool domain restrictions where provided.
- [x] Skip target selection when tool or domain is `other`.
- [x] Continue domain/target selection for `query_state` and report no service.

### 4. Classify the target

- [x] Implement a Decisions API request using Jev's `Choice` primitive.
- [x] Filter entity and area choices to the selected domain and pass the selected tool/domain as context.
- [x] Read the selected option, per-option probabilities, and confidence.
- [x] Validate that the selection belongs to the supplied choices.
- [ ] Verify a real OpenRouter request.

API failures and invalid responses are errors, not `other` selections.

### 5. Extract supported numeric values and execute tools

- [x] Share numeric request/decoding logic with the standalone probe.
- [x] Extract temperature and brightness using a third Score request and distinguish absolute/relative/unsupported requests.
- [x] Use cached thermostat target settings and brightness percentages as baselines.
- [x] Quantize temperature to the configured device step and preserve the direction of substep relative changes.
- [x] Require a common baseline for relative area changes; allow absolute area settings with differing baselines.
- [x] Send selected name/area and supported domain filter to the advertised MCP tool.
- [x] Reuse the original MCP session and cached discovery/context data for execution.
- [x] Skip queries, ambiguous/unsupported targets, duplicate entity names in scope, and unsupported additional values.
- [x] Display execution status and actual arguments; include the complete MCP response in JSON.
- [x] Report MCP/transport errors without retries.
- [ ] Verify a live device command.

### 6. Report and evaluate

- [x] Print a short service/domain/target summary.
- [x] Display cached values for state queries, including matching area members, measurements and available units.
- [x] Show unknown/unavailable states explicitly and suppress stale attributes in the summary.
- [x] Provide `--json` for both stages' selections, alternatives, scores, usage/cost, and timings.
- [x] Document setup and the CLI command.
- [ ] Try representative prompts: clear names, room-wide device groups, area aliases, STT mistakes, ambiguous targets, absent targets, and requests spanning multiple areas.
- [ ] Check live state-query routing for on/off questions and temperature measurements.
- [ ] Compare expected versus selected targets to judge whether this approach is useful.

## Done when

- A local CLI run fetches real HA tools/context once and returns real Jev service/domain and target decisions.
- Entity selections, area selections, and `other` can be inspected with their scores.
- Failures are visible, query-state requests do not execute device actions, and supported target-only and numeric setters can execute through MCP.

## Later

Use prototype findings to decide the next step toward a HA conversation agent. Natural-language responses, additional argument types beyond temperature/brightness, conversation history, and packaging as an integration remain outside this milestone.

## References

- [Home Assistant MCP server](https://www.home-assistant.io/integrations/mcp_server/)
- [Jev on OpenRouter](https://openrouter.ai/docs/guides/community/jev)
