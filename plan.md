# GlowAssistant plan

## Goal

Build a Home Assistant conversation agent integration in small steps. Develop and evaluate the core logic locally first, without HA deployments or restarts.

## First milestone: target selection prototype

One Python script accepts a transcribed voice prompt, fetches entity context from HA's built-in MCP server, and asks Jev through OpenRouter which entity or area the user meant.

```bash
python prototype.py "turn on the desk lamp"
```

The prototype selects a target only. It does not execute device actions, record audio, or perform speech recognition.

## Agreed choices

- Use HA's **built-in MCP server**, despite a third-party server also being available.
- Endpoint: `https://assistant.gosewis.ch/api/mcp`.
- Authenticate with a **long-lived HA token**.
- Start with **Jev through OpenRouter**. Keep the model ID in one constant so another compatible decision model can be tried by changing that string.
- Start with all available entity and area candidates in one classification request.
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
```

## Implementation steps

Current status: the user verified live MCP inspection and supplied `inspect.txt`: 151 entities across 17 areas, no registry IDs, and two entries with identical names/domain/area. Entity and area classification is implemented and tested offline; live model behavior and accuracy evaluation remain pending.

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

### 3. Classify through OpenRouter

- [x] Implement a Decisions API request using Jev's `Choice` primitive.
- [x] Read the selected option, per-option probabilities, and confidence.
- [x] Validate that the selection belongs to the supplied choices.
- [ ] Verify a real OpenRouter request.

API failures and invalid responses are errors, not `other` selections.

### 4. Report and evaluate

- [x] Print a typed entity/area target or `other`, top alternatives, confidence, timing, and cost where reported.
- [x] Document setup and the CLI command.
- [ ] Try representative prompts: clear names, room-wide device groups, area aliases, STT mistakes, ambiguous targets, absent targets, and requests spanning multiple areas.
- [ ] Compare expected versus selected targets to judge whether this approach is useful.

## Done when

- A local CLI run fetches real HA context and returns a real Jev decision.
- Entity selections, area selections, and `other` can be inspected with their scores.
- Failures are visible, and no device actions are executed.

## Later

Use prototype findings to decide the next step toward a HA conversation agent. Action selection, device control, conversation history, and packaging as an integration remain outside this milestone.

## References

- [Home Assistant MCP server](https://www.home-assistant.io/integrations/mcp_server/)
- [Jev on OpenRouter](https://openrouter.ai/docs/guides/community/jev)
