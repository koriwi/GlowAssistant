# GlowAssistant

Local prototype for a future Home Assistant conversation agent. See [plan.md](plan.md).

For isolated brightness/temperature extraction tests, see [numeric_probe.md](numeric_probe.md). That command does not execute HA tools or change the agent path.

## Setup

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` if it does not already exist, then fill in the HA token and OpenRouter key. Existing environment variables override `.env`.

## Select a service and target

```bash
python prototype.py "Mach die Stehlampe im Wohnzimmer an"
python prototype.py "Mach alle Lichter im Wohnzimmer aus"
python prototype.py "Wie warm ist es im Wohnzimmer?"
```

Fetches MCP tools and Assist context once per script run and keeps that snapshot in memory. The first OpenRouter request selects an advertised MCP tool or `query_state`, plus an entity domain. The second selects an entity or area from that domain, using the same snapshot. Non-query commands now execute the selected target-only tool through MCP. No cache is written to disk.

Example summary:

```text
service: assist__HassTurnOff
domain: light
target type: area
target: Wohnzimmer
execution: completed
arguments: {"domain": ["light"], "area": "Wohnzimmer"}
```

The `service` is the actual advertised MCP tool name, rather than an inferred HA service such as `light.turn_off`. If the tool or domain is `other`, target selection is skipped and the summary shows `none`. A target decision of `other` is displayed as `other`.

Execution supports tools whose parameters are limited to target fields (`name`, `area`, `floor`, `domain`, `device_class`). Entity calls use a known name/alias, the area where available, and a domain filter where supported. Area calls send the canonical area name and supported domain filter. Snapshot-local choice IDs are never sent as HA entity IDs. Duplicate entity names within the available scope are skipped.

Tools needing additional parameters, such as brightness, temperature, or a list item, are skipped for now. Missing/ambiguous targets and unsupported target schemas are also skipped with a reason. Calls are not automatically retried. `completed` means MCP reported completion; it does not independently verify the resulting device state. MCP errors and transport failures produce a nonzero exit code, and full tool responses are included in `--json` output.

State questions such as "is the lamp on?" or "how warm is the living room?" can select `query_state`. This uses the already fetched states and attributes and still selects a domain and target. The summary shows `service: none (query state)` and the selected entity's state and attributes, including units where provided. For example:

```text
service: none (query state)
domain: sensor
target type: entity
target: Temperatur
state: 21.5 °C
```

Area queries list the states of each entity in the selected area and domain. Climate entities display their state plus measurements such as `current_temperature` and `temperature`. Unknown/unavailable states are shown explicitly; the summary suppresses their potentially stale attributes. If no target is identified, it reports that no state data is available. Values come directly from the cached context without another MCP or model request.

JSON contains `service: null`, the `query_state` choice in the first decision, and a `state_info` list with the matching entries and their raw state/attribute data. `query_state` remains distinct from an unsupported request (`other`). Single measurements should select their reporting entity; questions about a group can select an area.

For full scores, alternatives, selected metadata, per-request usage/cost, and timings:

```bash
python prototype.py "Mach alle Lichter im Wohnzimmer aus" --json
```

Change `MODEL` in `prototype.py` to try another compatible decision model. The default is `typesafe/jev-1.14`.

Target choices get temporary IDs such as `entity_001` and `area_001`. These IDs belong to that classification request; they are not HA registry IDs. Area aliases are grouped with their canonical name. Entities outside the selected domain and areas without that domain are excluded.

Specific device requests select an entity. Room-wide requests such as "turn off the lights in the living room" select an area. `other` covers no match, unresolved ambiguity, or multiple targets that cannot be represented by one area. Identical entity entries remain separate candidates; the classifier is instructed to choose `other` when it cannot distinguish the target.

An area target identifies the room. The separately selected `domain`, such as `light`, limits the call to that category when the tool supports a domain parameter. Extraction of additional tool arguments is not implemented.

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
