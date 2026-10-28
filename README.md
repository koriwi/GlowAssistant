# GlowAssistant

Local prototype for a future Home Assistant conversation agent. See [plan.md](plan.md).

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

Fetches MCP tools and Assist context once per script run and keeps that snapshot in memory. The first OpenRouter request selects an advertised MCP tool or `query_state`, plus an entity domain. The second selects an entity or area from that domain, using the same snapshot. No cache is written to disk, and no device actions are executed.

Example summary:

```text
service: assist__HassTurnOff
domain: light
target type: area
target: Wohnzimmer
```

The `service` is the actual advertised MCP tool name, rather than an inferred HA service such as `light.turn_off`. If the tool or domain is `other`, target selection is skipped and the summary shows `none`. A target decision of `other` is displayed as `other`.

State questions such as "is the lamp on?" or "how warm is the living room?" can select `query_state`. This uses the already fetched states and attributes and still selects a domain and target. The summary shows `service: none (query state)`; JSON contains `service: null` and the `query_state` choice in the first decision. It remains distinct from an unsupported request (`other`). Single measurements should select their reporting entity; questions about a group can select an area. Answer generation is not implemented.

For full scores, alternatives, selected metadata, per-request usage/cost, and timings:

```bash
python prototype.py "Mach alle Lichter im Wohnzimmer aus" --json
```

Change `MODEL` in `prototype.py` to try another compatible decision model. The default is `typesafe/jev-1.14`.

Target choices get temporary IDs such as `entity_001` and `area_001`. These IDs belong to that classification request; they are not HA registry IDs. Area aliases are grouped with their canonical name. Entities outside the selected domain and areas without that domain are excluded.

Specific device requests select an entity. Room-wide requests such as "turn off the lights in the living room" select an area. `other` covers no match, unresolved ambiguity, or multiple targets that cannot be represented by one area. Identical entity entries remain separate candidates; the classifier is instructed to choose `other` when it cannot distinguish the target.

An area target identifies the room. The separately selected `domain`, such as `light`, preserves the device category for future execution. Tool argument extraction and execution are not implemented.

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

Tests cover parsing, service/state-query and target decisions, measurement metadata, domain filtering, one MCP fetch per run, error handling, and the CLI flow with mocked service calls. Real model accuracy still needs live prompt trials.
