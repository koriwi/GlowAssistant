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

## Select an entity or area

```bash
python prototype.py "Mach die Stehlampe im Wohnzimmer an"
python prototype.py "Mach alle Lichter im Wohnzimmer aus"
```

Fetches the current Assist context, adds choices for every entity and distinct area plus `other`, then makes one OpenRouter Decisions request. Output is JSON with a typed `target` (`entity` or `area`), probability, confidence, three alternatives, timings in milliseconds, and usage/cost where reported. `other` returns `target: null`. It does not execute actions.

Change `MODEL` in `prototype.py` to try another compatible decision model. The default is `typesafe/jev-1.14`.

Choices get temporary IDs such as `entity_001` and `area_001`. These IDs belong to that context snapshot; they are not HA registry IDs. Area aliases are grouped with their canonical name, and each area lists the entity domains present in the context.

Specific device requests select an entity. Room-wide requests such as "turn off the lights in the living room" select an area. `other` covers no match, unresolved ambiguity, or multiple targets that cannot be represented by one area. Identical entity entries remain separate candidates; the classifier is instructed to choose `other` when it cannot distinguish the target.

An area target identifies the room. Its `domains` field lists available categories; it is not an action filter. Future execution must also carry over the requested category, such as `light`, to address only that category in the room.

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

Tests cover parsing, choice mapping, error handling, and the CLI flow with mocked service calls. Real model accuracy still needs live prompt trials.
