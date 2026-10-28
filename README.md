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

## Select an entity

```bash
python prototype.py "Mach die Stehlampe im Wohnzimmer an"
```

Fetches the current Assist context, turns every entry into a choice, adds `other`, then makes one OpenRouter Decisions request. Output is JSON with the selected metadata, probability, confidence, three alternatives, timings in milliseconds, and usage/cost where reported. It does not execute actions.

Change `MODEL` in `prototype.py` to try another compatible decision model. The default is `typesafe/jev-1.14`.

Each entry gets a temporary ID such as `entity_001`. These IDs belong to that context snapshot; they are not HA entity IDs. Names/aliases and area aliases are preserved as provided by HA. `other` covers no match, unresolved ambiguity, and requests targeting multiple entities. Identical entries remain separate candidates; the classifier is instructed to choose `other` when it cannot distinguish the target.

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
