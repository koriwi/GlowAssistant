# Local benchmark

`benchmark.py` imports the addon's `AgentEngine` and `ModelClient` directly. Decisions, prompts, filtering and numeric/text extraction share the same implementation.

## Setup

1. Install this updated addon in HA and restart HA once to register the snapshot endpoint.
2. Install the local requirements: `python -m pip install -r requirements.txt`.
3. Set these values in the repository's `.env`:

```dotenv
HA_URL=https://your-home-assistant
HA_TOKEN=your-administrator-token
# Only required when multiple GlowAssistant entries are loaded
GLOW_ENTRY_ID=
```

The authenticated `GET /api/glow_assistant/snapshot` endpoint returns the exposed entities, available actions and integration settings. Exclusions apply. API keys and MCP bearer tokens are removed. It reads HA state without calling devices.

Set model keys with `DECIDER_API_KEY`, `ANSWER_API_KEY` and `GENERAL_API_KEY`. Non-empty `DECIDER_URL`/`MODEL`, `ANSWER_URL`/`MODEL` and `GENERAL_URL`/`MODEL` variables override the HA connections. Blank variables keep the exported values. `OPENROUTER_API_KEY` is a legacy fallback for the OpenRouter decider only.

## Run

```bash
python benchmark.py --query "Ist die Lampe im Büro an?" --language de
python benchmark.py --query "Mach das Licht im Büro an" --confidence 0.25
python benchmark.py --query "Sind im Haus Fenster offen?" --domain-filter-confidence 0.4 --target-filter-confidence 0.5 --json
```

`--confidence` overrides all decider thresholds. Individual flags override that value: `--service-confidence`, `--area-confidence`, `--target-scope-confidence`, `--domain-filter-confidence`, `--target-confidence`, `--target-filter-confidence`, `--numeric-confidence` and `--text-confidence`.

Match `--language` to the Assist pipeline. It defaults to `BENCHMARK_LANGUAGE` or `de`. Each run starts with no conversation history.

Output shows stage timings and decisions. `--json` includes the full engine result, trace and usage. Snapshot fetch, engine and total times are separate.

Device actions are marked `planned`. Parameter extraction and answer generation still run; device-command latency is excluded. General requests with enabled MCP servers stop before connecting or calling tools in plan-only mode.
