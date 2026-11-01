# Local benchmark

`benchmark.py` imports the addon's `AgentEngine` and `ModelClient` directly. Decisions, prompts and state-query filtering share the same implementation.

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
python benchmark.py --query "Sind im Haus Fenster offen?" --domain-confidence 0.4 --target-filter-confidence 0.5 --json
```

`--confidence` overrides all decider thresholds. Individual flags override that value: `--service-confidence`, `--area-confidence`, `--target-scope-confidence`, `--domain-confidence`, `--query-type-confidence`, `--binary-answer-confidence`, `--target-confidence`, `--target-filter-confidence`, `--numeric-confidence` and `--text-confidence`.

Match `--language` to the Assist pipeline. It defaults to `BENCHMARK_LANGUAGE` or `de`. Each run starts with no conversation history.

State queries follow service → area → single-domain selection → answer-type selection. Yes/no queries ask the decider for yes/no/unsure from all candidate states, without entity keep/drop. The answer model only translates yes/no into the Assist language. Other queries use entity keep/drop → answer model. Unsure decisions stop with the configured unsure reply. Domain selection runs for rooms and `all_areas`; actions get their domain from the service. `--domain-filter-confidence` remains an alias for `--domain-confidence`.

Output shows stage timings and decisions. `--json` includes the full engine result, trace and usage. Snapshot fetch, engine and total times are separate.

Add `--show-answer-input` to print the smart-home answer request: model, messages, temperature and output token limit. Messages include the system prompt, pipeline language, history and final state facts. For detailed queries, selected entities appear in one `entities` array, with each entity's ID, name, room and query state together. For yes/no replies, only `state_answer` is sent for translation. Nested JSON in `content` is expanded for display; the chat API still receives text. With `--json`, this readable view is available under `answer_input`, and the original messages remain in `trace`. API keys and headers are excluded.

Action requests select the service, then use one Choice question to select optional parameters. The console prints the selected action's complete HA metadata, including its target selector, fields, required flags, field selectors, descriptions and examples when supplied by HA. Fields are optional unless `required` is true. `--json` includes this under `result.action_metadata` and the service trace result. Optional parameter selection uses every optional HA field (including fields in nested UI groups) plus a `none` choice. Required fields are listed separately. Each actual field with probability strictly above `parameter_probability_threshold` is retained; overall confidence is not used, and `none` never overrides retained fields. The default threshold is 0.15; set it under Decider or override it with `--parameter-threshold 0.2`. `--confidence` does not change this probability threshold. With no optional fields, the extra call is skipped. Console output lists every option probability and the retained fields; JSON includes `parameter_selection` and `required_parameters`. No area/target selection, value extraction, device execution or answer-model call follows. The existing supported action choices remain in use. General requests with enabled MCP servers stop before connecting or calling tools in plan-only mode.

This is an experimental use of Choice: [its probabilities sum to one](https://docs.typesafe.ai/primitives/choice), so the scores compete with one another rather than representing independent inclusion probabilities.
