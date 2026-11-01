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

Action requests run **service → target type → area resolution/filtering → target resolution → parameter detection → stop**. Target types:

| Choice | Meaning |
| --- | --- |
| `entities` | Specific arealess entities: room-independent targets such as shopping lists or named timers |
| `entities_implied_area` | Specific entities within one implied speaker area |
| `entities_explicit_area` | Specific entities within exactly one explicit area |
| `areas` | One or more explicit areas, without specific entities |
| `implied_area` | Implied speaker area only, without specific entities |
| `unsure` | Unclear target type or an unsupported combination |

Specific entity targets tied to multiple explicit areas return unsure. Arealess means room-independent, not missing HA area assignments or an omitted room name. A globally named lamp is not automatically arealess. Room-independent logical targets do not inherit a speaker-area restriction automatically. Local roles such as desk/ceiling lamps can imply the supplied speaker area.

Target resolution follows the selected type:

| Target type | Area step | Final targets |
| --- | --- | --- |
| `entities` | No area filter; filter by action domain only | One Choice question selects one or more entities |
| `entities_implied_area` | Resolve the supplied speaker area programmatically | Filter by domain and area, then one entity Choice question |
| `entities_explicit_area` | One area Choice question; use the unique highest probability | Filter by domain and that area, then one entity Choice question |
| `areas` | One Choice question selects one or more explicit areas | Selected areas directly; no entity question |
| `implied_area` | Resolve the supplied speaker area programmatically | That area directly; no area or entity question |

Area candidates use exposed snapshot rooms and aliases. Arealess entity selection includes domain-matching entities with or without HA room assignments. Empty, invalid or ambiguous selections stop; the engine never falls back to other rooms or all entities. A supplied speaker area must resolve uniquely in the snapshot.

Multi-area and entity selection retain every real option strictly above its probability threshold; overall confidence is not used. `none` never overrides retained real options. Single-area selection ignores the multi-area threshold and uses only the unique highest probability; a tie or highest `none` stops. Prompts and probability thresholds are under Decider. Override the thresholds with `--action-area-threshold` and `--action-entity-threshold` (both default to 0.15). These are independent of `--confidence`. Full scores remain under `--json`; Choice probabilities share one distribution, so broad selections may need a lower threshold.

After successful target resolution, parameter detection runs against the selected action's HA field metadata. Required fields are listed automatically. Optional fields use one Choice question plus `none`; every real field strictly above `parameter_probability_threshold` (default 0.15) is kept. Overall confidence is not a filter, and `none` never overrides retained real fields. `--parameter-threshold` overrides this cutoff independently of `--confidence`. No optional fields means no extra model call. The request includes the resolved target, without the entity catalog or current numeric values.

Console output shows required/selected field names and probabilities for retained options plus `none`. `--json` retains the full distribution, including dropped fields. No values are extracted and no device or answer-model call runs.

Supply speaker context explicitly when benchmarking:

```bash
python benchmark.py -q "Mach das Licht aus" --speaker-area "Küche"
python benchmark.py -q "Mach das Licht im Büro aus"
python benchmark.py -q "Schalte die Schreibtischlampe aus"
```

Without `--speaker-area`, the speaker location is unknown and neither `entities_implied_area` nor `implied_area` is offered. HA obtains the room from the requesting device's area assignment when available. Text chat and unassigned devices provide no default room. The editable prompt and confidence threshold are under Decider; `--target-scope-confidence` overrides that threshold locally. The saved stock scope prompt upgrades automatically; custom prompts are preserved.

The default output is compact: selected action, speaker room, target type, area filter, candidate counts, chosen targets, detected parameter names, confidence and timings. `--json` retains full action field metadata and stage inputs/results. Existing supported action choices remain in use. General requests with enabled MCP servers stop before connecting or calling tools in plan-only mode.
