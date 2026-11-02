# Local testing

`local.py` imports the addon's `AgentEngine` and `ModelClient` directly. Decisions, prompts and state-query filtering share the same implementation.

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

Dry-run is the default. Add `--execute` to run selected actions, including read-only actions such as `todo.get_items`. General-model MCP connections and tool calls are also allowed in this mode. `benchmark.py` remains a compatibility entry point.

```bash
python local.py -q "Was steht auf der Einkaufsliste?" --execute
```

Actions that return data, such as `todo.get_items`, use the same target and parameter path. HA collects the response and includes it in debug records and optional LLM replies. With LLM replies off, to-do results list item names, descriptions and completion status; other action data is printed as JSON. Dry-run plans these calls without executing them; use `--execute` to see their returned data.

```bash
python local.py --query "Ist die Lampe im Büro an?" --language de
python local.py --query "Mach das Licht im Büro an" --confidence 0.25
python local.py --query "Sind im Haus Fenster offen?" --domain-confidence 0.4 --query-target-threshold 0.02 --json
```

`--confidence` overrides the single-choice confidence thresholds; probability cutoffs are separate. Individual flags override that value: `--service-confidence`, `--area-confidence`, `--target-scope-confidence`, `--domain-confidence`, `--query-type-confidence`, `--binary-answer-confidence`, `--numeric-confidence`, `--state-property-confidence`.

Match `--language` to the Assist pipeline. It defaults to `BENCHMARK_LANGUAGE` or `de`. Each run starts with no conversation history.

State queries follow service → area → single-domain selection → answer-type selection. Yes/no queries ask the decider for yes/no/unsure from all candidate states, without entity keep/drop. With LLM replies off, yes/no is formatted directly as Ja/Nein for German or Yes/No otherwise. With LLM replies on, the answer model translates the result into the Assist language. Other queries use one entity Choice question → probability filtering → reply generation. With LLM replies off, a single state_property Choice selects the requested scalar state/attribute for all retained entities; code formats name + value. With LLM replies on, the answer model receives the selected entity states as before. Unsure decisions stop with the configured unsure reply. Domain selection runs for rooms and `all_areas`; actions get their domain from the service. `--domain-filter-confidence` remains an alias for `--domain-confidence`.

Output shows stage timings and decisions. `--json` includes the full engine result, trace and usage. Snapshot fetch, engine and total times are separate.

With `--llm-answer`, add `--show-answer-input` to print the smart-home answer request: model, messages, temperature and output token limit. Messages include the system prompt, pipeline language, history and final state facts. For detailed queries, selected entities appear in one `entities` array, with each entity's ID, name, room and query state together. For yes/no replies, only `state_answer` is sent for translation. Nested JSON in `content` is expanded for display; the chat API still receives text. With `--json`, this readable view is available under `answer_input`, and the original messages remain in `trace`. API keys and headers are excluded.

The shared core runs actions in HA. The local tool uses `dry_run=True` by default and only invokes the executor with `--execute`. Execution uses the authenticated `POST /api/glow_assistant/execute` endpoint and the selected integration entry. The server rechecks current exposure and exclusions before each call and preserves the authenticated user context. Install this integration update and restart HA to register that endpoint. Calls are never automatically retried; timeouts leave the outcome unconfirmed. Action requests run **service → target type → area resolution/filtering → target resolution → capability prefilter → parameter detection → value extraction → action plan or execution → reply**. Target types:

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

Multi-area and entity selection retain every real option at or above its calculated probability cutoff; overall confidence is not used. `none` never overrides retained real options. Single-area selection ignores the multi-area threshold and uses only the unique highest probability; a tie or highest `none` stops. Prompts and probability thresholds are under Decider. Override the thresholds with `--action-area-threshold` and `--action-entity-threshold` (both default to 0.8). These are independent of `--confidence`. Full scores remain under `--json`; Choice probabilities share one distribution; the cutoff scales with the real option count.

After successful target resolution, expand the final targets to their exposed entity members in the cached snapshot. Apply HA field `filter.supported_features` and `filter.attribute` rules before parameter detection. A filtered field is retained when **at least one** resolved entity matches, following HA's field visibility rule; unfiltered fields remain available. Each field records its supporting and non-supporting entity IDs so partial support is explicit. A non-dimmable room member no longer hides brightness controls supported by other room members. Missing capabilities, unresolved feature names and unrecognized filter rules do not establish support. Nested field groups are flattened before filtering. No extra HA or model request is made.

The `parameter_capabilities` debug card/JSON record shows the actual target entity IDs, fields before/after filtering, retained fields, and removed fields with their HA filters and non-matching entity IDs, plus `field_support` for all fields. The optional-parameter table includes a `Targets` column such as `1/8` to show how many resolved entities support each retained parameter. The console shows a short capability-filter summary and the `capabilities` timing row. This prefilter does not infer conditional required fields beyond HA's supplied `required` flags.

Parameter detection then runs against the remaining action fields. Required fields are listed automatically. Optional fields use one Choice question plus `none`; every real field at or above the calculated cutoff from `parameter_probability_threshold` (default 0.5) is kept. Overall confidence is not a filter, and `none` never overrides retained real fields. `--parameter-threshold` overrides this cutoff independently of `--confidence`. No optional fields means no extra model call. The request includes the resolved target, without the entity catalog or current numeric values.

Console output lists all optional-field probabilities, including dropped fields with `drop: p < THRESHOLD`. Fields removed by the capability filter are labeled unsupported for the selected targets. `none` is labeled as a non-parameter. `--json` retains the full distribution. Required and retained optional fields are extracted next according to their HA selectors. Text, number and fixed-option select selectors are supported; other formats report not yet supported. Device calls run only with `--execute`. Successful action plans use the configured command reply (default Okay); with --llm-answer the model describes the plan without claiming execution.

Supply speaker context explicitly when benchmarking:

```bash
python local.py -q "Mach das Licht aus" --speaker-area "Küche"
python local.py -q "Mach das Licht im Büro aus"
python local.py -q "Schalte die Schreibtischlampe aus"
```

Without `--speaker-area`, the speaker location is unknown and neither `entities_implied_area` nor `implied_area` is offered. HA obtains the room from the requesting device's area assignment when available. Text chat and unassigned devices provide no default room. The editable prompt and confidence threshold are under Decider; `--target-scope-confidence` overrides that threshold locally. The saved stock scope prompt upgrades automatically; custom prompts are preserved.

The default output is compact: selected action, speaker room, target type, area filter, candidate counts, chosen targets, detected parameter names, extracted values, confidence and timings. `--json` retains full action field metadata and stage inputs/results. Advertised native actions can have required fields; format support is checked after field selection, rather than hiding the action. Actions with mandatory response data are included. General requests with enabled MCP servers stop before connecting or calling tools in plan-only mode.

## Relative Choice cutoffs

All multi-select Choice stages use `configured threshold / max(1, real option count)` as the effective cutoff. Exclude `none`/`unsure` from the count; retain probabilities at or above the cutoff. A threshold of 1 uses the uniform baseline, not independent per-option certainty; a threshold of 0 retains even zero-probability real options. With a configured threshold of 0.3, two real options use 0.15, three use 0.1, and ten use 0.03. Adding irrelevant options also lowers the cutoff and can retain weak matches.

This applies to action areas/entities, complex-query entities, optional parameters and text words. Defaults and flag names are unchanged. Single-choice confidence thresholds and the unique-highest explicit-area selection are unchanged. JSON/native debug show `probability_threshold`, `real_option_count` and `effective_probability_threshold`. The optional-parameter CLI table prints the calculation and uses the effective cutoff in drop reasons.

## State-query target experiment

Complex state queries use one Choice question containing all area/domain-filtered entities plus `none`, rather than one keep/drop question per entity. Each option includes its entity ID, name, aliases, area, device class, unit and current state; the catalog is not duplicated in the request state. Every actual entity with probability at or above the calculated cutoff from `query_target_probability_threshold` is retained. Overall confidence does not filter these selections, and `none` never overrides retained entities. An empty selection stops the query.

The default configured threshold is **0.7**; the effective cutoff is divided by the real option count. Set it under Decider or override it locally:

```bash
python local.py -q "wie warm ist es?" --query-target-threshold 0.02 --show-answer-input
```

The `target` timing row shows `selected N/M (one Choice)`. `--json` includes every option probability, the active threshold and selected entity IDs; `state_info` contains the selected states for quality comparisons. The old keep/drop helpers, settings and flags have been removed. Use the new state-query Choice prompt and probability threshold. Yes/no queries still bypass entity selection. Area/domain selection and answer generation are unchanged, including the current all_areas fallback for an unspecified room.

## Optional LLM replies

`local.py` requires `--llm-answer` to call the smart-home answer model, even if the exported HA setting is enabled. Without this flag, the answer model does not need a configured URL, model or API key. In HA, use **Smart-home answer → Enable LLM answer generation**, off by default. General-model routing has its own enable setting and is unchanged.

```bash
python local.py -q "Wie hell sind die Lampen im Büro?"
python local.py -q "Wie hell sind die Lampen im Büro?" --llm-answer --show-answer-input
```

Default replies:

- Successful commands: `ok_reply` (default `Okay`). HA executes selected actions; the local tool plans them unless `--execute` is set. Response-producing actions return their data instead of the command acknowledgement.
- Yes/no state queries: configurable `yes_reply` and `no_reply`. Blank values use Ja/Nein for German pipelines and Yes/No otherwise. Keep the existing binary-answer shortcut; no property selection or answer-model call.
- Other state queries: one `state_property` single-choice call selects the requested property for all selected entities. Output each name + value, with units. Brightness is shown as percent; hue and saturation come from `hs_color`. Unknown, unavailable or missing attributes are reported explicitly. Never substitute an entity's main state for a missing requested attribute. Duplicate names include room names when available.

The property selector receives the request, property names/metadata, language and enabled history, without the entity catalog or state values. Its prompt/confidence are under Decider. `--state-property-confidence` overrides the threshold. Only scalar state attributes and derived hue/saturation are offered; unsupported or ambiguous selections return unsure.

The CLI shows `answer mode` and the selected `state property`. JSON includes `state_property` and per-entity `state_values`; native debug has a property-selection card and an answer entry marked `model_call: false`. `--show-answer-input` reports that the answer model was not called. With `--llm-answer`, existing exact input tracing remains available. Fixed state labels use German for German pipelines and English/canonical values otherwise.

Set **Successful command reply**, **Yes reply** and **No reply** alongside the unsure/error replies in HA settings. Custom replies are used unchanged when LLM answer generation is off. The local tool reuses these exported HA settings.

## Parameter value extraction

Extract every required field and every retained optional field. Flattened HA selectors choose the extractor; the old item/message-only restriction is removed. Preflight all selected fields before making extraction calls. A boolean, date/time, object, generic list, formatted text, unknown selector or unusable numeric range returns a visible `not yet supported` error. Fixed-option select fields support scalar or list output as defined by HA. A dropped unsupported optional field does not block the request. The local tool executes device actions only with `--execute`. HA executes only after required and selected fields have been extracted successfully.

Text fields use one Choice over word occurrences plus `none`, with the existing relative cutoff. Each request includes the current field's HA metadata, all selected fields, the resolved target, pipeline language and enabled history. `other_parameters` explicitly lists the other selected fields; each has its own extractor. Prompts tell each extractor to leave those fields' values alone. Copy words only from the latest request. When both `item` and `description` are selected, extract the short item name separately from its details.

Number fields use one request with an absolute/relative Choice and a nine-anchor Score. HA selector bounds and step define the output; thermostat device bounds also apply. Relative final-setting requests need a known common current setting among the entities that support that field. Thermostat changes use the existing target and whole degrees, preserving the direction of small changes. Native brightness step fields extract signed deltas without requiring a device baseline. Numeric extraction prompts and confidence are under Decider; `--numeric-confidence` overrides confidence locally. A number selector without finite bounds stops explicitly.

```bash
python local.py -q "setze nudeln (weizen, fussilli, barilla) auf die einkaufsliste." --text-threshold 0.3
python local.py -q "mach das Licht etwas heller" --speaker-area "Wohnzimmer"
```

The console prints every extracted value in one `Parameter / Value` table. Each field has a separate timing row, such as `text:item`, `text:description` or `numeric:brightness_pct`. Native debug shows an extraction card per field. `--json` stores values in `parameters`, per-field details in `parameter_extractions`, and complete decisions, usage and timing under the field's stage name. Empty optional text fields are omitted; their extraction records and timing rows show skipped (no text). Required text fields with no words still stop the request. Invalid responses, unsupported formats and unclear numeric values still stop the request.

Edit the text, message, numeric, numeric-mode and select prompts under Decider. Saved stock text/service prompts upgrade automatically; custom prompts remain. The configured text threshold defaults to **0.4**; the optional-parameter threshold defaults to **0.5**. Existing saved thresholds and explicit flags remain unchanged. Text selection still uses `p >= configured threshold / real option count`, excluding `none`. Numeric confidence is separate from multi-select probability cutoffs.

## Action execution

Fixed-option select fields use one Choice call each. String options and `{value, label}` options are supported; extraction returns the actual HA value. Single-select returns one string, requiring a unique highest choice and `select_confidence` (default 0; `--select-confidence`). Multi-select returns a list and uses `select_probability_threshold` (default **0.3**; `--select-threshold`) divided by the real option count. Keep `p >= cutoff`, excluding fallback `none` from the count. Overall confidence does not filter multi-select; `none` does not override retained options. The prompt and settings are under Decider.

For `todo.get_items.status`, “muss noch eingekauft werden” maps to `["needs_action"]`; completed items use `["completed"]`. The CLI shows the value and a `select:status` timing row. Native debug uses `glow_assistant_extract_select`, including probabilities and thresholds. Empty or invalid selections stop before execution. Custom values and selectors without fixed options remain unsupported.

```bash
python local.py -q "Was steht auf der Einkaufsliste und muss noch eingekauft werden?" --execute --select-threshold 0.3
```

The core expands selected rooms to explicit exposed entity IDs. It groups targets by the parameters each supports, so an unsupported optional field is not sent to that target. Targets lacking any optional values can receive the base action. Missing required values or a target that cannot accept required fields stop before execution. Parameters cannot add broad `area_id`, `device_id`, `floor_id` or `label_id` selectors.

HA checks current Assist exposure and exclusions before every call, passes the Assist user's context and awaits the native service call. Calls run sequentially. A failed call is never retried, and later groups are not sent after a failure. The `execution` record includes each call's service, arguments, status, response, timing and error/cause. Partial or unconfirmed outcomes remain explicit. A completed call means HA accepted it; device state is not independently verified.

`dry_run=True` performs planning only and never invokes the executor, even if one is supplied. General MCP routes still stop before connecting or calling tools in dry-run mode. If only answer generation fails after successful execution, the reply and debug output say the command was already sent.

## Run logs

Logging is **off by default** and independent of `--json` and `--llm-answer`.

```bash
python local.py -q "stopp die musik" --speaker-area "Küche" --log-runs
python local.py -q "Wie hell sind die Lampen im Büro?" --log-runs --json
```

- Local files: `runs/successful/` and `runs/unsuccessful/`, beside `local.py`.
- HA files: `<config>/glow_assistant/runs/successful/` and `unsuccessful/`. Enable **Save run logs to disk** in the integration settings.
- The benchmark requires its own `--log-runs` flag even when HA logging is enabled.

One timestamp/UUID JSON file is written per completed or failed run. Unsure, unsupported, model/context errors, failed/partial actions and answer failures go under `unsuccessful`. A complete dry-run plan goes under `successful`. Success describes pipeline completion, not proof that the classifier understood the user or that a physical device changed state.

The shared `RunRecord` supplies the JSON envelope to both entry points. `--json` prints that report; `--log-runs` writes it regardless of console format. Records contain:

- Schema/run ID, source, integration version, UTC timestamps, mode and success flag.
- Request, pipeline language, speaker area, enabled history, entry/conversation IDs when available.
- Effective settings and prompts, full exposed snapshot and action metadata.
- Stage traces with offsets, decisions/probabilities, thresholds, usage, timings and execution outcomes.
- Exact model request bodies and JSON responses, including provider errors, HTTP status and timing. Non-JSON error bodies are not stored.
- Complete or partial results, plus exception details/tracebacks when a run raises.

Model inputs/responses are captured when logging or `--json` is enabled. API keys, bearer tokens, credential URL values and headers are redacted; environment variables are not dumped. Files use private permissions and atomic replacement. Writes run outside the event loop. A log failure is reported separately and never retries a device call or changes its outcome. Files are not automatically deleted. `runs/` is ignored by Git.
