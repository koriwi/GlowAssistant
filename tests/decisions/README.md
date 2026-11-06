# Decision cases

Run with `python scripts/decision_tests.py`. Uses the shared HA engine and the decider in `.env`.
No HA connection, device actions, answer text or answer-model calls.

- `snapshot.json`: fixed rooms, exposed entities, states, capabilities and action metadata.
- `cases.jsonl`: one case per line. Lines starting with `#` are disabled references. Add regressions here; extend the snapshot when needed.
- Reports: `benchmarks/decisions/`. Includes settings, prompts, inputs, responses, timings and scores. Credentials are removed.

```json
{"id":"todo.onions.de","category":"todos","language":"de","query":"Setze Zwiebeln auf die Einkaufsliste","speaker_area":"office","expect":{"route":"smarthome","service":"todo.add_item","entity_ids":["todo.einkaufsliste"],"parameters":["item"],"score_values":{"item":"Zwiebeln"}}}
```

`speaker_area` and `history` are optional. History contains prior `{ "role": "user"|"assistant", "content": "..." }` messages. Each case starts fresh. States do not change between cases.

Only fields in `expect` are checked. Entity, area, parameter and property lists ignore order. Area targets expand to fixture entities in the action domain. `parameters` checks required and selected optional field names. `select_values` checks enum values. Use a list of `expect` objects for equivalent valid outcomes.

`score_values` never changes pass/fail:

- Text: `1 - Levenshtein distance / max(lengths)`. Ignores case and extra whitespace.
- Number: `max(0, 1 - absolute error / scale)`. Default scale is the HA field's range. Set an explicit scale with `{"value": 22, "scale": 10}`. Reports also show absolute error.
- Missing, empty or uncertain extraction: score 0. Stops caused only by text/number extraction do not fail routing checks. API errors still fail.

Defaults use current code settings. `--settings FILE` loads settings from an export or run log; `.env` overrides model connections. `--confidence 0` disables confidence gates, but keeps relative probability thresholds.

`--confidence` sets one shared value for all confidence gates, including Noul. Default: `0.5`.
`--noul` replaces multi-select Choice calls with batched independent Noul questions. Keep each candidate with `p(yes) >= confidence`; Choice relative thresholds are ignored. With Noul, confidence `0` retains every candidate.

```bash
python scripts/decision_tests.py --noul --confidence 0.5
```

`--threshold` sets the general relative cutoff (default `0.5`). `--text-threshold` sets text extraction (default `0.4`). `--query-target-threshold` overrides the state-query entity cutoff. Tune it per model: replaying the stored D-model probabilities at `0.15` fixes its four incomplete query selections; that cutoff introduces false selections in the Jev log. The default is `0.15`.

Cases describe expected behavior, including known failures. The suite does not change prompts or repair decisions.

Cases copied from prompt examples, close variants and their DE/EN partners are commented out. Fresh replacements follow each reference. IDs retain their behavior categories; queries, contexts and expected values can change. Compare the case hashes before comparing benchmark scores. Keep evaluation queries out of prompt examples.
