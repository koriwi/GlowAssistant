# Architecture

[Decision flow (Mermaid)](decision_flow.mmd)

- **Shared confidence:** `decider.confidence`, default `0.5`. One HA field and one CLI flag: `--confidence`.
- **Single choice / numeric:** accept if model confidence is at least the shared value.
- **Multi-select, Choice (default):** keep each option if `p >= setting / option count`. Do not count `none` or `unsure`.
- **Multi-select, Noul:** one question per candidate, in one request per stage. Keep if `p(yes) >= confidence`. Ignore Choice thresholds. No `none` question.
- **Query fallback:** Choice requires both checks and rejects a chosen `none`. Noul uses the shared cutoff. Try once, without area/domain filters.
- Four settings: confidence `0.5`, general threshold `0.5`, query-target threshold `0.15`, text threshold `0.4`.

| Decision | Multi-select | Probability gate | Confidence gate |
| --- | :---: | ---: | ---: |
| Service / route | false | — | 0.5 |
| Query: areas | true | 0.5 / count | — |
| Query: domain | false | — | 0.5 |
| Query: yes/no or detailed | false | — | 0.5 |
| Query: yes/no answer | false | — | 0.5 |
| Query: entities | true | 0.15 / count | — |
| Query: properties | true | 0.5 / count | — |
| Query: fallback entities | true | 0.5 / count | 0.5 |
| Action: target type | false | — | 0.5 |
| Action: one area before entities | false | Highest wins; ties stop | 0.5 |
| Action: multiple areas | true | 0.5 / count | — |
| Action: entities | true | 0.5 / count | — |
| Action: optional parameters | true | 0.5 / count | — |
| Extract: text words | true | 0.4 / count | — |
| Extract: absolute/relative mode | false | — | 0.5 |
| Extract: numeric value (Score) | false | Sets the value; no gate | 0.5 |
| Extract: one enum value | false | Highest wins; ties stop | 0.5 |
| Extract: multiple enum values | true | 0.5 / count | — |

`—` = no gate. Confidence `0` disables the gate. Probability setting `0` keeps all options.
Every confidence gate shares the same setting. The table shows Choice defaults.

- `--threshold`: areas, fallback, properties, action entities, optional parameters and multiple enum values.
- `--query-target-threshold` and `--text-threshold`: independent relative cutoffs.
- `--confidence`: shared confidence value. Leave relative thresholds unchanged. With Noul, `0` retains every candidate.
- `--noul` / HA **Use Noul for multi-selection**: enable Noul for all multi-select stages. Single-choice and numeric calls stay unchanged.
- The division formula and tie rejection are fixed code.

[Probability/confidence review TODO](todo.md)
