# Architecture

[Decision flow (Mermaid)](decision_flow.mmd)

- **Single choice:** accept if `confidence >= setting`.
- **Multi-select:** keep each option if `p >= setting / option count`. Do not count `none` or `unsure`.
- **Query fallback:** require both checks. Reject a chosen `none`. Try once, without area/domain filters.
- All numeric gates below are configurable in HA and the CLI. These are code defaults.

| Decision | Multi-select | Probability gate | Confidence gate |
| --- | :---: | ---: | ---: |
| Service / route | false | — | 0.8 |
| Query: areas | true | 0.8 / count | — |
| Query: domain | false | — | 0.5 |
| Query: yes/no or detailed | false | — | 0.7 |
| Query: yes/no answer | false | — | 0.5 |
| Query: entities | true | 0.7 / count | — |
| Query: properties | true | 0.5 / count | — |
| Query: fallback entities | true | 0.8 / count | 0.8 |
| Action: target type | false | — | 0 |
| Action: one area before entities | false | Highest wins; ties stop | 0.8 |
| Action: multiple areas | true | 0.8 / count | — |
| Action: entities | true | 0.8 / count | — |
| Action: optional parameters | true | 0.5 / count | — |
| Extract: text words | true | 0.4 / count | — |
| Extract: absolute/relative mode | false | — | 0 |
| Extract: numeric value (Score) | false | Sets the value; no gate | 0 |
| Extract: one enum value | false | Highest wins; ties stop | 0 |
| Extract: multiple enum values | true | 0.3 / count | — |

`—` = no gate. Confidence `0` disables the gate. Probability setting `0` keeps all options.
Numeric mode and numeric value share one confidence setting.

- `--*-threshold`: probability setting. `--*-confidence`: confidence setting.
- `--confidence`: override all confidence gates. Leave probability settings unchanged.
- Legacy exceptions: `--area-confidence` and `--state-property-confidence` set probability thresholds.
- The division formula and tie rejection are fixed code.

[Probability/confidence review TODO](todo.md)
