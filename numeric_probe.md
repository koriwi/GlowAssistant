# Numeric extraction experiment

Run `numeric_probe.py` to test a decider on numeric settings independently of the conversation agent. It makes one request to the configured decider server and never connects to HA or executes a tool. Numeric request/scaling logic is shared with the main agent through `numeric_values.py`; probe flags and debugging output stay in this command.

## Commands

Absolute brightness:

```bash
python numeric_probe.py "Set brightness to 55%" --expected 55
```

Relative brightness:

```bash
python numeric_probe.py "Increase brightness by 15 percentage points" --current 40 --expected 55
```

Vague increase with an explicit default step:

```bash
python numeric_probe.py "Increase brightness" --current 40 --default-step 10 --expected 50
```

Temperature:

```bash
python numeric_probe.py "Make it 1.5 degrees warmer" --parameter temperature --unit °C --current 21 --min 15 --max 30 --levels 7 --expected 22.5
```

Vague room-temperature increase, using the prompt's 0.5 °C default:

```bash
python numeric_probe.py "Increase room temperature" --parameter temperature --unit °C --current 21.5 --min 18 --max 26 --levels 9 --expected 22
```

The prompt includes increase/decrease examples from 21.5 °C to 22.0/21.0 °C, plus explicit relative and absolute requests. `--default-step` overrides the vague-request default. These examples guide the model; they do not clamp or rewrite its output. Anchor spacing still limits resolution, especially for Choice.

Use your existing environment and `.env` decider settings (`DECIDER_URL`, `DECIDER_MODEL`, optional `DECIDER_API_KEY`), as described in [README.md](README.md#decider-server). The existing OpenRouter key still works for the default endpoint; unauthenticated local servers need no key. Supply the current value, range, and default step in the same units. The probe does not convert raw HA values. Omit `--current` for absolute requests; relative requests need it to determine a new value.

## What it compares

The default `--method both` asks two questions in the same request:

- **Score:** ordered descriptions of numeric settings. The returned ordinal score is linearly mapped to the supplied numeric range.
- **Choice:** the same settings as individual choices. Shows the selected anchor and a separate probability-weighted mean.

Defaults: brightness, %, range 0–100, six evenly spaced anchors (0, 20, 40, 60, 80, 100). Use `--method score` or `--method choice` to test one primitive. `--levels` accepts 2–10.

The interpolation is:

```text
value = min + raw_score × (max − min) / (levels − 1)
```

This is an experiment in interpreting a classifier's output. [Score is a weighted position on descriptive levels](https://docs.typesafe.ai/primitives/score), so a fractional result is not evidence of precise numeric extraction. Opposing probability peaks can average to a plausible middle value. The mapping stays within the anchor range; it does not extrapolate outside it. For broader experiments, change the range explicitly.

Vague Celsius room-temperature requests use a prompted 0.5-degree adjustment when current is known. Other vague requests without `--default-step` remain underspecified; the probe still shows the model's estimate so it can be examined. It is not applied to any device.

## Inspect and adjust

Add `--json` to see the full request, raw response, probabilities per anchor, decoded values, confidence, usage/cost, and latency:

```bash
python numeric_probe.py "Set brightness to 55%" --expected 55 --json
```

`--expected` is only used locally for absolute-error reporting. It is never sent to Jev. Use `--instructions "..."` to replace the wording for this probe run. Default numeric instructions live in `numeric_values.py` and are also used by the main agent.

Offline checks:

```bash
python -m unittest -v test_numeric_probe
```

These check request construction, scaling, validation, and isolation. Numeric extraction accuracy requires live runs against your prompts.
