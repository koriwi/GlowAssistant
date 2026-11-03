# Local tools

Run from the repository root. All Python tools use the root `.env`.
Run logs stay in `runs/`. Area benchmarks stay in `benchmarks/area_filter/`.
Files starting with `_` are internal helpers, not commands.

| Tool | Purpose |
| --- | --- |
| `local.py` | Test the shared HA engine. Actions require `--execute`. See [setup](../local.md). |
| `entity_probe.py` | Select one entity from the full exposed snapshot. No actions. |
| `numeric_probe.py` | Compare numeric Score and Choice results. See [options](numeric_probe.md). |
| `area_filter_probe.py` | Run the DE/EN area-filter benchmark. Save results for model comparisons. |
| `release.sh TAG` | Amend the manifest version, tag, push and publish a release. |

```bash
python scripts/local.py -q "Ist die Lampe im Büro an?"
python scripts/entity_probe.py -q "Wann wurde zuletzt gemäht?" --repeat 5
```
