# Experiment configurations

These runnable configurations inherit the canonical settings in `configs/` so dataset schemas and training parameters stay consistent.

| Configuration | Purpose |
|---|---|
| `debug.yaml` | Small CPU execution check |
| `gas_pipeline.yaml` | Paper-informed BPSO search and fixed model |
| `gas_low_false_alarm.yaml` | Square-root-weighted hybrid |
| `submission_evaluation.yaml` | Fixed 200-epoch reference evaluation |
| `research_extension.yaml` | Training-duration and model controls |
| `ua_amf.yaml` | Initial adaptive multi-fidelity variant |
| `ua_amf_prediction.yaml` | Predictive allocation study |
| `ua_amf_efficient.yaml` | Execution efficiency study |
| `ua_amf_online.yaml` | Archive-free allocation study |

Run from the repository root, for example:

```bash
python -m src.cli search --config experiments/debug.yaml --output results/new_debug
```

Full experiment runners are in `scripts/`. Final frozen protocols, resolved configurations and seed measurements are retained under each `results/` study. Choose a fresh output directory for a new run.
