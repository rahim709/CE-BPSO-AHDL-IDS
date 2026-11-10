# Implementation map

| Component | Files |
|---|---|
| Binary BPSO and architecture encoding | `optimization/bpso.py`, `optimization/encoding.py` |
| Successive Halving, Hyperband and DEHB integration | `optimization/search.py` |
| UA-AMF-BPSO variants | `optimization/adaptive.py`, `optimization/predictive.py`, `optimization/dynamic.py`, `optimization/online.py`, `optimization/curve_predictor.py` |
| Neural network models and controls | `models/hybrid.py`, `models/controls.py` |
| Dataset preprocessing | `data/preprocessing.py` |
| Training and checkpoint continuation | `training/engine.py`, `training/runtime.py` |
| Metrics, diagnostics and specialist evaluation | `evaluation/` |
| Experiment CLI | `cli.py` |

Paths in the table are relative to `src/`.

From the repository root use `python -m src.cli prepare --config configs/gas_pipeline.yaml` for preprocessing. Experiment commands and result reproduction are documented in the main README, `experiments/README.md` and `plots/README.md`.
