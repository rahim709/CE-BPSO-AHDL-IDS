# Synthetic dataset generation and validation

These files are **artificial software fixtures**, not the original ToN-IoT, Bot-IoT or SWaT observations. They were generated at the user's request with representative column names, class imbalance, overlapping stochastic signals, correlated numeric measurements and occasional missing values. They are not fitted to the real corpora and do not establish their attack behavior, joint distributions, minority-class difficulty or architecture-ranking behavior. No synthetic result may be presented as an original-data reproduction.

The real, author-linked Gas Pipeline ARFF remains separate and unchanged; its recorded SHA256 was verified after generation.

| Profile | Rows | Files and schema | Labels |
|---|---:|---|---:|
| ToN-IoT-inspired |3,000|2,400 train +600 test;44 total columns including binary/multiclass labels; categorical protocol, DNS/TLS/HTTP fields and numeric traffic measurements|10|
| Bot-IoT-inspired |3,000|2,400 train +600 test;23 total representative flow/identifier/label columns; **not** the paper's unidentified10-feature subset|5|
| SWaT-inspired |3,000|Timestamp +51 named sensor/actuator columns +Normal/Attack; correlated stochastic stage signals and contiguous synthetic anomaly blocks|2|

ToN's44 total columns are not44 predictors. The loader removes alternate labels and configured identifiers before encoding. Bot's feature list is a representative schema subset, not a claim to match every field of a specific official release. SWaT's simple stochastic process is not a hydraulic digital twin; anomaly labels describe injected signal changes only.

Each file is prefixed `SYNTHETIC_`. Each dataset has `SYNTHETIC_MANIFEST.json` with the seed, generator version, class counts, column names and SHA256 checksums, plus a `README_SYNTHETIC.txt`. Raw fixtures are under `datasets/synthetic/`; configs are under `configs/synthetic/`; processed data and result directories remain separate from official data. Experiment metadata retains `data.synthetic: true`; test metrics, search summaries, diagnostics and search CSVs carry a synthetic-data flag.

## Commands

Run from the repository root:

```bash
source .venv/bin/activate
export MPLCONFIGDIR=/tmp/bpso-mpl
python -m scripts.generate_synthetic_datasets --rows 3000 --seed 381
python -m pytest -q
python -m scripts.test_synthetic_datasets
```

Generation is deterministic for a given seed and row count. At least500 rows are required. Minority coverage has an explicit minimum count per generated network partition to keep small software checks valid; this changes prevalence and is not an estimate of real minority rates. Increasing `--rows` produces larger artificial fixtures, not more real observations.

Individual CLI commands use the synthetic configurations explicitly:

```bash
python -m src.cli prepare --config configs/synthetic/ton_iot.yaml
python -m src.cli train --config configs/synthetic/ton_iot.yaml --output results/synthetic_manual/ton_train
python -m src.cli search --config configs/synthetic/ton_iot.yaml --output results/synthetic_manual/ton_search
```

Substitute `bot_iot` or `swat` for the other profiles. The combined runner creates a timestamped directory, runs preparation, fixed training, BPSO, diagnostics, seven baseline commands and reports for all three profiles. Its processed data stay inside that run directory. Comparison budgets are tiny smoke budgets with one seed; surrogate search stays in its initial random warmup. Fixed-model reports and comparison reports are separate to avoid treating duplicate BPSO seeds as independent replicates.

## Executed tests

The expanded unit/integration suite passed **57 tests in5.10 seconds**, including generation reproducibility, changed-seed checksums, class coverage in all partitions, target exclusion, finite processed values, all51 SWaT sensor fields and forward/training/evaluation on all three profiles. The JUnit artifact is `results/synthetic_datasets/tests.xml`.

The first combined CLI attempt completed ToN preparation/training/search/diagnostics/comparison but its aggregate report correctly rejected duplicate-seed aggregation. The runner was corrected to report fixed-model and method-comparison results separately. The corrected rerun passed all18 CLI commands, including21 baseline searches, three fixed-model runs and three ranking diagnostics. All generated search summaries and fixed-model metrics were verified to carry their synthetic-data flags.

Completed artifacts: `results/synthetic_datasets/20261105T112453Z`. Stable pointer: `results/synthetic_datasets/latest.json`. Verification summary: `results/synthetic_datasets/20261105T112453Z/verification.json`.
