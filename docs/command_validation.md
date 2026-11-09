# Command execution and test results

Executed on 2026-11-05 using the existing project virtual environment, CPU, and fresh explicitly artificial software records. No official dataset was present in datasets/raw at the start of the fixture checks. The author-linked Gas Pipeline file was then acquired successfully; the real-data results below supersede that initial missing-data condition.

## Verification

- Dependency installation/check: requirements already satisfied; `pip check` passed.
- Unit/integration tests: **53 passed**, zero failures/errors, pytest reported3.77 seconds. XML: `results/command_validation/tests.xml`.
- Compilation and CLI help passed.
- Fresh CLI runs: **16 passed,8 blocked by missing official input data,0 unexpected failures**.
- Comparison workflow: **21 searches and21 final models** (seven methods × three independent seeds42,43,44), all final models trained for the same two-epoch software-validation budget.
- Three-seed early-ranking diagnostics, selected-model training, search resumption, resource/time caps and report generation passed.
-146 PDF and146 PNG artifacts were generated across comparison, diagnostics and standalone reports. This includes repeated report exports in separate directories.

## Search methods exercised

| Method | Seeds | Completed search epochs per seed | Stop reason |
|---|---|---:|---|
| BPSO |42,43,44|12|Debug generation limit|
| Reduced-epoch BPSO |42,43,44|6|Debug generation limit|
| BPSO + successive halving |42,43,44|8|Debug generation limit|
| Hyperband |42,43,44|2|Debug bracket limit|
| Upstream DEHB |42,43,44|2|Debug bracket limit|
| Random |42,43,44|12|Shared resource cap|
| Surrogate |42,43,44|12|Shared resource cap|

These are functional checks under a shared cap, not matched-consumed-compute research comparisons. Fixture scores are deliberately not presented as IDS results.

## Blocked commands

Preparation using each official dataset configuration was attempted: Gas Pipeline, ToN-IoT, Bot-IoT and SWaT. All reported absent local raw files. Gas Pipeline fixed training, original BPSO, diagnostics and repeated comparisons also stopped because the processed official dataset was absent. Logs retain the exact errors.

## Artifacts and rerun command

- Complete command arguments, exit codes, durations and log paths: `results/command_validation/20261105T110119265514Z/commands.json`.
- Summary: `results/command_validation/20261105T110119265514Z/validation_summary.json`.
- Comparison tables: `results/command_validation/20261105T110119265514Z/reports/final_results.csv`, `search_results.csv`, `seed_summaries.csv`, `final_seed_summaries.csv`.
- Diagnostic results: `results/command_validation/20261105T110119265514Z/diagnostics/ranking_diagnostics.json`.
- Stable latest-run pointer: `results/command_validation/latest.json`.

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/bpso-mpl python -m pytest -q
MPLCONFIGDIR=/tmp/bpso-mpl python -m scripts.validate_commands
```

The command-validation script creates a fresh timestamped run and labels all artificial inputs explicitly. It does not download or silently replace official data. Full30/200-epoch scientific experiments remain unexecuted.


## Real Gas Pipeline follow-up

The published university-hosted `gas_final.arff` was retrieved successfully. It contains **97,019 rows,26 predictors and eight classes**, matching the paper's dimensional statistics. Its exact source and SHA256 are in `docs/gas_pipeline_source.json`. Preprocessing completed with **58,211 train /19,404 validation /19,404 test** rows and no exact train/validation or train/test row overlap. This is a configured stratified split, not a recovered paper split.

Measured fixed-model smoke result (tiny172-parameter CNN–GRU, two epochs):

| Metric | Actual value |
|---|---:|
| Held-out accuracy |63.0334%|
| Macro precision |.07879174|
| Macro recall |.12500000|
| Macro-F1 |.09665718|
| Weighted-F1 |.48741044|
| Recall for each of seven attack classes |0|
| Training plus validation |2.1063 seconds|
| Batched held-out inference |.0430 seconds|

Every prediction was the majority class0. This is an undertrained software check, not a useful IDS or successful reproduction. Test results were not used to change candidate selection.

The two-particle/one-generation BPSO check completed two one-epoch candidate evaluations in13.8568 search wall seconds. The separate two-candidate diagnostic completed six cumulative candidate epochs (1/2/3 per candidate) in45.5310 wall seconds. Its tied scores made Spearman/Kendall undefined; no inference about the research hypothesis is supported.

All seven baseline CLI commands also completed on real Gas data using `configs/gas_baselines_smoke.yaml`, a four-search-epoch cap, one seed and no final retraining. The surrogate smoke remains in its initial random warmup; surrogate acquisition is exercised by the unit/integration suite. This is not a full method-performance comparison. Exact budgets, stop reasons, times and validation scores follow:

| Method | Charged epochs | Search seconds | Best validation macro-F1 | Stop |
|---|---:|---:|---:|---|
|bpso|4|29.2098|0.09665718|completed|
|bpso_reduced|2|14.1692|0.09665718|completed|
|bpso_sh|3|22.6685|0.09665718|completed|
|dehb|3|16.7002|0.09665718|completed|
|hyperband|3|21.2901|0.09665718|budget_exhausted|
|random|4|26.0273|0.09665718|completed|
|surrogate|4|26.1172|0.09665718|completed|

Artifacts: `results/gas_real_smoke/real_data_summary.json`, `fixed_model/test_metrics.json`, `baseline_commands/reports/search_results.csv`, `diagnostics/ranking_diagnostics.json` and `reports/`. Source acquisition resolved the earlier Gas missing-data blocker, while ToN-IoT/Bot-IoT/SWaT inputs remain absent. Full30-epoch search /200-epoch final training,25-candidate diagnostics and repeated real-data comparisons were not launched at paper scale on this CPU environment. The latest test run passed **53 tests in3.97 seconds**.
