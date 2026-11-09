# Results review and implemented improvements

Reviewed the saved two-epoch smoke runs, validation predictions and class-recall trajectories. Software success did not imply useful learning: the initial fixed models each predicted only one class. ToN and SWaT synthetic models even predicted a non-majority class at this budget. Gas and Bot also missed all but one class. Synthetic findings are software diagnostics only.

## Changes

1. Added prediction counts, predicted-class coverage, zero-recall classes and a single-class-collapse flag. Validation trajectories, candidate-evaluation records and final CSV/JSON reports expose these diagnostics.
2. Added a majority-class reference selected using **training labels only**, scored separately on validation or test for context. Accuracy alone can hide total attack-recall failure.
3. Added optional standard inverse-frequency (`balanced`) and square-root (`sqrt_balanced`) class-weighted cross entropy. Weights use only training labels; no resampling or test information is used. Default `none` preserves the paper-informed baseline. Configuration-scoped cache keys and exact checkpoint continuation still apply.
4. Corrected ranking interpretation: tied selection boundaries now yield null top-k overlap/elimination values rather than arbitrary perfect stability. Deterministic candidate-index values remain separately labelled descriptive. Tied early winners receive a regret interval when a single regret is not identifiable. Constant rankings carry `ranking_informative: false`.
5. Recorded short-sequence shape repairs per training epoch. Added a conventional reference CNN16/GRU32 preset and bounded learning sanity runner. This is an architecture/training ablation, not a novel optimizer.

Original checkpoints/results remain archived. `results/result_review/initial_validation_health.json` audits the four initial validation runs; `corrected_smoke_rankings.json` recomputes eight comparisons from the already saved trajectories without retraining or test access.

## Fixed-budget validation-only comparison

All16 runs used seed42, batch256, Adam(lr.001),10 epochs, identical frozen partitions per dataset and the same gradient clipping. Architecture presets are: tiny CNN4(kernel2,pool1)→GRU4(dropout0), and reference CNN16(kernel3,pool2)→GRU32(dropout.1), followed by the same last-sequence-position classifier convention. The architecture factor changes multiple block settings, so it is not a pure parameter-count ablation. Different sizes have different computational costs. Real Gas results are separate from all synthetic profiles.

| Dataset | Architecture | Weighting | Validation macro-F1 | Predicted classes | Classes with zero recall |
|---|---|---|---:|---:|---:|
|gas_pipeline_real|tiny|none|0.096657|1|7|
|gas_pipeline_real|tiny|balanced|0.096657|1|7|
|gas_pipeline_real|reference|none|0.096657|1|7|
|gas_pipeline_real|reference|balanced|0.228693|5|4|
|ton_iot_synthetic|tiny|none|0.076923|1|9|
|ton_iot_synthetic|tiny|balanced|0.006061|1|9|
|ton_iot_synthetic|reference|none|0.076923|1|9|
|ton_iot_synthetic|reference|balanced|0.080073|7|7|
|bot_iot_synthetic|tiny|none|0.009756|1|4|
|bot_iot_synthetic|tiny|balanced|0.009756|1|4|
|bot_iot_synthetic|reference|none|0.135195|2|3|
|bot_iot_synthetic|reference|balanced|0.151469|3|2|
|swat_synthetic|tiny|none|0.431818|1|1|
|swat_synthetic|tiny|balanced|0.411604|2|0|
|swat_synthetic|reference|none|0.431818|1|1|
|swat_synthetic|reference|balanced|0.428571|2|1|

The real Gas reference changed from validation macro-F1.096657 without weights to0.228693 with balanced weights at the same budget (see exact JSON values). The tiny balanced model remained collapsed, indicating that weighting alone did not resolve that preset's learning problem. On synthetic SWaT, weighting restored two-class predictions but slightly reduced macro-F1; no universal improvement or automatic default change is justified. The synthetic ToN and Bot reference scores improved only modestly and still left classes with zero recall.

## One held-out Gas verification

The two following configurations were fixed after the validation comparison and restarted using the same final seed10042 and10-epoch budget. The sanity runner itself never reads test.npz. This is a single split/seed verification, not the paper's200-epoch final protocol, not statistical evidence of generality, and not an equal-time search comparison.

| Metric | Tiny unweighted control | Reference balanced model |
|---|---:|---:|
|Accuracy|0.630334|0.707792|
|Macro-F1|0.096657|0.299371|
|Macro recall|0.125000|0.378023|
|Weighted-F1|0.487410|0.592088|
|Predicted classes|1.000000|6.000000|
|Parameters|172.000000|5128.000000|
|Training + validation seconds|11.215265|14.605253|

| Class code | Control recall | Balanced reference recall |
|---|---:|---:|
|0|1.000000|0.999836|
|1|0.000000|0.019892|
|2|0.000000|0.000000|
|3|0.000000|0.000000|
|4|0.000000|0.006545|
|5|0.000000|0.965217|
|6|0.000000|0.032698|
|7|0.000000|1.000000|

The reference still has zero recall on classes2 and3. Its larger parameter count and training cost are explicitly reported; these changes do not demonstrate compute savings, IoT energy reduction or reproduction of the paper's reported results. Older Gas data confounds and unknown paper split identities remain. Numeric class codes have not been assigned attack names without verified release documentation.

## Verification and reproducible commands

The expanded suite covers train-only weighting, minority gradient contribution, exact weighted checkpoint continuation, weighted search with the test file removed, collapse detection, a training-fitted majority reference and non-identifiable tied rankings. The final test record is `results/learning_sanity/tests.xml`.

```bash
source .venv/bin/activate
export MPLCONFIGDIR=/tmp/bpso-mpl
python -m pytest -q
python -m scripts.audit_saved_results
python -m scripts.learning_sanity --epochs 10
python -m src.cli train --config configs/gas_learning_control.yaml --output results/learning_sanity/final_verification/control
python -m src.cli train --config configs/gas_balanced_reference.yaml --output results/learning_sanity/final_verification/balanced_reference
```

Use a fresh output directory when changing configs. Optional class weighting must be held identical across search methods in a compute-efficiency comparison and treated as a documented departure from the original unweighted baseline. The primary research question about unreliable low-fidelity rankings still requires the prespecified diverse cohort, longer budgets, repeated seeds and established multi-fidelity controls.

Final verification: **67 tests passed in5.16 seconds**, with no failures. Sixteen validation-only runs and both fixed held-out Gas checks completed successfully. No universal or statistically confirmed improvement is claimed.
