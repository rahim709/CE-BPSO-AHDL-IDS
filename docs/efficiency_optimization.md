# GPU-efficiency optimization: verified implementation and reproduction

The complete measured report is `results/ua_amf_efficiency_20261108/REPORT.md`; the machine-readable conclusion is `research_conclusion.json` in that directory. The current course manuscript remains `docs/latex/manuscript.pdf`. Its prior version and source were preserved in `results/ua_amf_efficiency_20261108/manuscript_before`.

## What changed

`src/training/runtime.py` adds bounded CPU-array/resident-tensor caches, vectorized batch fetching with the original sampler RNG sequence, and a single hot model/optimizer cache isolated by trial path. The existing trainer accepts optional runtime flags; defaults preserve dense validation and original training behavior. The accepted profile uses FP32, epoch-boundary loss scalar reads and validation at 3/10/30. Every completed epoch still has an atomic checkpoint and trajectory record. Model, optimizer, Python/NumPy/Torch/CUDA and DataLoader RNG states remain resumable. AMP scaler state is saved when AMP is explicitly requested; AMP was measured and rejected for the matched experiments due to changed outcomes. Architecture batch sizes and the 73-bit search space were not narrowed or silently changed.

`src/optimization/dynamic.py` implements the investigational eager policy. It initially probes two candidates at 3 epochs, chooses further continuations by predicted terminal F1, calibrated radius and normalized remaining cost, and forces another probe after two promotions. It prunes only when an uncertainty upper bound is below an actual terminal incumbent plus 0.01. All personal/global best and reported winners use observed 30-epoch scores. Unlike fixed successive halving, promotion counts are not preset.

`Evaluator` and `run_search` add terminal-quality target stopping and an explicit execution-time budget mode. Standard BPSO updates, Hyperband and DEHB proposals are unchanged. The same target-stop wrapper and fast runtime apply to every method. Controlled target/budget exceptions are not numerical failures. DEHB's exception-suppressing wrapper is handled explicitly. Only finite, actually evaluated architectures can win.

The priority is

    P(a) = predicted_final_F1(a) + beta*radius(a)
           - lambda*remaining_cost(a)/median_active_remaining_cost.

Remaining cost is mean observed training time times remaining epochs, plus mean observed validation time times the known number of remaining validation passes. This avoids extrapolating dense validation into a sparse-validation future. Beta=0.1 and lambda=0.2 were frozen by architecture-disjoint development cross-validation before seeds 601/602/603. The predictor uses sparse current-rung features with timing excluded, avoiding hardware-timing covariate drift. Fit IDs 0–11, calibration IDs 12–17 and held-out IDs 18–24 remain separate. No own 30-epoch target enters a prediction feature. Upper-bound coverage remains an empirical small-archive assessment, not a guarantee under adaptive BPSO proposals.

## Diagnosed bottlenecks

The earlier 30-second experiment ran all methods until the same cap; it could not demonstrate time-to-quality savings. There was no target-reached stop. Its continuation and decoded caches were working. Predictor calls cost about 0.10 seconds per full-UA search; archive collection cost 403.819 GPU seconds. Median continuation-cost error was 1.9% on terminal survivors (selection-biased). The original three-epoch profile recorded 319 stream synchronizations and 1251 scalar reads, including framework/optimizer operations. These counts are not all neural kernel work.

GPU seconds retain the project's synchronized training/validation elapsed convention, which includes host stalls inside those intervals. They are not pure kernel time or energy. Checkpoint writes, model initialization and scheduler/predictor work also enter search wall time. The backend engineering cuts this measured elapsed time while retaining weights; it should not be described as a measured FLOP or energy reduction.

## Executed protocols

The same real Gas Pipeline subset is used throughout: 8192 parent-training and 4096 parent-validation rows, all eight labels, parent training-fitted MinMax, balanced training-only class weights, the full 73-bit space and common 30-epoch terminal fidelity. The parent scaler saw training rows outside the reduced subset; validation/test rows never fitted preprocessing. Data fingerprints and parent indices are in the preceding reliability study.

The engineering profile ran eight profiles on three fixed development architectures at 30 epochs. The original 180-second profile cap interrupted the final combined trial; it resumed under an explicit additional 30-second cap. Original partial results were preserved. That resumed trial's GPU time is cumulative, but its wall field is continuation-only; it is not compared as a full-trial wall measurement. All combined FP32 parameter tensors matched their legacy counterparts exactly. Aggregate GPU elapsed dropped 25.059→20.182 seconds (19.47%). One-pass fixed-order timing and possible first-use effects limit generalization. A separate immediate-continuation check measured model constructions 3→1 and checkpoint loads 2→0 with exact weights.

The first actual search pilot completed 108 runs: nine variants × three search seeds × (independent target 0.4/0.5/0.6 runs plus fixed compute). Each had an 8-second GPU execution cap and 20-second wall safety cap. Many runs missed even target 0.4. Every miss remains censored; no seed was discarded.

A declared exploratory continuation completed another 54 continued runs: the same nine methods and seeds, target 0.6 and fixed compute, with total GPU allowance 20 seconds and wall allowance 50. Algorithms, coefficients, seeds and initial swarms did not change. All original epochs/GPU/wall and checkpoint-import time were charged. Checkpoints and ledgers were copied into a new output root with explicit identity/provenance migration; original directories remain untouched. These are continuations, not new independent replications. Model/optimizer/RNG continuation is exact at completed epochs; native scheduler decisions are reconstructed by replay/caching, with upstream DEHB replay subject to its own proposal/state limitations.

At target 0.6, full UA used 12.803 GPU seconds versus BPSO 17.008: an actual-consumption saving of 24.72%, with 2/3 versus 1/3 hits. Full UA was 5.78% slower than BPSO-SH, which used 12.103 seconds with 2/3 hits. DEHB used 6.776 seconds with 3/3 hits (60.16% less than BPSO). All figures retain failed seeds. Restricted completion time min(T,20) is reported separately; its 25.14% saving is not substituted for the actual-cost 24.72% to claim the 25% goal was met.

At fixed 20-second compute, full UA macro-F1 was 0.5391, BPSO 0.2696, SH 0.5030 and DEHB 0.7356. Thus full UA does not retain quality within 0.02 of the strongest control. Fixed-compute GPU totals near 20 seconds are not called savings. Mean/SD anytime points require three terminal candidates, one per seed; raw seed traces retain early missing outcomes.

The uncertainty-bonus ablation had identical winner scores to full UA. Cost helped F1 by 0.0320 over no-cost but did not beat SH on GPU time. Fixed-quota prediction was faster and slightly stronger than dynamic prediction. Bound-pruning/promotion logs expose allocation behavior; skipped candidates alone do not establish an advantage.

## Archive and final-model costs

New predictor archive training GPU cost is zero because existing trajectories were reused. Their historical cost is still charged: the full archive cost 403.819 GPU seconds, and the fit/calibration subset alone cost 308.386. Full UA's unamortized target-run cost is 416.622 GPU seconds versus 17.008 for BPSO. There is no net single-search saving. Smaller CPU-only refits reduce attributable historical subsets but increase point-prediction errors; they were not substituted after the benchmark. An online calibration guarantee was not inferred from too few adaptively selected terminal models.

All nine methods' 20-second winners were frozen before historical-test evaluation. Dedicated seed 1601 and 30 epochs produced 27 model references from 15 unique fits, in 92.174 seconds under a separate 180-second cap. The historical test was already examined and is not independent/blind. Full-UA mean test macro-F1 was 0.3100 versus SH 0.5759 and DEHB 0.6859; full-UA normal FPR was 94.42%. This configuration is rejected as a robust detection-performance improvement. One final seed per selected architecture and shared historical rows remain limitations. Per-class recall, accuracy, precision, confusion matrices, parameters and inference latency are retained per model. Existing full-data MLP/CNN/GRU/reference results remain contextual; different endpoints and row counts prevent matched superiority claims.

## Reproduction

Run from the repository root in the existing environment. CUDA commands require device access outside the sandbox on this machine. Completed raw experiments are reused; choose fresh output paths to measure timings again.

```bash
# Correctness and exact continuation
.venv/bin/python -m pytest -q
BPSO_TEST_DEVICE=cuda .venv/bin/python -m pytest -q tests/test_training_search.py tests/test_adaptive.py tests/test_curve_predictor.py tests/test_runtime_efficiency.py tests/test_efficiency_search.py

# Recorded development/profile study (existing outputs reused)
.venv/bin/python -m scripts.efficiency_development
MPLCONFIGDIR=/tmp/eff-mpl .venv/bin/python -m scripts.efficiency_profile

# Fresh timing replication using the frozen development bundle and costs
MPLCONFIGDIR=/tmp/eff-mpl .venv/bin/python -m scripts.efficiency_benchmark --output results/efficiency_replication --development-root results/ua_amf_efficiency_20261108 --phase all
MPLCONFIGDIR=/tmp/eff-mpl .venv/bin/python -m scripts.efficiency_extend --pilot-root results/efficiency_replication --phase all
MPLCONFIGDIR=/tmp/eff-mpl .venv/bin/python -m scripts.efficiency_analyze --input results/efficiency_replication

# Rebuild current verified report/manuscript section, without retraining
MPLCONFIGDIR=/tmp/eff-mpl .venv/bin/python -m scripts.efficiency_analyze
MPLCONFIGDIR=/tmp/eff-mpl .venv/bin/python -m scripts.efficiency_report

# Standalone future search; give controls the same configuration/runtime
.venv/bin/python -m src.cli search --config configs/ua_amf_efficient.yaml --method ua_amf_bpso --seed 604 --output results/efficient_future604
```

`efficiency_extend` explicitly migrates only the compute allowance into a fresh output tree and retains spent compute. It must not be replaced by resetting wall state or copying checkpoints without importing the ledger. To repeat engineering profiles use `--output` with a fresh directory. `efficiency_report` updates the current manuscript from checked default-root results; it never retunes the method. Build `docs/latex/manuscript.tex` using pdflatex, bibtex and two further pdflatex passes.

## Course-project conclusion

The code and measured findings are reproducible, with 105 CPU tests and 50 targeted CUDA tests passing. The defensible contribution is FP32 execution engineering and benchmark design. Full UA shows a limited online saving against standard BPSO, but misses the engineering target on actual consumption, is slower than SH, is much weaker than DEHB, loses final-model performance and has no net archive-inclusive saving. Recommend DEHB as the strongest measured control; use BPSO-SH as the simpler BPSO baseline. Do not claim a superior or novel uncertainty-aware optimizer or deployment readiness from these three-seed reduced-row experiments.
