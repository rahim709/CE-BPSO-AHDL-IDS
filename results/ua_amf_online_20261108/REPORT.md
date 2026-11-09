# Archive-free computational-efficiency study

## Outcome

Zero separately trained GPU predictor archive was required. Archive-free full used 11.947 actual GPU seconds on average versus 16.824 for BPSO: 28.99% lower total GPU cost in the target-or-cap pilot. Targets were reached in 2/3 versus 1/3 seeds. All failures remain right-censored in the saved analysis.

However, all three online variants made 0 predictive promotion decisions across 18 matched runs. They used SH fallback. They produced identical winner scores to SH, whose mean target-or-cap GPU was 12.004. The 0.47% timing difference versus SH is small execution variation, not evidence that online learning improves SH. DEHB reached 3/3 targets at 6.778 GPU seconds and remains the recommended measured optimizer for this pilot. SH is the recommended simple BPSO alternative.

## Measured target-or-cap comparison

| Method | Online GPU s | Wall s | Standalone total GPU s | Terminal F1 (SD) | Target reached |
|---|---:|---:|---:|---:|---:|
| BPSO | 16.824 | 17.542 | 16.824 | 0.3054 (0.3215) | 1/3 |
| BPSO + SH | 12.004 | 12.754 | 12.004 | 0.4971 (0.3180) | 2/3 |
| DEHB | 6.778 | 7.265 | 6.778 | 0.7026 (0.0350) | 3/3 |
| Offline UA | 12.631 | 13.364 | 416.450 | 0.5391 (0.2719) | 2/3 |
| Online prediction | 11.919 | 12.469 | 11.919 | 0.4971 (0.3180) | 2/3 |
| Online + uncertainty | 11.950 | 12.474 | 11.950 | 0.4971 (0.3180) | 2/3 |
| Archive-free full | 11.947 | 12.485 | 11.947 | 0.4971 (0.3180) | 2/3 |

Means use all three seeds601/602/603; failed target0.6 runs are retained. Scores are best genuinely evaluated30-epoch validation macro-F1 before the strict GPU/wall horizon, not extrapolated prediction scores. Large sample SD indicates instability. Actual GPU time includes final epoch overruns; target decisions beyond the cap are censored. Restricted time-to-target is recorded separately and is not substituted for measured GPU consumption. No superiority test is claimed.

Standalone total GPU assigns the entire403.819-second historical archive to one offline-UA search. `campaign_costs.csv` also charges it once across three seeds, avoiding triple-charging a reused archive. Offline archive wall elapsed was420.722 seconds. The old offline CPU development/refitting preparation is not fully isolated in these new search logs, so archive-inclusive wall is a documented lower bound; no zero-cost assumption is made. CPU predictor fitting for the new variant is measured inside each search wall time.

In the primary online trials CPU preparation ranged 0.00173--0.00784 seconds; CPU fits totalled 0. Candidate maturation is paid search work; it is not an uncharged calibration phase.

Total newly executed search/diagnostic work across all methods was 794.105 GPU seconds; the separate diagnostic consumed 120.061 of these seconds. Its model was not transferred into any matched run. Peak allocated GPU memory across this study was 3158.0 MiB. Numerical/non-budget failures: 0. Largest final-epoch cap overrun: 0.448 seconds. Tests and analysis are separate verification work, not search savings.

## Fixed-compute and ablations

| Method | 20-second terminal macro-F1 mean (SD) | Epochs mean |
|---|---:|---:|
| BPSO | 0.2696 (0.3715) | 178.7 |
| BPSO + SH | 0.5030 (0.3231) | 138.3 |
| DEHB | 0.7356 (0.0402) | 195.0 |
| Offline UA | 0.5391 (0.2719) | 205.0 |
| Online prediction | 0.5030 (0.3231) | 137.3 |
| Online + uncertainty | 0.5030 (0.3231) | 137.3 |
| Archive-free full | 0.5030 (0.3231) | 137.7 |

Equal GPU caps measure quality at fixed compute; near-equal consumption does not prove savings. Five/ten/twenty-second anytime scores and raw seed traces are saved. Prediction-only, uncertainty-only and full cost-aware policies have no causal allocation difference when the gate never activates. Any terminal score difference at a hard time boundary would be a timing effect, not evidence for a component. The older offline control uses its frozen development-tuned coefficients and does not transfer labels or tuned coefficients into the new modes.

At fixed20-second compute, archive-free full preserved SH scores exactly but trailed offline UA by 0.0361 and DEHB by 0.2325 absolute macro-F1. Both gaps exceed the proposed0.02 engineering margin. Competitive quality against the strongest control was not preserved.

## Online predictor diagnostic

The separate seed 601 diagnostic used 120.061 GPU seconds and 128.449 wall seconds, matured 16 distinct terminal architectures, performed 1 CPU fits and applied prediction at 0 promotion decisions. Preparation consumed 0.0475 CPU wall seconds. Gate counts: insufficient fit or calibration architectures: 32; insufficient out of time check architectures: 1. This is a cold-start diagnostic, not a matched performance comparison.

The conservative predictor requires enough disjoint fit/calibration architectures and four genuinely prequential checks per rung. Its held-out coverage is descriptive under adaptive sampling, not a formal coverage guarantee. Insufficient observations or inaccurate/overwide predictions trigger SH. Removing gates retrospectively to obtain activation would invalidate this pilot. A next experiment should prespecify a larger matched horizon based on maturation counts and assess prediction diagnostics before optimizer superiority claims. If reliable activation remains uneconomic, retain SH or DEHB.

## Changes and verification

- `src/optimization/online.py`: CPU forests, frozen prefix features, fixed architecture partitions, prequential checks, SH fallback, measured cost ranking and independent audit logs.
- `src/optimization/search.py`: additive online policy dispatch; standard BPSO remains available.
- `configs/ua_amf_online.yaml`: archive-free deterministic settings.
- `scripts/archive_free_benchmark.py`, `archive_free_analyze.py`, `archive_free_report.py`: matched runs, ledger/prediction/leakage audits, tables, plots and manuscript updates.
- `tests/test_online_search.py`: future-feature exclusion, own-architecture guards, degraded-prediction fallback, exact SH fallback/continuation, no offline loader access.

Final checks: CPU 110 passed; targeted CUDA 55 passed. Independently re-scored 948 saved validation prediction archives; all checked metrics and GPU/epoch ledgers matched. Architecture fit/calibration/query identities and prefix features were audited. Logs and per-run checkpoints remain under the study directory.

No new test evaluation was run for this change. Existing MLP/CNN/GRU and selected-model test results remain contextual; the historical Gas test is not new blind independent data. The study uses a bounded8192/4096 real-data view and three seeds on one GPU; full-data/full-scale performance, calibrated uncertainty under proposal shift and IEEE-level optimizer novelty are not established.

## Reproduce

See `docs/archive_free_ua_amf.md` for formulation, pseudocode, frozen protocol and exact commands. Run benchmark in a fresh output for a fresh replication; existing completed runs are intentionally skipped. The data folder references shared training/validation arrays, not an offline predictor. The dependency audit distinguishes those artifacts.
