# Computational efficiency of predictive UA-AMF-BPSO

Completed: 25 sampled architectures; 61/75 terminal reliability trajectories; 24 timed searches (eight methods/ablations, three seeds). Reliability charged 1830 epochs, 420.722 wall seconds and 403.819 synchronized GPU seconds. Fourteen third-seed reliability trajectories remain unexecuted.

## Early-fidelity reliability

| Seed | Paired architectures | Early epoch | Spearman | Kendall | Top-5 recall | Regret |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 25 | 3 | 0.3632 | 0.2857 | 0.4000 | 0.1371 |
| 42 | 25 | 10 | 0.9134 | 0.8098 | 0.6000 | 0.0199 |
| 43 | 25 | 3 | 0.8048 | 0.6525 | 0.8000 | 0.2565 |
| 43 | 25 | 10 | 0.8053 | 0.6809 | 0.8000 | 0.0625 |
| 44 | 11 (partial prefix) | 3 | 0.6424 | 0.5017 | 0.8000 | not identifiable (ties) |
| 44 | 11 (partial prefix) | 10 | 0.9116 | 0.8131 | 1.0000 | 0.0000 |

Late-blooming early-rejected candidates and regret tie ranges are exported separately. The incomplete third-seed prefix is not pooled as a complete replicate.

## Timed baseline comparison

| Method | Seeds | Macro-F1 mean ± SD at 30 s | GPU s mean ± SD | Wall s | GPU saving vs BPSO |
|---|---:|---:|---:|---:|---:|
| bpso | 3 | 0.4915 ± 0.1686 | 28.76 ± 0.97 | 30.17 | 0.0% |
| bpso_prediction | 3 | 0.6832 ± 0.0080 | 28.49 ± 0.28 | 30.08 | 1.0% |
| bpso_prediction_uncertainty | 3 | 0.6335 ± 0.0788 | 28.55 ± 0.34 | 30.05 | 0.7% |
| bpso_sh | 3 | 0.6832 ± 0.0080 | 28.71 ± 0.53 | 30.17 | 0.2% |
| dehb | 3 | 0.6362 ± 0.0706 | 28.50 ± 0.40 | 30.11 | 0.9% |
| hyperband | 3 | 0.6026 ± 0.1266 | 28.69 ± 0.36 | 30.16 | 0.2% |
| random | 3 | 0.5010 ± 0.1467 | 28.67 ± 0.32 | 30.08 | 0.3% |
| ua_amf_bpso | 3 | 0.6335 ± 0.0788 | 28.76 ± 0.36 | 30.29 | -0.0% |

The strict-30-second validation macro-F1 difference (full minus standard BPSO) is +0.1420; full minus ordinary halving is -0.0496. GPU execution saving versus standard BPSO is -0.01%; search-wall saving is -0.41%. Equal wall caps constrain total-time differences by design. A lower GPU figure alone is not a speedup if detection performance is worse. SDs describe three search seeds, not independent capture variability or confidence intervals.

## Time-to-target and anytime behavior

The complete 10/20/30-second results are in `anytime_results.csv`; missing terminal models remain missing. Target hit counts and conditional hit times are in `time_to_target_summary.csv`, with censoring at 30 seconds. A target missed by either method does not receive a fabricated saving.

- Target 0.2, bpso: 3/3 seeds attained; conditional mean hit time 14.068 s.
- Target 0.2, bpso_sh: 3/3 seeds attained; conditional mean hit time 8.372 s.
- Target 0.2, ua_amf_bpso: 3/3 seeds attained; conditional mean hit time 14.713 s.
  Paired full-versus-BPSO saving: mean -43.28% across 3 jointly attaining seeds only.
- Target 0.4, bpso: 2/3 seeds attained; conditional mean hit time 15.561 s.
- Target 0.4, bpso_sh: 3/3 seeds attained; conditional mean hit time 11.508 s.
- Target 0.4, ua_amf_bpso: 3/3 seeds attained; conditional mean hit time 14.713 s.
  Paired full-versus-BPSO saving: mean 5.00% across 2 jointly attaining seeds only.
- Target 0.6, bpso: 1/3 seeds attained; conditional mean hit time 9.495 s.
- Target 0.6, bpso_sh: 3/3 seeds attained; conditional mean hit time 11.508 s.
- Target 0.6, ua_amf_bpso: 2/3 seeds attained; conditional mean hit time 8.781 s.
  Paired full-versus-BPSO saving: mean 16.58% across 1 jointly attaining seeds only.

## Predictor, promotions and ablations

At epoch 3, held-out observation coverage is 1.000, all-seed architecture coverage 1.000, mean width 0.6992, MAE 0.1261 (7 held-out architectures). Nominal coverage was 0.8.

At epoch 10, held-out observation coverage is 1.000, all-seed architecture coverage 1.000, mean width 0.3942, MAE 0.0816 (7 held-out architectures). Nominal coverage was 0.8.

The intervals are broad and conservatively covering in this small holdout; universal calibration on adaptive BPSO proposals is not demonstrated. `heldout_promotion_replay.csv` compares raw early rank, prediction, uncertainty and cost scoring on disjoint held-out architectures; it is offline analysis, not simulated detection evidence or timed search. `promotion_survivor_analysis.csv` checks later scores and continuation-cost predictions for terminal survivors only, which is selection-biased.

- bpso_prediction: validation macro-F1 0.6832 ± 0.0080; mean GPU execution 28.488 s; mean scheduling/setup 0.175 s, predictor calls 0.117 s (already included in scheduling and search wall).
- bpso_prediction_uncertainty: validation macro-F1 0.6335 ± 0.0788; mean GPU execution 28.554 s; mean scheduling/setup 0.146 s, predictor calls 0.102 s (already included in scheduling and search wall).
- ua_amf_bpso: validation macro-F1 0.6335 ± 0.0788; mean GPU execution 28.764 s; mean scheduling/setup 0.160 s, predictor calls 0.100 s (already included in scheduling and search wall).

Uncertainty-minus-prediction-only macro-F1 difference: -0.0496; cost-term difference (full minus uncertainty-only): +0.0000. These component contrasts are descriptive and cannot establish a causal benefit with three seeds.


One-time predictor build cost is 0.189 wall seconds, in addition to the archive cost. Charging the complete archive to one predictive search gives 432.583 GPU seconds versus 28.763 for ordinary BPSO. Thus any online advantage must not be presented as demonstrated net end-to-end savings. No archive amortization or cross-task transfer is assumed.

## Final detector evaluation

Final models use 30 epochs and dedicated training seeds 1501/1502/1503, frozen after validation selection. The test is the already examined historical split, **not a blind independent test**. Training uses the same 8192-row pilot subset; test rows are the historical full partition.

- bpso: 3 selected search-seed architectures; mean test macro-F1 0.5416, sample SD of search-seed means 0.1520; accuracy 0.3978; normal FPR 0.9037.
- ua_amf_bpso: 3 selected search-seed architectures; mean test macro-F1 0.6259, sample SD of search-seed means 0.0946; accuracy 0.4576; normal FPR 0.8404.

Per-class recalls, precision, confusion matrices, parameter counts and measured inference latency are retained in each model directory and `results.json`. Repeated training seeds and repeated architectures are not independent datasets. Existing MLP/CNN/GRU/CNN-GRU results remain contextual, with different protocols; no matched detector-baseline superiority is inferred.

## Verification and research interpretation

485 benchmark prediction archives reproduce logged validation macro-F1; reliability endpoints have a separate audit. Epoch ledgers match summaries. The main table is generated from seed-level results; deadline overshoots, scheduler/predictor overhead, memory, unique architectures, promotions, controlled budget stops, numerical failures and retries are exported. Best/winner scores are actual epoch-30 evaluations, with strict wall cutoffs for comparisons.

The candidate combines established learning-curve prediction, forest ensembles, conformal residual calibration, cost scoring, checkpoint continuation and BPSO exploration. It is distinct in implementation from Hyperband bracket allocation, BOHB proposal models and DEHB differential evolution, but algorithmic novelty and superiority are not established. BOHB was discussed, not executed as a benchmark. Three seeds, one release, a reduced training subset, incomplete third-seed reliability, seven predictor-holdout architectures, interval width, adaptive distribution shift and archive cost prevent an IEEE superiority claim. No statistical significance or formal noninferiority is claimed.

The next justified study is an independently frozen, larger architecture calibration cohort, full-training candidate evaluations, more search seeds and matched total resources including archive amortization assumptions. Use a prespecified practical noninferiority margin and independent captures before making a detector-generalization claim. Existing historical results and unsuccessful pilot findings remain preserved.
