# Verified GPU-efficiency investigation

The new experiments separate fixed quality from fixed compute. All nine controls/ablations use the same FP32 runtime, 73-bit space, eight-class data view, common 30-epoch selection fidelity, and seeds 601/602/603. No test scores tune coefficients or promotions.

## Diagnosis before changes

The prior 30-second benchmark intentionally exhausted equal wall caps, so equal GPU totals did not measure computational saving. It had no target-stop condition. Continuation already preserved optimizer/RNG; deduplication already worked. Predictor calls averaged 0.10 s, while archive training cost 403.82 GPU seconds. Cost prediction median error was 1.9% on terminal survivors only. The original profile contained 319 stream synchronizations and 1251 scalar reads in three epochs; these counts include framework/optimizer activity and are not all model arithmetic.

## Engineering changes and measured preservation

Vectorized batches preserve the original RandomSampler/DataLoader generator sequence. CPU-array and one-model caches are bounded; the model cache key includes the trial path, preventing cross-experiment weight reuse. Resident tensors reduce transfers. Loss scalar synchronization is deferred to the epoch boundary. Validation runs at required 3/10/30 fidelities. Checkpoints still commit every completed epoch: less frequent writes were not adopted because they weaken hard-interruption recovery and do not directly reduce the original training/validation timer.

Across three development architectures, summed GPU elapsed fell from 25.059 to 20.182 s: **19.47%**. All final FP32 tensors were exactly equal. These are one-pass development profiles in fixed order, not a population speedup guarantee; first-use effects may remain. The interrupted last profile resumed under an explicit 30-second extension; its GPU time includes both parts, while its recorded wall field is continuation-only and is not used as full-trial wall time.

AMP was profiled and rejected: it changed weights and shifted one development macro-F1 beyond 0.01. Pinned-memory-only timing was inconsistent, so resident data was used instead. The continuation mechanism check reduced model constructions 3→1 and checkpoint loads 2→0, with exact weights. Its combined-runtime GPU times were 4.746→2.996 s; that one case does not isolate caching from other runtime changes.

## Dynamic allocation

The optimized scheduler initially probes two candidates, promotes by predicted final F1 plus a smaller uncertainty bonus minus median-normalized remaining cost, and forces another probe after at most two promotions. There is no rigid halving quota. It stops a candidate when its full calibrated upper bound is below the incumbent terminal score plus 0.01. It never prunes solely on low initial F1. Only actual 30-epoch scores update swarm attractors. Remaining cost separates training epochs from the number of future validation passes.

Beta 0.1 and cost weight 0.2 were frozen using architecture-disjoint three-fold development replay on archive IDs 0–11. Replay costs are historical proxies, not new executed timing. The sparse, timing-invariant predictor was refit from existing logs; no new GPU label archive was trained. The held-out calibration set was not used to select coefficients. Nominal 80% interval observation coverage was 13/14 at 3 epochs and 14/14 at 10, with broad mean widths 0.573 and 0.481; universal calibration under adaptive proposals is not established.

## Experiment A: fixed quality

The initial 8-GPU-second pilot executed all three targets 0.4/0.5/0.6 independently. Low standard/SH attainment motivated a declared exploratory continuation to 20 GPU seconds for the primary 0.6 target. Every previous epoch, GPU second and wall second is retained; checkpoint-copy/replay overhead is charged. This extension is not a new independent replication and no coefficients or seeds were changed. All seeds—including misses—remain in the table. GPU times below are mean actual consumption to the target or deadline; unsuccessful cases are right-censored, not discarded.

| Algorithm | GPU s ± SD | Wall s | Macro-F1 ± SD | Saving vs BPSO | Reached 0.6 |
|---|---:|---:|---:|---:|---:|
| BPSO | 17.008 ± 5.274 | 18.462 | 0.2636 ± 0.3613 | 0.00% | 1/3 |
| BPSO-SH | 12.103 ± 8.592 | 12.717 | 0.4971 ± 0.3180 | 28.84% | 2/3 |
| DEHB | 6.776 ± 2.144 | 7.373 | 0.7026 ± 0.0350 | 60.16% | 3/3 |
| Fixed prediction | 11.940 ± 8.304 | 12.883 | 0.5291 ± 0.2626 | 29.80% | 2/3 |
| Hyperband | 17.876 ± 3.802 | 19.089 | 0.5018 ± 0.2461 | -5.10% | 1/3 |
| Dynamic prediction | 12.978 ± 6.570 | 13.668 | 0.5071 ± 0.3272 | 23.70% | 2/3 |
| Full UA | 12.803 ± 7.092 | 13.647 | 0.5391 ± 0.2719 | 24.72% | 2/3 |
| UA without cost | 13.035 ± 6.524 | 13.763 | 0.5071 ± 0.3272 | 23.36% | 2/3 |
| UA without bonus | 12.800 ± 7.253 | 13.660 | 0.5391 ± 0.2719 | 24.74% | 2/3 |

Full UA saves **24.72%** versus BPSO, just below the 25% engineering goal, but is **5.78% slower than SH**. It attains 2/3 targets versus 1/3 for BPSO and 2/3 for SH. DEHB attains 3/3 with **60.16%** less GPU than BPSO. These finite-horizon, three-seed results do not establish statistical superiority or unconditional expected time-to-target.

Restricted completion GPU time uses min(T,20), retaining misses at 20; the full method saves 25.14% on this different censored estimand. It must not replace the actual-consumption result 24.72% to claim the 25% goal was met. The full tables include both measures and explicit success counts.

## Experiment B: fixed compute and ablations

All methods consume approximately 20 GPU seconds by design. These totals do not demonstrate saving; compare terminal quality and the 5/10/20-second anytime curves. Mean/SD points require all three seeds to have a terminal model; raw seed traces also preserve missing early outcomes.

| Algorithm | Macro-F1 ± SD | GPU s |
|---|---:|---:|
| BPSO | 0.2696 ± 0.3715 | 20.072 |
| BPSO-SH | 0.5030 ± 0.3231 | 20.081 |
| DEHB | 0.7356 ± 0.0402 | 20.030 |
| Fixed prediction | 0.5291 ± 0.2626 | 20.040 |
| Hyperband | 0.5018 ± 0.2461 | 20.035 |
| Dynamic prediction | 0.5071 ± 0.3272 | 20.017 |
| Full UA | 0.5391 ± 0.2719 | 20.133 |
| UA without cost | 0.5071 ± 0.3272 | 20.038 |
| UA without bonus | 0.5391 ± 0.2719 | 20.159 |

Full fixed-compute F1 is 0.5391, versus BPSO 0.2696, SH 0.5030, and DEHB 0.7356. Its gap to DEHB is -0.1965, exceeding the 0.02 preservation margin. Full UA is therefore rejected as the best quality-preserving optimizer in this study.

The uncertainty bonus changes no selected terminal F1 relative to the no-bonus ablation. The cost term improves mean F1 by 0.0320 over no-cost, but does not make full UA beat SH on GPU time. Fixed-quota prediction is faster than dynamic prediction here; dynamic allocation is not independently supported as an improvement. Bound-pruning counts and promotions are retained in decision logs; skipped work alone is not evidence of a better search.

## Predictor and end-to-end accounting

New archive GPU cost is 0; CPU development/refitting cost is recorded separately. Historical full archive cost is 403.819 GPU seconds (308.386 for the fit/calibration subset alone). Charging the full archive to one full-UA quality search gives 416.622 GPU seconds versus BPSO 17.008; **no net single-search saving remains**. Reusing historical logs is legitimate, but their computation is not free.

Smaller archive CPU refits use actual historical labels:8 fit/4 calibration architectures cost 142.656 historical GPU seconds for that subset and retained broad intervals;4/4 cost 96.386 but worsened prediction errors/widths. They were not substituted into the frozen benchmark after seeing outcomes. Lightweight online calibration was not adopted: very few terminal candidates at these budgets and adaptive sampling do not support an independently validated calibration set. No single-curve variation is described as a confidence interval.

## Final detector evaluation

Every method’s fixed 20-second winners were frozen before final evaluation. One dedicated training seed 1601 and 30 epochs are common to all methods; identical architectures share one final fit. The test is the previously examined historical split, not blind independent data. Existing MLP/CNN/GRU/reference results remain contextual.

| Algorithm | Test macro-F1 ± search-seed SD | Accuracy | Normal FPR |
|---|---:|---:|---:|
| BPSO | 0.2733 ± 0.3733 | 0.2840 | 0.8038 |
| BPSO-SH | 0.5759 ± 0.3893 | 0.5027 | 0.6180 |
| DEHB | 0.6859 ± 0.0226 | 0.5134 | 0.7549 |
| Fixed prediction | 0.5678 ± 0.2857 | 0.4958 | 0.7532 |
| Hyperband | 0.5798 ± 0.2400 | 0.4687 | 0.8086 |
| Dynamic prediction | 0.2716 ± 0.3616 | 0.1883 | 0.9442 |
| Full UA | 0.3100 ± 0.3442 | 0.2650 | 0.9442 |
| UA without cost | 0.2716 ± 0.3616 | 0.1883 | 0.9442 |
| UA without bonus | 0.3100 ± 0.3442 | 0.2650 | 0.9442 |

Precision/recall, attack-class recalls, confusion matrices, parameter counts and measured inference latency are retained per model. One final training seed and repeated historical rows limit generalization conclusions. High false-alarm rates prevent deployment claims.

## Submission assessment

The experiments are reproducible course-project evidence of engineering optimization and the need for strong controls. They do not support superior/novel UA-AMF-BPSO. DEHB is the strongest measured free control; BPSO-SH is the simpler BPSO alternative and beats full UA on fixed-quality GPU cost. Do not present the exploratory continuation as confirmatory, discard misses, or omit archive cost.

## Reproduction

See docs/efficiency_optimization.md for exact commands, source/configuration hashes, caps and dependencies. Old validated results and the previous manuscript are retained separately.
