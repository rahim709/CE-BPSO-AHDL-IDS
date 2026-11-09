# Research diagnostics — evidence status

At initial implementation no real-data architecture-search experiment had run. Subsequently, a two-particle, one-generation, one-epoch BPSO smoke search and a two-candidate1–3-epoch diagnostic completed on the author-linked Gas Pipeline data. These tiny runs do not test the planned research hypothesis. The repository initially contained only the paper. No official dataset files were initially supplied. Gas Pipeline was later acquired directly from the author-linked university source; ToN-IoT, Bot-IoT and SWaT are still absent. Software fixtures validate behavior but cannot establish any intrusion-detection or research hypothesis. The validation report records the actual tests separately.

| Research question | Current evidence | Required measurement |
|---|---|---|
| Are early rankings unreliable? | Unknown | Repeated-seed3/10→30 correlations, overlap, regret and ties |
| Are slow-learning candidates eliminated? | Unknown | Saved predictions/trajectories, uniform rejected audit, full-cohort200 reference if making population claims |
| Do minority/difficult classes drive reversals? | Paper notes Theft/Mitm weakness; causal role unknown | Per-class recall trajectories versus training support and paired rank changes |
| Do noisy evaluations distort personal/global best? | Mechanically possible, not established | Repeated training seeds on the same candidate; best-selection disagreement and margin versus seed variation |
| Where is search time spent? | Paper states repeated training is costly | Ledger training/validation/scheduler/I/O timing, actual epochs, wall and synchronized GPU seconds |
| Do established baselines already solve it? | Unknown | Equal-time and equal-resource repeated comparisons with SH/Hyperband/DEHB/random |

The framework implements those measurements and the established controls. It does not establish an unsolved research gap. Do not describe an adaptive budget or confidence-aware swarm mechanism as novel on this evidence.

If early rankings are stable and regret is small, prioritize reduced fixed budgets, caching, efficient data loading and existing halving. If rankings are unstable but Hyperband/DEHB preserve final minority recall under the same search time, use those baselines; a new swarm policy is not justified. If a repeated residual problem persists, quantify the affected cohort, minority recall and uncertainty before selecting a mechanism. Any specific contribution then requires a targeted prior-art review in PSO/NAS and learning-curve prediction.

Full scientific conclusions, measured speedups, paper-result differences and statistical claims remain pending official data and sufficient hardware/runtime. See paper_specification.md for reproduction ambiguities and protocol.md for definitions, selection censoring and fairness limits.

The real Gas smoke diagnostic produced tied macro-F1 values and undefined Spearman/Kendall correlations; the two-epoch fixed model had zero attack recall. These observations show undertraining at the chosen tiny budgets, not a demonstrated architecture-ranking problem. A full diverse cohort,30-epoch reference, representative200-epoch continuation and repeated seeds remain necessary.

## Subsequent learning sanity audit

The initial smoke models' single-class collapse and tied rankings were audited directly from saved validation observations. Tied top-k/retention boundaries now produce non-identifiable metrics; stable index-based values are descriptive only. See results_review.md for16 fixed-budget validation checks separating architecture presets and optional training-only class weighting. A conventional larger/weighted Gas reference improved validation macro-F1 and one held-out verification, while some synthetic profiles did not benefit. This supports strengthening ordinary training controls before interpreting low-fidelity NAS comparisons. It does not demonstrate an unsolved PSO allocation problem or establish a novel optimization contribution. Both the unweighted paper-informed protocol and any weighted comparison must be reported separately and kept consistent across methods.
