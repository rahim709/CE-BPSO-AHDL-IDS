# Final result files

The folders retain their experiment names so the analysis and graph scripts can use the original paths. Supporting studies are included where they contribute measurements to the final paper.

| Folder | Contents |
|---|---|
| `verification/` | Export checks, fresh CPU test XML and links to saved CPU/GPU records |
| `../plots/final/` | Rebuilt detector, target-cost, fixed-compute and anytime comparison plots |
| `nmri_tradeoff/` | Final specialist predictions, thresholds, protocols and measurements |
| `false_alarm_optimization/` | Weighted-hybrid validation and final evaluation |
| `submission_final_evaluation/` | Three fixed-endpoint hybrid reference evaluations |
| `ua_amf_online_20261108/` | Final archive-free comparison, CPU/GPU records, prediction and compute-limit audits |
| `ua_amf_efficiency_20261108/` | Execution profiles, target/fixed-compute comparisons and charged continuation |
| `ua_amf_prediction_20261108/` | Predictor development and reliability records |
| `ua_amf_prediction_benchmark_20261108/` | Matched predictive-allocation comparisons |
| `ua_amf_prediction_final_20261108/` | Historical-test final evaluations for the earlier search comparison |
| `research_extension/`, `state_grouped_validation/`, `learning_sanity/` | Supporting controls used in the reports |

Prediction archives, trajectories, CSV/JSON summaries and graph inputs are included. Training checkpoints, raw/processed feature arrays, serialized models, source backups and debug-only runs are excluded. Original artifact hashes describe the historical full runs; `SUBMISSION_MANIFEST.json` describes this exported subset.

From the repository root, regenerate final comparison plots with `python -m plots.generate`. The original `scripts.archive_free_analyze` and `scripts.efficiency_analyze` also run on the included result records without model checkpoints or raw feature arrays. Retraining or inference requires separately prepared official data and trained models.
