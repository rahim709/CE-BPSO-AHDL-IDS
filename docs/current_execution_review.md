# Training-to-testing execution review — 8 November 2026

The completed work is an academic implementation and empirical evaluation study. The current [IEEE paper](latex/manuscript.pdf), [technical report](latex/implementation_report.pdf), [research report](reports/CE-BPSO-AHDL-IDS.md) and [implementation report in Markdown](reports/CE-BPSO-AHDL-IDS-Implementation-Report.md) include the new fixed-reference test results. Research-paper reproducibility and conclusion sections describe the method and findings in prose; filenames and execution commands are in the technical appendix and build instructions.

## Executed checks

| Check | Measured outcome | Evidence |
|---|---|---|
| Full CPU suite | 74 passed, no failures or skips | `latex/evidence/current_tests_cpu.xml` |
| CUDA-configured suite | 74 passed, no failures or skips | `latex/evidence/current_tests_cuda.xml` |
| All bounded workflow stages | 30/30 commands passed across original Gas and three artificial profiles | `latex/evidence/current_gpu_commands.json` |
| Integration training device | 76 recorded training trajectories used CUDA | `latex/evidence/current_gpu_summary.json` |
| Search-selected architecture to final testing | Completed on real Gas; 2-epoch integration budget | `latex/evidence/current_selected_model_metrics.json` |
| Fixed reference final evaluation | Three independently initialized models completed 200 epochs each | `latex/evidence/final_commands.json` |
| Original source and partition identities | Fresh preparation matched original checksum and all split row identities | Final-study runner checks before training |
| Extended validation prediction integrity | 94/94 records match saved predictions and hashes | `latex/evidence/extended_prediction_audit.json` |
| All available saved test predictions | 56/56 archives reproduce six checked metrics, including confusion matrices | `latex/evidence/current_prediction_audit.json` |
| Conventional validation controls | Six refits reproduce earlier scores | `latex/evidence/current_validation_controls.json` |
| Dependency consistency | No broken requirements | Executed `python -m pip check` |
| Python compilation | Passed for source, scripts and tests | Executed `python -m compileall -q src scripts tests` |

Both suites execute the complete tests. Device-aware checks use the configured device; the three component-continuation tests explicitly use CPU. Integration commands cover preparation, fixed-model training/testing, search, diagnostics, comparison of seven controls, reporting and search resume. The separate selected-model check confirms the architecture exported by search is the one subsequently trained and tested. Its small-budget collapse is recorded as a learning failure and is not a paper performance result. GPU access required execution outside the filesystem sandbox.

## Final reference results

The final protocol was recorded before generating new test predictions. It fixes Min–Max preprocessing, the CNN16–GRU32 reference, balanced loss, Adam at 0.001, batch 256, clipping norm 1, three dedicated seeds and exactly 200 epochs. All preprocessing and class weights use training only. No intermediate checkpoint or setting was selected from the new test outcomes.

| Measure | Mean | Sample SD across three training seeds |
|---|---:|---:|
| Test macro-F1 | 0.845412 | 0.001193 |
| Test accuracy | 0.724043 | 0.000373 |
| Test macro recall | 0.921640 | 0.002237 |
| Normal false-positive rate | 0.427002 | 0.000189 |

All eight classes have nonzero recall in each final run. However, normal recall is about 57.30%, and NMRI precision is about 9.44%. The large normal-to-NMRI error explains why high attack recall coexists with a 42.70% normal false-alarm rate. The documents discuss this limitation explicitly and include class precision/recall/F1, learning curves and confusion matrices.

These are fixed-reference results on the historical split, which was development-known. They are not a full-budget BPSO winner, a new blinded test, a matched-compute search comparison or a deployment claim. Training-seed variability does not establish uncertainty across independently collected datasets. The earlier ten-epoch macro-F1 of 0.299371 is retained as historical evidence; the two experiments change both scaling and training duration and cannot isolate one cause.

## Rebuild and further research

Bundled snapshots include the three final prediction archives, frozen protocol and hash, environments, seed-level metrics, earlier validation measurements and software test records. The final figures can be regenerated with `python -m scripts.build_final_evaluation_assets --snapshot`. Run `python -m scripts.build_submission_documents` for current Markdown and use the Word exporter for each document. `python scripts/compile_submission.py` rebuilds the LaTeX PDFs. `python scripts/package_submission.py` produces a review archive with per-file hashes; the actual archive is built after document checks.

Full 20-particle/30-population paper-scale search, matched-resource search-method comparisons, a complete 25-architecture long-budget cohort, independently collected/capture-separated evaluation and original ToN-IoT/Bot-IoT/SWaT experiments remain unexecuted. Synthetic fixtures support software checks only. Their scores are excluded from the research tables. The grading rubric and institutional submission template were not supplied; the package supports assessment of implementation, experimental design, reproducibility and honest result interpretation without guaranteeing a mark.
