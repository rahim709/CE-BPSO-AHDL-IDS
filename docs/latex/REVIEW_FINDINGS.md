# Current revision note — 8 November 2026

The [current execution review](../current_execution_review.md) supersedes the final-reference status in the historical audit below. Three separately initialized Min–Max reference models completed 200 epochs: mean test macro-F1 0.8454 ± 0.0012, accuracy 0.7240 and normal FPR 0.4270 on the development-known historical split. Both software suites passed 74 tests; 30 workflow commands passed. The current paper uses scientific prose in reproducibility and conclusion. Full-budget BPSO comparison and independent evaluation remain outstanding.

# Current review findings — 7 November 2026

## Completed experimental extension

The revised manuscript is titled **Budget-Constrained Hybrid Neural Architecture Search for IoT Intrusion Detection**. It now uses Abdur Rahim, Arpit Kumar Singh and Gosala Buela, with contact addresses omitted. The earlier review record below is historical; the new study fills several former gaps.

- 27 neural trajectories reached 3/10/30 epochs across three training seeds, with four prespecified reference continuations to 200 epochs.
- Reference mean validation macro-F1 at 30 epochs was 0.4267 (column L2), 0.7190 (row L2), and 0.7831 (Min--Max). Min--Max MLP/CNN/GRU controls reached 0.8325/0.8275/0.7510. These are validation means, not test scores or significance claims.
- At epoch 10, the four-preset early winner matched the epoch-30 winner in every seed. Regret was zero in all three seeds; the three epoch-3 regrets were 0, 0.0284 and 0. This small cohort did not establish a severe early-selection problem.
- Column-L2 reference continuations reached mean validation macro-F1 0.8390 at epoch 200 (three seeds); the one Min--Max continuation reached 0.8407. They were continued trajectories, not independent final-seed test evaluations.
- A separate state-grouped split had zero shared non-time state groups and evaluated six tabular fits plus three hybrid runs. Mean macro-F1 was 0.8372 for forest, 0.8018 for logistic regression and 0.7559 for the hybrid. Capture-level and temporal independence remain unverified.
- High normal-class false-positive rates remained. The Min--Max reference's mean FPR at 30 epochs was 53.24%; the MLP's was 42.17%. These observations do not justify deployment.
- All 94 new validation measurement records matched saved predictions. The expanded CPU suite passed 74 tests. Historical held-out predictions and scores were preserved.

The main remaining gaps are full matched-resource NAS comparisons, the full diverse architecture cohort, a locked final evaluation and broader original datasets. See `REMAINING_ACTIONS.md` and the extended-study chapter of the report.

## Earlier review record

The implementation addresses the stated software workflow within explicit reproduction choices. The experimental evidence does **not** establish the intended search-efficiency or ranking contribution. It is not ready for a competitive IEEE research submission. This assessment supersedes the earlier Markdown/Word readiness wording.

The review inspected all source modules and experiment configurations, the supplied IEEE PDF, test and CLI scripts, and saved logs/measurement artifacts. A checksum inventory recorded 6,136 files at audit time. Prediction integrity was assessed for 47 archived held-out prediction files; it was not an exhaustive proof of all checkpoints or all possible architectures.

| Finding | Verified evidence and implication |
|---|---|
| Neural learning was weak | The real Gas ten-epoch balanced reference had validation macro-F1 0.228693 and historical test macro-F1 0.299371; CMRI (2) and MSCI (3) had zero test recall. |
| Ordinary comparisons were missing | Six new validation-only controls completed. All-feature random forest/logistic regression scored 0.818609/0.829630 macro-F1. No new baseline test result exists. |
| Aggregate scores hid operational errors | Those all-feature controls falsely flagged 44.38%/42.17% of configured normal validation rows. No deployment claim is justified. |
| Random row separation did not establish independence | All 26 features had zero predictor hash overlap. Removing `time` exposed 2,033 shared train–validation hashes, 2,087 train–test hashes, 1,448 validation–test hashes and 105 predictor hashes with conflicting corpus labels. Counts are unique hashes, not rows. |
| Time removal did not solve dependence | Without-time forest/logistic macro-F1 was 0.843997/0.829657. These remain random-split validation scores with repeated states. |
| Early-ranking hypothesis was unverified | Existing tiny architecture cohorts tied; no full diverse repeated-seed 3/10/30 or full 200-epoch study completed. |
| Search-method checks were not fair performance comparisons | Fresh fixture search epochs were 12/6/8/2/2/12/12 across BPSO/reduced/halving/Hyperband/DEHB/random/surrogate. Shared caps did not equalize consumption. |
| Reproduction was incomplete | Source splits, seeds, metric averaging and several architecture conventions were unavailable; only original Gas data was present. |
| Novelty was unsupported | BPSO architecture search derives from the supplied paper. Weighting, halving, Hyperband, DEHB and surrogate proposals are established controls. No new policy was validated. |

Changes in this pass:

1. Added predictor-only overlap, conflicting-label counts and all three partition comparisons to preprocessing diagnostics; added a contradictory-label regression test.
2. Marked a remaining artificial CLI validation configuration with `synthetic: true`.
3. Added `scripts/review_submission.py` for explicitly fixed conventional controls and artifact auditing. The initial attempts encountered parameter-serialization errors; the exporter was fixed and the completed six-control record is the one included. No failed attempt is reported as a completed experiment.
4. Added evidence-based table/figure generation, offline LaTeX compilation and a review-archive builder.
5. Prepared complete report/manuscript LaTeX sources and a separately verified bibliography. Earlier working edits and historical data/checkpoints were preserved.

Measured checks: **71 CPU tests passed in 7.98 s**; dependency consistency passed; Python compilation passed; fresh artificial integration completed all seven search paths and reporting; all six audited metrics matched 47 historical prediction files. Historical 70-test CUDA validation is distinct from this pass; the added test was not rerun on CUDA.

The six controls used seed 42, one frozen split, no hyperparameter sweep, and no new test model evaluation. The review inspected raw held-out labels for overlap/class-coverage audits and re-scored historical predictions, so it was not test-blind. New control fitting and setting choices did not use those labels or historical test scores. Random forest used 100 trees and balanced weights; logistic regression used balanced weights, 1,000 maximum iterations and train-fitted standard scaling. No convergence warning was recorded. Compute was not equalized against neural training; these are descriptive controls, not statistical or search-efficiency comparisons.

The source bibliography was verified against the local paper and primary publisher/author pages, including [Hyperband](https://www.jmlr.org/papers/v18/16-558.html), [DEHB](https://ijcai.org/proceedings/2021/296), [BOHB](https://proceedings.mlr.press/v80/falkner18a.html) and the [dataset owner catalogue](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets). The dataset owner warning motivates caution; it does not prove the cause of any score here.

Follow-up verification resolved the class-name uncertainty using the original Morris–Gao paper, Table VII, and the owner-linked Hsu–Mudd–Thornton report, Table 1. Tables and plots now display the verified names, with original codes retained. The original paper also defines `time` as the query–response interval. The historical sorted-coverage result is therefore interval-sorted, not chronological. Author names were inserted in supplied order; the latest revision uses Gosala Buela and omits contact addresses. These presentation and interpretation corrections did not change saved training labels, model checkpoints, splits, predictions or scores.
