# Historical submission review — 7 November 2026

**Superseded by the [8 November training-to-testing review](current_execution_review.md).** The current documents include three new fixed-reference 200-epoch test runs, current CPU/CUDA checks and research-style reproducibility/conclusion sections. The record below describes the earlier review; its old test counts, document lengths and unexecuted final-reference status are historical.

**Updated verdict after the LaTeX research audit: not ready for a competitive IEEE research submission.** The current review is in [REVIEW_FINDINGS.md](latex/REVIEW_FINDINGS.md), with [exact remaining actions](latex/REMAINING_ACTIONS.md), a [compiled implementation report](latex/implementation_report.pdf) and an [IEEEtran manuscript](latex/manuscript.pdf). The later experimental extension added repeated-seed neural controls, normalization studies, 200-epoch reference continuations and state-grouped validation. Full matched-resource NAS comparisons and a locked final evaluation remain outstanding. The sections below retain the earlier review record; the LaTeX package supersedes its readiness wording and document versions.

## Scope reviewed

Reviewed preprocessing and saved inference transforms, partition isolation, CNN/recurrent construction, the binary codec and BPSO update, cumulative checkpoint training, validation-only evaluation caches, search controls and compute caps, ranking diagnostics, metrics, report generation, CLI entry points, configurations, tests and existing measured results. This review and its tests provide evidence for the covered cases; they are not a proof of correctness for every architecture, dataset release or interruption scenario.

## Changes made

| Issue | Correction | Verification |
|---|---|---|
| A valid source could mask another missing file pattern | Require a match for each supplied input pattern | Regression test with one present and one missing source |
| Multiple path aliases could duplicate source rows | Resolve and deduplicate paths before reading | Regression test with direct, symlink and wildcard inputs |
| General artificial smoke metrics defaulted to real-data flags | Set `data.synthetic: true` throughout the workflow | Fresh run checks flags in metadata, all search summaries, final metrics and diagnostics |
| Fixed smoke output paths could conflict with previous runs | Create independent timestamped outputs and a latest pointer | Fresh complete integration run |
| Tied seed winners could appear to agree through index tie-breaking | Use null winner agreement for non-unique winners; save winner sets | Controlled tied-score regression test |
| Full long-budget cohort could retain a subset-bias label | Derive scope and censoring from actual cohort coverage | Complete-cohort regression test |
| Implementation report still said CUDA was unverified | Update it from saved GPU evidence and fresh CUDA tests | Prior GPU report and fresh JUnit record |
| Word export assumed exactly 23 sections and an old date | Accept a source/output path; validate actual heading count; use current export date | Export and reopen both reports |

Training defaults, published reference values and historical held-out results were preserved. No further model tuning was selected from held-out test results during this review.

## Measured verification

| Check | Result | Evidence |
|---|---|---|
| Complete CPU test suite | 70 passed in 6.38 s | `results/submission_review/tests.xml` |
| Complete suite with CUDA model/training tests | 70 passed in 9.06 s | `results/submission_review/tests_cuda.xml` |
| Dependency consistency | No broken requirements | `.venv/bin/python -m pip check` |
| Python source compilation | Passed | `.venv/bin/python -m compileall -q src scripts tests` |
| Saved-results audit | Four validation runs and eight ranking comparisons | `results/result_review/` |
| Fresh artificial integration | Seven search summaries, one final evaluation, two ranking rows | `results/software_validation/20261107T151905510387Z/` |
| Synthetic provenance propagation | Passed | Fresh metadata, seven search summaries, test metrics and ranking diagnostics |
| Historical held-out metric integrity | Recomputed metrics match archived predictions for both Gas runs | `results/submission_review/artifact_audit.json` |
| Previous real/synthetic CUDA integration | 30 successful commands on 6 November | `docs/gpu_validation.md`; historical evidence, not a fresh rerun |

The new artificial integration used 12/6/8/2/2/12/12 search-training epochs for BPSO/reduced BPSO/halving BPSO/Hyperband/DEHB/random/surrogate respectively. All selected a candidate. Random and surrogate stopped with `budget_exhausted`, as expected. These unequal resource totals must not be presented as an equal-compute performance comparison. CUDA verification required execution outside the filesystem sandbox to access the driver.

## Submission documents

- `docs/reports/CE-BPSO-AHDL-IDS.md`: research-style manuscript with abstract, methodology, protocol, measured results, limitations and references.
- `docs/reports/CE-BPSO-AHDL-IDS.docx`: editable Word export.
- `docs/reports/CE-BPSO-AHDL-IDS.pdf`: seven-page PDF export.
- `docs/reports/CE-BPSO-AHDL-IDS-Implementation-Report.md` and `.docx`: consolidated implementation report, updated for this review.
- `docs/reports/CE-BPSO-AHDL-IDS-Implementation-Report.pdf`: PDF export of the consolidated report.
- `docs/results_review.md`: detailed historical learning ablations and held-out measurements.
- `docs/paper_specification.md`: source-paper ambiguities and implementation decisions.

Generated Word/PDF exports, raw data and experiment outputs are ignored by Git. Include the desired exports and permitted evidence files separately when delivering a submission archive. A plain Git checkout includes the Markdown manuscripts and reproducibility instructions, but not the original datasets or historical checkpoints. Optional export dependency: `python-docx`.

## Outstanding scientific work and formatting

The real Gas ten-epoch reference has macro-F1 0.299371 and zero recall for codes 2 and 3. Original ToN-IoT, Bot-IoT and SWaT evaluations remain outstanding. The planned full-budget BPSO run, diverse repeated-seed architecture cohort, 200-epoch final protocol and fair-budget comparisons remain unexecuted. The primary low-fidelity-ranking hypothesis remains unverified. These are research limitations, not findings that can be filled in through editorial changes.

Add the submitting authors, affiliations, supervisor details and any required declarations, then apply the institution's or venue's template. No required venue format was supplied. The manuscript is framed around the completed implementation study so the evidence and claims remain consistent.
