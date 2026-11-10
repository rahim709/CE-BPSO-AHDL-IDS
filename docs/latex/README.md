# Paper and technical report

The main manuscript and implementation report use the reviewed wording and the completed compute-limited comparisons. The initial pilot, exploratory continuation and fresh archive-free study remain separate. DEHB was the strongest baseline; predictive promotion was inactive in the matched archive-free trials. Full-scale comparisons and independent evaluation remain outstanding.

Authors: **Abdur Rahim**, **Arpit Kumar Singh**, and **Gosala Buela**, IIITDM Kancheepuram.

- [Main paper PDF](manuscript.pdf), [entry source](manuscript.tex), [body](manuscript_body.tex), and [efficiency section](amf_efficiency.tex).
- [Implementation report PDF](implementation_report.pdf), [entry source](implementation_report.tex), and [efficiency chapter](report_efficiency.tex).
- [Research Markdown](../reports/CE-BPSO-AHDL-IDS.md) and [implementation Markdown](../reports/CE-BPSO-AHDL-IDS-Implementation-Report.md).
- [Review record and equation-notation clarification](review_report.md).
- [Bibliography](references.bib) and [reference verification](reference_verification.md).

The manuscript-specific false-alarm and NMRI study/conclusion includes are shared with the report. `common.tex` holds shared definitions; `tables/` and `figures/` hold measured tables and visual assets. `evidence/` contains saved summaries, predictions, protocols, environment and test records. The upstream IEEEtran files retain their original notices.

Run from the repository root:

```bash
python3 scripts/compile_submission.py
python3 -m scripts.build_submission_documents
```

The first command compiles both PDFs. It needs pdfLaTeX, BibTeX and the TeX packages used by the bundled sources, including Latin Modern, AMS math, booktabs, graphics, xurl, hyperref, listings, placeins and TikZ. IEEEtran class/style files are bundled. The second command exports both current LaTeX documents to Markdown using the Python standard library. If a figure lacks a PNG preview, it uses `pdftoppm` when available. Neither command trains models, downloads data or changes saved measurements.

To compile only the implementation report, run in this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error implementation_report.tex
bibtex implementation_report
pdflatex -interaction=nonstopmode -halt-on-error implementation_report.tex
pdflatex -interaction=nonstopmode -halt-on-error implementation_report.tex
```

Use `manuscript` instead of `implementation_report` to compile only the paper. The separate top-level Overleaf project compiles `main.tex`; editing the modular sources does not automatically rebuild that package.

The Markdown synchronization reads the existing LaTeX, rather than regenerating prose from older experiments. Study-specific writers such as `scripts.amf_write_report`, `scripts.efficiency_report` and `scripts.archive_free_report` describe successive historical revisions and can overwrite later editorial changes. Asset regeneration with `scripts.build_submission_assets --snapshot`, `scripts.build_extended_assets --snapshot` and `scripts.build_final_evaluation_assets --snapshot` is separate from this editorial workflow and requires the repository's scientific Python dependencies.

Reproduction protocols and prerequisites are documented in [the main README](../../README.md), [efficiency reproduction](../efficiency_optimization.md) and [archive-free reproduction](../archive_free_ua_amf.md). Raw data, processed feature arrays and trained checkpoints are excluded from the export. Synthetic fixtures check software, not IDS performance. The `time` feature is a query–response interval, not a timestamp. Historical random-split test results are retained; they do not provide an independent-release evaluation.

The latest saved final-study records contain 110 CPU and 55 targeted CUDA tests, plus a separate five-test online CUDA record. Earlier report chapters retain the test counts recorded at their own stages. These overlapping counts must not be summed, and rebuilding documents does not rerun them. [REMAINING_ACTIONS.md](REMAINING_ACTIONS.md) and [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md) are historical review records; the linked current review and completed comparison chapter supersede their earlier pending-study descriptions.
