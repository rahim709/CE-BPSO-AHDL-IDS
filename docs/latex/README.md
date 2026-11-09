# Course project reports

The current package presents an academic implementation and empirical evaluation study: training budgets, normalization, neural controls, predictor-state grouping and three fixed 200-epoch reference test runs. The latest execution record is [current execution review](../current_execution_review.md). A search-efficiency or competitive IEEE novelty claim still requires matched-resource NAS comparisons and independently locked evaluation; see [REMAINING_ACTIONS.md](REMAINING_ACTIONS.md). The course manuscript uses IEEEtran conference format.

Authors: **Abdur Rahim**, **Arpit Kumar Singh**, and **Gosala Buela**, IIITDM Kancheepuram.

- [Technical report PDF](implementation_report.pdf) and [LaTeX source](implementation_report.tex).
- [IEEEtran manuscript PDF](manuscript.pdf) and [LaTeX source](manuscript.tex).
- [Verified bibliography](references.bib) and [verification record](reference_verification.md).
- `tables/`: generated LaTeX tables; `figures/`: vector PDFs and 300 dpi PNGs.
- `evidence/`: exact measured summaries, archived prediction checks, environments, test record, frozen metadata, overlap audit and file-inventory snapshot.
- `build_logs/`: local pdfLaTeX and BibTeX transcripts, generated during compilation and excluded from Git.
- [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md): current scientific verdict and changes.

The licensed reference PDF, original dataset, virtual environment and historical checkpoints are not redistributed. Gas class names are verified from the original dataset paper and owner-linked report; see [gas_class_names.json](evidence/gas_class_names.json). The `time` feature is a query–response interval, not a timestamp, so sorting it cannot create chronological splits. Synthetic fixture measurements do not establish IDS accuracy. The earlier extension is validation-only, including its 200-epoch continuations. The subsequent fixed-reference study includes three final test runs on the historical random split; it is not a newly blinded or independent-release evaluation.

From the repository or extracted bundle root:

```bash
python3 scripts/compile_submission.py
```

Requirements: `pdflatex`, `bibtex`, standard TeX packages (Latin Modern, AMS math, booktabs, graphics, xurl, hyperref, listings, placeins, TikZ). IEEEtran 1.8b class/style are bundled from CTAN with their original notices. Fonts in the figures use embedded TrueType, and the report uses Latin Modern to avoid bitmap text. A compilation does not require datasets, checkpoints, Python ML dependencies, network access or the original paper. Manual equivalent, run in this directory for each document:

```bash
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

To regenerate all tables and scientific figures using the bundled measured evidence, install the repository Python requirements and run:

```bash
export MPLCONFIGDIR=/tmp/bpso-mpl
python -m scripts.build_submission_assets --snapshot
python -m scripts.build_extended_assets --snapshot
python -m scripts.build_final_evaluation_assets --snapshot
python -m scripts.build_submission_documents
python3 scripts/compile_submission.py
```

The `--snapshot` mode requires no historical `results/` directories. Without it, the generator refreshes evidence from the explicitly selected local historical experiments and fresh CPU integration record. `scripts.review_submission` runs the six conventional validation controls and audits historical predictions; that command requires original raw data, frozen processed partitions and saved historical artifacts. It refuses to overwrite a completed control record. Use `--output results/review_repeat/controls --evidence-output results/review_repeat/evidence` for a new reproduction record. Do not treat it as a test-driven tuning loop.

`python3 scripts/package_submission.py` creates `docs/submission_review_bundle.zip` with source, configurations, tests, current Markdown/Word/PDF documents, evidence and figures. Every archived file is hashed in its embedded manifest; `ZipFile.testzip()` is checked. This is an academic review archive, not a venue upload package. Its inventory snapshot describes files present during the audit; the archive manifest describes the final delivered files. Original data and complete experimental checkpoints must be obtained separately to rerun research experiments.

The extended study used `configs/research_extension.yaml`, `scripts.extend_research_evidence` and `scripts.state_grouped_validation`. Its protocols, seed-level measurements, curves and summaries are under `evidence/extended_*` and `evidence/grouped_*`; `scripts.build_extended_assets --snapshot` regenerates its figures/tables without original runs. A fresh GPU run requires CUDA access; all alternate training directories deliberately omit test arrays.

The final-reference study uses `configs/submission_evaluation.yaml` and `python -m scripts.final_evaluation_study --device cuda --output results/final_repeat`. It records the protocol before new test predictions, prepares fresh data, verifies original split identities and runs three independent initializations to exactly 200 epochs. Its snapshots are `evidence/final_*`, including the prediction archives needed to re-score the results. Research-paper reproducibility and conclusion sections use scientific prose; commands and filenames are confined to the technical appendix and this guide.

The UA-AMF-BPSO investigation is a separate, additive draft in `ua_amf_manuscript.tex`; the existing submission manuscript is preserved. Rebuild its measured pilot tables with `python -m scripts.build_ua_amf_report --input results/ua_amf_multiseed_20261108` from the repository root, then run `pdflatex -interaction=nonstopmode -halt-on-error ua_amf_manuscript.tex` twice from this directory. Its negative pilot findings do not establish optimization superiority or new independent test performance.

The current `manuscript.pdf` now includes `amf_efficiency.tex`, generated by `python -m scripts.amf_write_report` after `python -m scripts.amf_analyze`. The 25-architecture predictive investigation, timed ablations and historical-test final models are documented in `../ua_amf_prediction_study.md`. Its original manuscript/body/PDF are preserved in `results/ua_amf_prediction_20261108/manuscript_before` relative to the repository root. The older standalone `ua_amf_manuscript.pdf` remains the earlier SEM-only negative pilot draft.

The latest efficiency section is generated by `scripts.efficiency_report`, after `scripts.efficiency_analyze`. Use `docs/efficiency_optimization.md` for current commands. The 8-second pilot, exploratory charged 20-second continuation and all old studies remain separate. The previous main manuscript/source are archived under `results/ua_amf_efficiency_20261108/manuscript_before`; do not rerun older manuscript writers to replace the current section inadvertently.

The main manuscript now also includes the fresh archive-free extension from `results/ua_amf_online_20261108`. It distinguishes zero external GPU archive cost and SH-fallback savings from the unproven benefit of online prediction. The preceding main manuscript is preserved under that study's `manuscript_before/` directory. Rebuild the new report with `python -m scripts.archive_free_report`, then compile `manuscript.tex`; older report generators describe older studies and should not replace this newer extension.
