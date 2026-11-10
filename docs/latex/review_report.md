The 11-page PDF and its LaTeX source were reviewed together. At the author's request, the reviewed version is now synchronized with the main paper, top-level Overleaf source/package, and the manuscript PDF and modular LaTeX sources in `CE-BPSO-AHDL-IDS/docs/latex`. The implementation report and both Markdown reports have also been synchronized with the reviewed paper and latest comparisons. README build instructions and the Markdown exporter were updated. Experiment code and saved results were not changed.

**Velocity notation clarified and merged**

Before Equation (6), the text now states that $v_{ij}$ is assigned the updated velocity $v'_{ij}$, with out-of-range values replaced by independent Uniform$[0,1]$ draws. This matches the implementation and technical report. The equations and numerical results are unchanged. The clarification has been merged into the main paper, both Overleaf sources and ZIP packages, the project manuscript sources/PDF, and the research Markdown report. The implementation report already states the assignment explicitly.

**Corrections made**

- Simplified the large nonbreaking text groups that caused visibly loose spacing, while retaining protection for the reported word pairs. Enabled explicit PDF interword spaces in pdfTeX. The reported phrases were checked in the rendered PDF, including the Conclusion.
- Clarified that a fixed-quality run stops at the target **or** the compute cap, consistent with the saved protocol and censored misses.
- Changed the Section VI.D cross-reference description to “archive-free online extension” for Section VI.E. This distinguishes the fresh archive-free experiment from the exploratory continuation described in VI.C.

**Compilation, source and layout checks**

- No compilation errors, undefined citations/references, missing figure files or overfull boxes in the final build.
- All 10 cited bibliography keys resolve. The Section VI-C and VI-E cross-references resolve correctly.
- All 11 pages were inspected for layout; the 13 tables and 9 figures have no detected clipping or overlap.
- All numerical values, table bodies, equations, inline mathematical notation and citations are unchanged from the pre-review source.
- Four minor underfull-line notices and one minor underfull-column notice remain. No severe justification notice remains; column balancing on the final bibliography page is optional.
- Reference [8] was left unchanged; the bundled bibliography style does not enforce an access-date field for this technical-report entry. No access date was invented.
- The unusual “Hyberband” spelling in reference [5] matches the [official IJCAI record](https://www.ijcai.org/proceedings/2021/296), so the cited title was preserved.

**Checks against saved results**

- Every displayed row in the fresh target-or-cap and fixed-compute tables matches `results/ua_amf_online_20261108/comparison.csv` at the printed precision.
- Every displayed row in the nine-method continuation cost table matches `results/ua_amf_efficiency_20261108/followup20/comparison.csv` at the printed precision.
- At target 0.6 in the initial 8-GPU-second pilot, BPSO reached the target in 0/3 runs and SH in 1/3. The SH success was seed 601.
- The epoch-10 Spearman correlation and zero regret, and the positive epoch-3 regret for seed 43, match `results/research_extension/ranking_results.json`.
- Final-reference macro-F1, sample SD, accuracy and normal false-positive-rate macros match the saved three-seed, 200-epoch test summary.
- The 28.99% saving recomputes from the unrounded fresh comparison values. All 18 matched online trials have zero predictive promotion decisions and zero CPU fits.

These are checks of the saved results and paper artifacts; models were not retrained. The PDF and Overleaf ZIP were rebuilt and checked for consistency with the reviewed source.
