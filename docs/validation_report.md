# Executed validation

Date: 2026-11-05. Initial repository: supplied paper only; no raw official datasets. Environment: Python3.14, PyTorch2.15.0, NumPy2.5.4, Pandas3.1.1, scikit-learn1.9.1, SciPy1.18.1, DEHB0.1.1. Full versions are in requirements.lock.txt and the smoke environment JSON. NVIDIA driver unavailable (`nvidia-smi` failed); computation used CPU.

Executed:

```bash
MPLCONFIGDIR=/tmp/bpso-mpl .venv/bin/python -m pytest -q
MPLCONFIGDIR=/tmp/bpso-mpl .venv/bin/python -m scripts.software_smoke
.venv/bin/python -m pip check
.venv/bin/python -m src.cli --help
python -m compileall -q src tests
```

The initial suite passed 42 tests after correcting one incorrect test expectation about Min-Max scaling of a constant one-hot column. Coverage includes Fig.5 bit mappings,100 randomized phenotype round trips per space, all recurrent kinds, short-input shape compatibility and gradients, all optimizer steps, maximizing personal/global best, exact inertia update, Eq.19 random velocity reset, RNG/swarm state continuation, split disjointness, official test extrema excluded from transform fitting, train-only column-L2, reusable saved transforms, missing-file failure, exact CPU interrupted-versus-continuous training weights, data/seed resume mismatch rejection, cache budget charging, search with test file removed, all seven methods, ranking/metric known cases, end-to-end diagnostics, final evaluation and figure export. The final expanded suite passed **53 tests in 3.69 seconds**. Added checks cover capture-group isolation, parsed chronological splits, missing-value handling, unseen-target rejection, seed-scoped caches, interrupted swarm replay, time-cap enforcement, multi-seed reliability output, independent-seed summaries, scientific-notation config round trips and malformed active encodings.

The persistent smoke run completed local preparation of90 explicitly artificial three-class software records, all seven methods, four-candidate ranking diagnostics at epochs1/2 plus a selected3-epoch extension, and two-epoch fixed-model training/test evaluation. It produced seven search summaries, one final summary and two ranking comparisons, with CSV/JSON/PDF/PNG artifacts under results/software_validation. These metrics have no intrusion-detection meaning and are intentionally not reproduced as research results here. Separate method runs used the same cap but consumed different resources because the smoke test limits generations/brackets for speed; this is not a scientific equal-compute comparison.

No real Gas Pipeline, ToN-IoT, Bot-IoT or SWaT preprocessing/training/search was performed. No25-candidate30-epoch scientific cohort,200-epoch paper baseline, full repeated-seed comparison, GPU test or measured energy experiment has run. Missing official data is the immediate blocker; full-scale runtime and memory needs remain unmeasured. No scientific hypothesis, reproduction success, significant improvement or unresolved novel research contribution is claimed.


A separate CLI comparison also completed all seven methods and seven fixed two-epoch final-training runs on the same artificial fixture:

```bash
MPLCONFIGDIR=/tmp/bpso-mpl .venv/bin/python -m src.cli compare \
  --config results/software_validation/resolved_config.json --output results/cli_validation_v2
```

Artifacts include final-result and seed-summary CSVs, confusion matrices, convergence plots and final-accuracy-versus-search-time plots. These remain software validation only. The first CLI attempt exposed YAML1.1 parsing JSON scientific notation as strings; the loader now reads JSON explicitly and accepts unquoted scientific notation in YAML, verified by a regression test. The failed attempt is retained under results/cli_validation for transparency. The successful output is results/cli_validation_v2.


## Subsequent command run with real Gas Pipeline data

The author-linked university download succeeded using the published HTTP URL. The file has97,019 rows,26 numeric predictors and eight label codes, with SHA256 recorded in gas_pipeline_source.json. This supersedes the earlier statement that no real dataset was available. Preparation, two-epoch fixed-model training, a two-particle/one-generation/one-epoch BPSO search, a two-candidate1–3-epoch diagnostic and figure/table generation all completed on CPU.

The latest test suite passed53 tests in3.97 seconds after updating the actual Gas target name (`result`), its explicit feature allowlist and inheritance overrides for other dataset profiles. The fixed tiny model obtained accuracy.63033395, macro-F1.09665718 and weighted-F1.48741044, predicting only the majority class0. Recall for every attack class was zero. These are measured real-data smoke results, not a paper reproduction or detector-quality claim. Ranking scores were tied, so Spearman and Kendall were undefined; no hypothesis conclusion follows.

Full30/200-epoch paper experiments,25-candidate diagnostics, repeated independent real-data method comparisons and GPU verification remain outstanding. ToN-IoT, Bot-IoT and SWaT local raw files remain absent.

All seven baseline commands subsequently completed on real Gas Pipeline data under tiny caps with final retraining disabled. Exact measured command results are in command_validation.md and results/gas_real_smoke/real_data_summary.json. No paper-scale performance comparison is implied.
