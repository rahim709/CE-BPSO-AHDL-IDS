# Archive-free UA-AMF-BPSO

This is an experimental candidate, not a claim of algorithmic novelty. The implementation is `src/optimization/online.py`; select `search.promotion_policy: online`. The historical, archive-based method remains available and unchanged.

## Audit and design

The offline `dynamic.py` and `predictive.py` schedulers load a serialized `CurvePredictor` bundle, whose path and SHA are configured in the historical YAML files. `curve_predictor.py` provides the loader as well as reusable feature functions. Predictor development/fit scripts read the historical learning-curve archive. Its collection cost is 403.8190207847874 GPU seconds, including held-out reliability observations; the fit/calibration subset alone is not the whole end-to-end experiment. Reusing a bundle does not make collection free. The new module imports only feature and forest-output helpers and rejects offline bundle settings. See the machine-readable `archive_dependency_audit.json` in the new results directory.

A common preprocessed training/validation view is a data artifact, not a predictor archive. New predictors read only observations generated inside their own search directory. Different methods/seeds do not share candidate checkpoints or labeled observations. Existing seed/data/training/phenotype identities govern within-search reuse, including optimizer and RNG continuation. No test arrays are supplied to search.

## Formulation

Let f_b(a,s) be validation macro-F1 after b cumulative epochs and g_q the measured incremental GPU training/validation elapsed for request q. The objective is to find a genuinely evaluated architecture with large f_30, subject to sum g_q <= C. Actual enforcement is at epoch boundaries, so the last epoch may overrun; strict-horizon quality and target checks exclude out-of-cap observations. Wall limits provide a separate safety bound. CPU fitting and scheduler overhead enter wall accounting. GPU elapsed uses the project's synchronized training/validation convention, including host stalls inside those intervals; it is not pure kernel execution or energy.

For each candidate, freeze features x_b using only the prefix through b=3 or 10: canonical architecture encodings, available validation learning-curve features, log(1+incremental elapsed) and log(1+cumulative elapsed). Do not construct early features from terminal results. Attach y=f_30 only once the architecture actually reaches 30 epochs. Hash the architecture before inspecting its target into disjoint fit/calibration/check roles (50/20/30 percent expected). Train a 24-tree, depth-4 random forest on CPU using matured fit architectures, separately for each rung. Exclude every current query architecture from fitting and calibration. A forecast for a check architecture is saved only before its terminal outcome is known.

Forest dispersion is sigma=max(tree prediction SD,0.02), not a confidence interval on one measured F1. On held-out calibration architectures compute |y-mu|/sigma; q is the ceil((n_cal+1)*0.8)-th order statistic. The empirical interval is [mu-q*sigma,mu+q*sigma] clipped to [0,1]. Adaptive proposal and promotion induce distribution shift: the split-conformal construction does not establish distribution-free coverage in this search. Coverage is measured on genuinely prequential held-out architectures, not asserted from the formula.

At least six fit architectures, four calibration architectures and four subsequently matured prequential check architectures are required per rung. A predictor must have MAE <=0.08, improve current-F1 prediction MAE by at least 10 percent and 0.005 absolute, and show Spearman >=0.3 over targets spanning >=0.02. Uncertainty modes additionally require empirical coverage >=0.75 and average interval width <=0.3; current-query intervals must also be <=0.3 wide. These are prespecified conservative engineering gates, not a statistically powered calibration certification. The most recent at most20 completed checks are used. Failure of any gate invokes ordinary SH ranking.

After a gate passes, score a candidate by

    S_b(a) = mu_30(a) + beta * radius(a) - lambda * c_remaining(a)/median(c_remaining)

where radius=min(1,q*sigma); costs are GPU elapsed seconds estimated separately from measured per-epoch training and scheduled remaining validation passes. Full mode uses beta=0.1 and lambda=0.05; prediction-only uses zero for both, uncertainty mode uses lambda=0. These generic coefficients are frozen before new search seeds and are not imported from historical development outcomes. Prospective development-only tuning is unfinished. Time features are machine/load dependent.

The optimizer starts with exactly the ordinary BPSO-SH particle order and fixed promotion quotas (eta=3). Only ranking changes after a gate passes. Terminal scores alone update personal/global bests using the existing binary velocity/position procedure. Unpromoted particles get no artificial calibrated swarm fitness. No additional restart or forced calibration job changes the SH fallback trajectory. This conservative version deliberately retains SH quotas to isolate the effect of online ranking. Model maturation consumes ordinary search training, already charged to the search; there are zero separately trained GPU archive jobs. No terminal candidate implies no reportable winner.

## Pseudocode

1. Initialize ordinary BPSO, empty observations and CPU models.
2. Evaluate each active particle to the next cumulative rung, continuing checkpoints.
3. At a lower rung, freeze prefix features; at the terminal rung, attach the observed target.
4. Form fit/calibration sets from past matured architectures excluding all current queries.
5. Fit/cache CPU forests where sample counts permit; save shadow forecasts for previously unlabelled check architectures.
6. Test the frozen earlier forecasts against newly arrived terminal outcomes.
7. If all relevant gates pass, rank by S; otherwise rank by current-rung F1 exactly as SH. Keep the same SH promotion count.
8. Update swarm attractors only with actual terminal scores, then perform the ordinary BPSO step.
9. Stop at an actual terminal target or the resource limit. Persist decisions, observations, fit identities, checkpoints and cost ledger.

## Relationship to established methods

This retains BPSO proposals and SH budgets; Hyperband schedules multiple brackets and DEHB uses differential evolution with Hyperband. An online forest plus empirical uncertainty/cost ranking is a candidate allocation policy, not an established novel optimizer. Conservative gating can leave it identical to SH. Cold-start and selected-label bias are material limitations: SH determines which terminal labels become available, and a small search may never provide enough evidence to activate prediction. Do not relax gates after inspecting benchmark outcomes to create a favorable result.

## Reproduction

From the repository root, with the existing environment and processed Gas arrays:

```bash
.venv/bin/python -m pytest -q tests/test_online_search.py
MPLCONFIGDIR=/tmp/online-mpl .venv/bin/python -m scripts.archive_free_benchmark --output results/ua_amf_online_20261108
MPLCONFIGDIR=/tmp/online-mpl .venv/bin/python -m scripts.archive_free_benchmark --output results/ua_amf_online_20261108 --diagnostic
MPLCONFIGDIR=/tmp/online-mpl .venv/bin/python -m scripts.archive_free_analyze --input results/ua_amf_online_20261108
```

The first benchmark is42 fresh runs: seven methods, three seeds, target0.6 and fixed-compute phases, each with a20 GPU-second cap and50-second wall safety cap. All use the same8192/4096 training/validation rows, eight labels, parent-training-fitted Min-Max, balanced loss,73-bit space and3/10/30 fidelities. A separate single-seed120-GPU-second run diagnoses maturation; it is not a matched baseline comparison. The analyzer re-scores saved validation predictions, verifies epoch/GPU ledgers and feature cutoffs, retains target failures as censored, and generates CSV/JSON/plots. Existing completed output is skipped; use a fresh output directory for an independent rerun. Same hardware and protocol do not imply bit-identical timing or identical resource-boundary decisions.

Actual results and limitations are generated separately after experiments complete. No new final detector test evaluation is implied by these search benchmarks. The existing historical test partition has already informed earlier experiments and is not blind independent data.
