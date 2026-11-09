# UA-AMF-BPSO implementation and research audit

## Audit (2026-11-08)

The existing implementation is retained. `src/optimization/encoding.py` has 73 paper-space bits: 10 global bits, 40 CNN-slot bits, one RNN-count bit, and 22 RNN-slot bits. Only active slots decode; canonical decoded dictionaries deduplicate inactive-slot aliases. The debug space is different and must not be described as the paper space. `HybridIDS` repairs infeasible short-sequence convolution/pooling shapes; that engineering choice and the linear recurrent option are documented interpretations, not an exact author implementation.

`bpso.py` implements maximizing binary PSO, sigmoid sampling and the paper-informed random velocity replacement. `search.py` implements BPSO, minimum-budget BPSO, BPSO successive halving, random search, random-forest proposals, Hyperband brackets and the upstream DEHB adapter. The RF method is an established acquisition control, not a novel algorithm. All use the shared validation-only evaluator. Existing BPSO-SH updates attractors using terminal-rung observations only. The highest actually attained fidelity is used in legacy summaries, and incomplete maximum-fidelity selections are explicitly marked.

`train_to` stores model, optimizer, Python/NumPy/Torch/CUDA and DataLoader generator states every completed epoch. It loads train/validation only. Exact interrupted-versus-continuous weights are tested on CPU; CUDA determinism remains environment-dependent. `Evaluator` keys checkpoints by decoded architecture, training configuration, seed and dataset fingerprint. Resource ledgers charge incremental completed epochs, synchronized train/validation time, and total evaluation wall time; search wall includes scheduling. A deadline is checked at epoch boundaries, so an epoch can overrun a time limit. Partial failed epochs consume wall time but are not counted as completed epochs. Committed unlogged epochs are recovered from checkpoints/trajectories. Uncommitted partial work and setup wall time after a hard process kill cannot be recovered exactly; graceful exceptions and epoch-boundary continuation are tested. Cross-run checkpoint sharing is deliberately avoided to prevent uncharged compute and protocol contamination.

A read-only inventory of source/configuration/tests, raw/processed datasets, experiments, logs, checkpoint files, reports and historic search summaries is saved in `results/ua_amf_validation/repository_audit.json`. This is a structural/evidence audit, not a rerun of all historic experiments. The original local IEEE paper was text-extracted and reviewed; it uses 30-epoch search and 200-epoch final training, with population 20 and 30 generations for Gas Pipeline. Existing manuscript/evidence files report bounded software checks, preset learning studies and historical-test final reference evaluations. These do not constitute a matched-budget adaptive NAS study. Prior final-test outcomes are already known and cannot become a fresh blind test. Three-seed reference results and false-alarm studies are preserved and cannot be attributed to the new scheduler.

Reproducible software evidence includes configuration guards, seed-scoped cache tests, exact epoch continuation, validation-only selection and seven legacy search paths. Existing experiment commands/configuration, predictions, source hashes and evidence exports support inspection/reruns within their recorded environments. Large full-space 3/10/30 adaptive comparisons, complete common-budget candidate cohorts, independent-capture testing and causal component evidence are missing. No statement of full paper reproduction or operational detector superiority is justified.

## Mathematical specification

Let x be a binary string in {0,1}^73 and a=D(x) its decoded architecture. For fixed train/validation partitions, Y(a,s,b) is validation macro-F1 after b cumulative epochs with independent training seed s. The target is to maximize E_s[Y(a,s,B)] under total completed-epoch, wall-time and GPU-execution constraints. This objective is conditional on the available data split, not prospective population performance.

At rung b, evaluate every active architecture using K distinct seeds. For uncertainty-enabled allocation K>=3. Compute

    m(a,b) = (1/K) sum_s Y(a,s,b)
    SE(a,b) = sqrt(sum_s (Y(a,s,b)-m)^2 / (K*(K-1)))
    c(a,b->b') = K*(b'-b)*mean_observed_epoch_seconds(a,b)
    A(a,b) = m(a,b) + beta*SE(a,b) - lambda*c(a,b->b')/median_active(c).

The estimator uses independent initializations, not correlated epoch values. SE is a Monte Carlo estimate of seed-mean sampling variability conditional on this validation split. With K=3 it is noisy; beta*SE is an exploration bonus, **not a calibrated confidence interval**, future-fidelity error bar, or guarantee. No single-seed CI is constructed. No future-F1 extrapolation is used: promotion assumes current-rung performance is informative. Late-learning architectures can still be rejected. Cost extrapolation assumes locally stationary epoch cost; realized time, including prediction/scheduling overhead and replication, determines reported savings.

Promote the largest max(1,floor(n/eta)) acquisition scores at each rung; ties preserve candidate order. beta=0 removes uncertainty; lambda=0 removes cost-sensitive ranking. This is cost-aware ranking within fixed rung/quota structure, not an adaptive Hyperband bracket-width rule. Before each replicated job reserve all needed incremental epochs. If the remaining cap cannot accommodate that job, terminate; there is no hidden overspending on uncertainty measurements. Rungs, replication offsets, weights and restart fraction are configurable. Finalists and swarm attractors use only complete K-seed means at B. Partially replicated or lower-fidelity candidates cannot become the proposed method's reported winner.

For particle i, maintain personal best p_i and global best g using those B-fidelity means. Apply the existing update

    v_i <- w*v_i + c1*r1*(p_i-x_i) + c2*r2*(g-x_i)
    x_ij <- Bernoulli(sigmoid(v_ij)).

The existing out-of-bound velocity replacement is unchanged. Restart ceil(rho*N) worst-personal-fitness particles with independent uniform bit strings each generation and clear their personal fitness. This preserves exploration but is an ordinary diversity mechanism. Deduplication is by D(x), with all aliased particles receiving the same terminal score. Scheduler replay reconstructs positions/RNG deterministically and reuses persisted evaluations; time caps can change where a replay stops because replay has real overhead. Wall time accumulates across invocations.

The GPU cap conservatively charges complete CUDA evaluation wall time, including failures; an epoch-boundary overrun remains possible. Reported GPU time is synchronized device-resident train/validation elapsed time (includes Python gaps), not kernel-profiler time or energy. Peak allocated GPU memory and actual parameter counts are logged. CPU pilot GPU time/memory are zero. A 4 GB device can run suitable small candidates, but the full unconstrained space is not guaranteed to fit; OOM is recorded as failure. Serial candidates, one DataLoader worker process (num_workers=0), and no cached live models limit resident memory. No silent space restriction, AMP substitution or batch-size alteration changes the searched architecture.

## Pseudocode

```
initialize seeded binary swarm
for generation:
    canonicalize/deduplicate decoded swarm candidates
    for cumulative rung b:
        reserve all K-seed incremental epochs
        continue each seed's model/optimizer/RNG checkpoint to b
        compute seed mean, SEM and continuation-cost prediction
        if b < B: retain best floor(n/eta) acquisition scores
        else: update particle fitness using K-seed mean
    update personal/global best only with B-fidelity means
    apply original BPSO velocity and position update
    restart configured fraction of worst particles
return best actually evaluated B-fidelity replicated architecture
```

## Relationship to established methods

Ordinary BPSO-SH ranks single-seed performance with fixed halving; the candidate adds charged seed replication, a SEM exploration bonus, continuation-cost ranking and optional swarm restarts. Hyperband varies successive-halving brackets and randomly samples configurations. DEHB combines differential evolution proposals with Hyperband-style resource allocation. The candidate retains BPSO proposals and one configured rung ladder. These ingredients are established ideas; their composition is an investigational engineering candidate, **not an established novelty claim**. No literature-wide novelty clearance has been completed.

Primary sources: Li et al., *Hyperband*, JMLR 18(185), 2018, https://jmlr.org/papers/v18/16-558.html; Awad, Mallik and Hutter, *DEHB*, IJCAI 2021, upstream paper/software record https://github.com/automl/DEHB. The original BPSO paper is the local IEEE PDF listed in the audit.

## Protocol and interpretation

The automated pilot uses fixed stratified subsets of **already training-fitted** full Gas Pipeline train/validation arrays. It preserves all eight labels, the parent preprocessing and full 73-bit space. Indices, class supports and source hashes are saved. The pilot scaler has seen parent training rows outside the reduced training subset; this is disclosed and shared across methods. No validation or test rows fit preprocessing, and no historical test array is opened. This is reduced-row evidence, not a full-data result.

All search variants receive the same 24-completed-epoch cap and 60-second wall cap, and run serially on the same CPU. Epoch caps are inclusive of replicate training. Hardware heterogeneity across architectures means equal epochs do not mean equal time; both are reported. Variant order is fixed, so thermal/load/order effects remain possible. At two epochs, performance may be near a constant predictor and is unsuitable for claims about 30-epoch learning. Common assessment seeds 90042/90043/90044 evaluate only selected architectures at two epochs on validation, separately from search. These assessment costs are saved and excluded from search-time savings; they are not a held-out generalization test or an independent dataset replicate.

Ablations include ordinary BPSO, single-seed BPSO-SH, replicated fixed halving, uncertainty-only ranking, budget-only ranking and the full candidate. The three replicated scheduler comparisons use the same seed offsets. Full versus budget-only changes uncertainty alone; uncertainty-only versus replicated fixed changes uncertainty alone without restarts. Full versus uncertainty-only changes both cost and restart, so diversity's isolated causal contribution is unmeasured.

Common-reference selection regret is null unless every compared candidate has a common terminal reference. It cannot be inferred from only survivors. Normal FPR, per-class attack recall, confusion matrices and parameters are derived from saved validation assessments. MLP/CNN controls are validation-only fixed two-epoch controls, not final detector baselines. No new final test training is justified if the feasibility gate fails. A final architecture/protocol has not been frozen for a submission-quality full study.

Savings are 100*(measured baseline search wall - measured proposed search wall)/measured baseline search wall. Negative savings are reported. Search-time/F1 plots and seed-level tables describe actual CPU pilot measurements; no statistical significance is inferred from tiny seed counts, and no preselected 30--50% target is used.

## Reproduction

From the repository root:

```bash
.venv/bin/python -m scripts.audit_ua_amf
.venv/bin/python -m pytest -q --junitxml=results/ua_amf_validation/tests_cpu.xml
.venv/bin/python -m scripts.ua_amf_study --output results/ua_amf_pilot_20261108 --seeds 42 --device cpu
# Completed three-seed feasibility study:
.venv/bin/python -m scripts.ua_amf_study --output results/ua_amf_multiseed_20261108 --seeds 42 43 44 --device cpu
# Full-data 3/10/30 search, bounded but NOT yet executed:
.venv/bin/python -m src.cli search --config configs/ua_amf.yaml --output results/ua_amf_full_seed42 --seed 42
```

The last command uses auto device, a 540-epoch / 1800-second cap and 1500 GPU-second cap; it is a planned follow-up, not completed evidence. For a 4 GB CUDA machine, first run `BPSO_TEST_DEVICE=cuda .venv/bin/python -m pytest -q tests/test_adaptive.py tests/test_training_search.py` and inspect memory/failures before increasing resources.

## Completed evidence and submission status

The final suite and experiment counts are recorded in `results/ua_amf_validation/FINAL_SUMMARY.json`. The CPU multi-seed study completed 27 searches (nine variants, seeds 42/43/44); a separate CUDA pilot completed nine searches at seed 42. The initial one-seed CPU study is preserved separately and is not pooled into the three-seed estimates. Each search charged 24 epochs. Common validation assessments use three dedicated initialization seeds per selected architecture; costs and predictions are retained separately.

The three-seed CPU full method averaged 12.4655 seconds, compared with BPSO 10.8201 seconds: saving -15.2072%. Full-method assessment macro-F1 was 0.096645, compared with BPSO 0.069544 and replicated fixed/budget-only controls 0.096645. These are weak collapsed predictors, not preserved competitive detectors. The one-seed CUDA full method took about 1.146 seconds versus BPSO 0.857 seconds (saving approximately -33.8%). Peak full-method GPU allocation was 528.41 MiB. Exact values, SDs, epoch counts, architecture counts, promotions and overheads are in the seed CSV/JSON and comparison tables. CPU timing overlapped briefly with the CUDA jobs, limiting causal timing interpretation.

A separate complete three-architecture, three-seed diagnostic cohort reached cumulative 3/10/30 epochs, charging 270 epochs, 12.7629 wall seconds and 8.4760 synchronized GPU seconds. All nine 30-epoch models predict a single class and tie at macro-F1 0.096645. Regret is zero in this small cohort because terminal performance ties; correlations are undefined/uninformative. This does not establish successful ranking or generalize regret to the search universe. Saved predictions at all 27 endpoints reproduce the trajectory scores.

There is no measured performance advantage attributable to uncertainty and no useful-performance search-speedup claim. The method is not ready to support an IEEE optimization-superiority submission. The next experiment should establish noncollapsed full-training candidate learning, retain the eight-class setup and training-fitted preprocessing, and then freeze a full-data candidate cohort and matched-GPU-time study. Existing full-data normalization/loss-control evidence can motivate prespecified follow-ups, but the current protocol must not be retroactively changed or its negative results discarded. New final detector training and historical-test evaluation were not performed for the proposed method.

Research artifacts: `results/ua_amf_multiseed_20261108/REPORT.md`, its seed-level results, comparison/ablation table, promotion analysis, validation controls and assessments, `results/ua_amf_cuda_pilot_20261108/REPORT.md`, and `results/ua_amf_cohort_20261108/cohort_audit.json`. The separate IEEE-style draft is `docs/latex/ua_amf_manuscript.tex`; the historic manuscript and PDFs are retained unchanged by this task.

To reproduce the completed CUDA pilot and diagnostic (GPU access may require leaving the sandbox):

```bash
MPLCONFIGDIR=/tmp/ua-amf-mpl .venv/bin/python -m scripts.ua_amf_study --output results/ua_amf_cuda_pilot_20261108 --seeds 42 --device cuda
MPLCONFIGDIR=/tmp/ua-amf-mpl .venv/bin/python -m src.cli diagnostics --config configs/ua_amf_cohort.yaml --output results/ua_amf_cohort_20261108
MPLCONFIGDIR=/tmp/ua-amf-mpl .venv/bin/python -m scripts.build_ua_amf_report --input results/ua_amf_multiseed_20261108
MPLCONFIGDIR=/tmp/ua-amf-mpl .venv/bin/python -m scripts.build_ua_amf_report --input results/ua_amf_cuda_pilot_20261108
MPLCONFIGDIR=/tmp/ua-amf-mpl .venv/bin/python -m scripts.build_ua_amf_cohort
```

Existing search-summary directories are reused by the pilot runner to avoid charging a second experiment; choose fresh directories to rerun timing. The cohort configuration explicitly points to the CUDA pilot's saved reduced-row data. Planned full-data matched-time command (not executed):

```bash
.venv/bin/python -m src.cli compare --config configs/ua_amf_matched.yaml --output results/ua_amf_full_matched
```

The full-method common validation assessments predicted only the normal class: mean normal FPR was 0, but recall of every attack class was also 0. This is detector collapse, not operational success. The final verification passed 90 CPU tests and 35 CUDA-targeted tests on the 4 GB GPU. The additive IEEE draft compiled successfully to `docs/latex/ua_amf_manuscript.pdf`.
