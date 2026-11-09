# Learning-curve UA-AMF-BPSO: implementation and bounded validation

This extends the earlier SEM-only UA-AMF candidate. It preserves that path, its negative experiments, standard BPSO, all historical detector results, and the original 73-bit CNN/RNN search space. It adds final-fidelity prediction and measured architecture-disjoint uncertainty calibration. The main manuscript is backed up under `results/ua_amf_prediction_20261108/manuscript_before` before the new section is inserted.

## Audit and resolved gaps

The repository already contains the paper-space codec, maximization BPSO, reduced BPSO, successive halving, random search, RF proposals, native Hyperband and an upstream DEHB adapter. The training engine saves optimizer and Python/NumPy/Torch/CUDA/DataLoader random states each completed epoch. Evaluator caching uses decoded architecture, seed, training settings and dataset identity; inactive-slot aliases share a checkpoint. Training and search use train/validation arrays only. Existing column-L2, MinMax, conventional controls and historical-test studies remain contextual evidence, not new optimizer results.

The previous UA method used repeated-seed SEM rather than future-fidelity prediction. The new `curve_predictor.py` builds a forest predictor from architecture-disjoint groups and uses calibration-architecture residuals. `predictive.py` supplies a separate scheduler selected by `search.promotion_policy: learning_curve`. With no such option, existing SEM behavior is retained. The only legacy scheduler extension is an optional explicit rung list, so BPSO-SH can use precisely 3/10/30; its old derived ladder remains the default. `bpso.py`, architecture encoding and model definitions are unchanged.

Full file inventory, source hashes, configurations, protocols, metrics, seeds and environment are retained in the new study directories. This is an inventory plus targeted implementation/evidence audit, not a rerun of every prior experiment or certification of all historical claims.

## Frozen reliability pilot

Before current measurements, the protocol fixed 25 distinct uniformly sampled decoded architectures from the paper space (cohort seed 271828), seeds 42/43/44 and cumulative budgets 3/10/30. The row subset is fixed stratified sampling (seed 39173): 8192 parent-training and 4096 parent-validation rows, all eight labels. It uses the existing MinMax arrays and balanced training loss with norm-1 clipping. These choices come from earlier validation evidence, not the current outcomes. Parent training-only preprocessing includes training rows outside the reduced subset; this is shared across methods and disclosed. No validation/test rows fitted transforms or class weights.

The cap is 2250 completed epochs, 480 wall seconds and 420 conservatively charged GPU seconds. GPU enforcement charges complete CUDA evaluation wall, including setup/failure overhead; separately reported GPU execution is synchronized training/validation elapsed time rather than kernel-profiler time. The study stopped at this fixed cap: **61/75 terminal trajectories**, 1830 completed epochs, 420.7223 search wall seconds and 403.8190 synchronized GPU execution seconds. Seeds 42 and 43 complete all 25 architectures; seed 44 completes indices 0--10. Fourteen third-seed trajectories remain unexecuted. The third-seed prefix is cost-censored and must not be pooled as a complete 25-candidate replicate.

The first complete seed gives Spearman correlation 0.3632 from 3 to 30 and 0.9134 from 10 to 30. Corresponding selection regret is 0.13705 and 0.01991. Other seeds, Kendall correlation, tie-aware top-k recall and regret ranges are in `reliability_summary.json`. Undefined selection quantities remain null when ties prevent identification. All common-reference scores are actual trained endpoints. Rejected search candidates outside this cohort do not have measurable global selection regret.

## Predictor and uncertainty construction

Architecture IDs are partitioned before targets are read: indices 0--11 fit, 12--17 calibration, 18--24 held out. All training seeds for one architecture stay in one group. Only completed terminal trajectories enter fitting/calibration/evaluation; completion counts are reported. At rung b, features contain the canonical 73 bits, log parameter count, current epoch, observed F1 last/mean/max/min/SD/slope, current loss and log mean observed epoch execution seconds. The extractor filters rows to epoch <= b. It never includes the architecture's later loss, later F1 or later cost.

Two 64-tree random forests (one for b=3, one for b=10; minimum leaf size two) predict the seed-specific 30-epoch validation macro-F1. The architecture count is small relative to feature dimension. Forest settings are fixed, not selected on held-out targets. Query features include the current training trajectory; no IDS test information enters the model.

Let mu_b(z) be the tree-mean prediction and s_b(z)=max(0.02, SD of tree predictions). For calibration architecture a, take the largest normalized absolute residual across its completed seeds:

    R_a = max_s |Y(a,s,30) - mu_b(z(a,s,b))| / s_b(z(a,s,b)).
    k = ceil((n_cal+1)*(1-alpha)), alpha=0.2.
    q_b = kth order statistic of calibration-architecture R_a.
    u_b(z) = min(1, q_b*s_b(z)).
    I_b(z) = [max(0,mu_b-u_b), min(1,mu_b+u_b)].

If the corrected quantile exceeds the available calibration count, the radius is conservatively uninformative. The six-architecture calibration set gives a coarse, conservative quantile. This is a group split-conformal construction; its exchangeability assumptions and the change from uniformly sampled architectures to BPSO proposals prevent claiming a universal coverage guarantee. In particular, empirical coverage on seven held-out architectures is not proof of calibration on adaptively selected architectures.

Held-out architecture coverage is 7/7 at both rungs (14 seed observations); mean interval widths are 0.6992 at epoch 3 and 0.3942 at epoch 10. MAE is 0.1261 and 0.0816 respectively. Broad intervals cover many outcomes and can reward uncertain poor candidates. Full predictions, coverage, widths, errors and partition IDs are saved. Mutating held-out targets cannot change fitting or calibration (tested). Every development-archive architecture is excluded from new predictive search queries to prevent use of its known terminal target.

## Acquisition, resources and swarm updates

The conditional target is to maximize expected 30-epoch validation macro-F1 subject to measured wall and device budgets. It is not an independent-population test objective. At a current rung, use

    S(a,b) = mu_b(a) + beta*u_b(a) - lambda*c_hat(a,b->30)/60.
    c_hat = (30-b)*mean(observed epoch training+validation seconds through b).

F1 and radius are dimensionless on [0,1]. Cost is measured in seconds and divided by the fixed unit of 60 seconds. beta=1 and lambda=0.05 are protocol defaults frozen before current outcomes; the ablations set one or both to zero. They are not optimized against held-out architectures or IDS test data. Cost prediction assumes local epoch-time stationarity and is checked against actual continuation cost for survivors.

The fixed rung ladder is 3/10/30. At each rung keep max(1,floor(n/eta)), eta=3, by acquisition score. Thus the method adapts which candidates receive continuation, not the quota or bracket design. All training resumes from the exact model/optimizer/RNG checkpoint. Only actually observed epoch-30 F1 updates swarm personal/global best; predictions and lower fidelities cannot become attractors or reported winners. Apply the existing BPSO velocity update, sigmoid sampling and random velocity replacement. Restart 20% of worst-personal-fitness particles in every predictive variant. The three predictive ablations differ only in their scoring terms; diversity is not independently ablated here.

Pseudocode:

```
fit rung-specific predictor on fit architectures
calibrate residual scales on separate calibration architectures
evaluate intervals on held-out architectures; freeze model and hash
initialize seeded binary swarm
repeat while measured compute remains:
    deduplicate decoded positions and exclude archive IDs
    evaluate candidates at 3 epochs
    predict 30-epoch F1 and radius from current features
    retain top acquisition candidates; continue checkpoints to 10, then 30
    update swarm attractors only with observed 30-epoch scores
    apply existing BPSO update and common predictive-variant restarts
return best observed terminal candidate within the specified wall limit
```

Serial evaluation limits resident memory. The full paper space is retained; no silent width/batch restrictions guarantee a 4 GB fit. GPU OOMs and nonfinite loss are logged and excluded from winners. Deadlines are checked at epoch boundaries, so one epoch plus final persistence can overrun. Comparisons at strict 10/20/30 seconds use only terminal observations completed within each limit. Partial final training still consumes measured cost. Hard process kills can lose uncommitted partial-epoch wall time; committed epochs are recoverable. Scheduler replay reuses cached training but adds real wall overhead. Search from a fresh output directory is required for clean timings.

## Matched-time benchmark and ablations

The protocol fixes search seeds 501/502/503, per-run wall cap 30 seconds and GPU charge cap 30 seconds, population six, high iteration ceilings, maximum fidelity 30, and identical data/training/hardware. Methods run serially in a seeded random order, with no concurrent GPU experiments. The controls are standard BPSO, ordinary BPSO-SH, random, Hyperband and DEHB. Predictive ablations are prediction-only (beta=lambda=0), prediction+uncertainty (beta=1, lambda=0), and full prediction+uncertainty+cost. Hyperband and DEHB retain their native bracket mechanisms.

Anytime comparisons are measured at strict wall limits 10/20/30 seconds. Missing terminal candidates remain missing, not zero. Time-to-target uses prespecified F1 targets 0.2/0.4/0.6. Unattained targets are censored at 30 seconds; conditional hit times and hit counts are reported separately. A paired time saving is computed only for seeds where both methods attain the same target. Three search seeds do not establish statistical superiority.

Reported resource fields include full search wall, synchronized GPU execution, completed epochs, unique architectures, promotions, allocated peak memory, failures, predictor load/inference and scheduling overhead. Predictor fitting and the reliability archive have separate ledgers. Unamortized predictive totals add the entire 403.8190 GPU seconds and 420.7223 wall seconds of archive construction plus predictor-build wall time. Online savings alone do not establish net end-to-end savings; reuse/amortization is an explicit further assumption, not free training data.

## Relationship to prior work and limitations

Successive halving already allocates incremental budgets; Hyperband explores bracket tradeoffs; BOHB combines model-guided proposals with Hyperband; DEHB uses differential evolution with Hyperband. Freeze-thaw optimization already uses learning curves, and split conformal prediction is an established uncertainty construction. Here proposals remain BPSO, scoring uses a fixed forest/calibrated residual scale and measured cost, with a single fixed ladder and terminal-only attractors. This combination is a research candidate, not an established novel algorithm or a literature-wide novelty finding.

Primary sources: [Hyperband](https://jmlr.org/papers/v18/16-558.html), [BOHB](https://proceedings.mlr.press/v80/falkner18a.html), [DEHB](https://ijcai.org/proceedings/2021/296), [Freeze-Thaw](https://lips.cs.princeton.edu/pdfs/swersky2014freeze.pdf), and [Angelopoulos and Bates on conformal prediction](https://arxiv.org/abs/2107.07511). The original local IEEE BPSO paper was retained and reviewed in the earlier audit.

Limitations include reduced rows, parent-scaler fitting on extra training rows, a single release/split, tiny architecture calibration sets, broad intervals, architecture distribution shift, incomplete third-seed reliability, fixed beta/lambda, a coarse cost model, fixed promotion quotas, limited search-seed count, offline archive cost and historically examined test data. Existing MLP, CNN, GRU and hybrid studies remain contextual baselines with their original protocols; they cannot establish a matched final-model advantage for this pilot.

## Reproduction

Run from the repository root with the existing environment. GPU commands require CUDA access, which is outside the sandbox on this machine.

```bash
.venv/bin/python -m pytest -q
MPLCONFIGDIR=/tmp/amf-pred-mpl .venv/bin/python -m scripts.amf_reliability --output results/ua_amf_prediction_20261108 --device cuda
.venv/bin/python -m scripts.amf_fit_predictor --input results/ua_amf_prediction_20261108
MPLCONFIGDIR=/tmp/amf-pred-mpl .venv/bin/python -m scripts.amf_benchmark --study results/ua_amf_prediction_20261108 --output results/ua_amf_prediction_benchmark_20261108 --seeds 501 502 503
MPLCONFIGDIR=/tmp/amf-pred-mpl .venv/bin/python -m scripts.amf_analyze
```

The cohort cap persists across resumes, so repeating the original command does not grant fresh time or fabricate missing trajectories. Completed benchmark summary files are reused. For fresh timings choose fresh output directories. Finishing the reliability remainder requires an explicitly revised protocol/output and a transparently charged continuation budget; it is not counted as completed here. The predictive CLI configuration is `configs/ua_amf_prediction.yaml`. Search requires a bundle with an identical dataset/training/space context and checks its frozen hash when supplied.

## Verified completed comparison and final models

All 24 searches completed (eight variants, three seeds). The full method attained validation macro-F1 0.63354 ± 0.07882 versus standard BPSO 0.49155 ± 0.16861 and ordinary BPSO-SH/prediction-only 0.68317 ± 0.00802. Full GPU execution averaged 28.76442 seconds versus 28.76273 for BPSO: saving **-0.00589%**, effectively no demonstrated saving. Wall saving was -0.40513%, under equal 30-second caps with epoch-boundary overruns. The full method's F1 difference versus ordinary halving was -0.04963. Uncertainty hurt the third seed; the cost term produced no terminal-score benefit. No coefficient was revised after this negative result.

At target F1 0.2, every seed attained the target, and full mean hit time was 14.7133 seconds versus 14.0676 for BPSO. At target 0.6, full attained 2/3 seeds and BPSO 1/3; any conditional-time comparison is censored and cannot be generalized to all seeds. Prediction-only and ordinary halving attained 0.6 in 3/3 seeds. Full online results therefore do not show a uniform time-to-target improvement.

After validation selection, all six standard/full architecture selections were frozen. Dedicated seeds 1501/1502/1503 completed 18 model references (15 unique fits, because one architecture was shared) at a common 30 epochs in a separate stage taking 68.1946 seconds, below its 180-second cap. The final-stage script froze every choice before reading that stage's historical test array. Parent train/validation arrays were verified exactly equal between the MinMax source and the historical-test preprocessing source. No test outcome selected architectures, beta/lambda, loss, preprocessing, endpoint or threshold.

Historical-test macro-F1 was 0.62595 for full versus 0.54162 for standard BPSO, averaging dedicated-seed means by search seed. Normal FPR was 84.036% versus 90.366%; these are severe false-alarm rates, not operationally competitive IDS performance. Model-level accuracy, precision, recall, attack recalls, confusion matrices, parameters and measured inference latency are retained. The historical test was already development-known and is not independent/blind. Existing full-data MLP/CNN/GRU/reference results are preserved and are contextual rather than matched detector controls here.

Verification: all 183 reliability endpoint prediction archives, 485 successful benchmark prediction archives and 15 unique final test archives reproduce recorded metrics. Benchmark epoch ledgers and every training-job start were checked against caps. The final CPU suite passed 96 tests and the CUDA-targeted suite passed 41 tests. The new anytime mean/SD figure includes a point only when all three seeds have a terminal model at that limit; separate per-seed step curves preserve early censored behavior.

The main manuscript now contains the section **Computational Efficiency of Adaptive Multi-Fidelity BPSO**, generated from these checked artifacts. Its old PDF/body remain archived. The result does not support a novel IEEE optimizer-superiority claim: the full method fails to improve on ordinary halving and has no demonstrated net saving. The reliability problem and measurable early-late failures are supported, but the tested uncertainty/cost mechanism does not solve them better than the simpler controls.

To rebuild the analyses, final results and manuscript section from saved artifacts:

```bash
MPLCONFIGDIR=/tmp/amf-pred-mpl .venv/bin/python -m scripts.amf_analyze
.venv/bin/python -m scripts.amf_write_report
# Executed final stage; existing model directories reuse verified fits:
MPLCONFIGDIR=/tmp/amf-pred-mpl .venv/bin/python -m scripts.amf_final_evaluation
```

The final predictor-fit command reuses an existing frozen bundle; a fresh study directory is required to refit. After rebuilding the section, run `pdflatex`, `bibtex`, and `pdflatex` twice on `manuscript` from `docs/latex`. New original timing runs require fresh benchmark output directories, not deleting or rewriting historical ledgers.

Promotion mechanism diagnosis: uncertainty changed four full-method promotion decisions relative to removing its bonus on the same logged pools; the cost term changed zero. The frozen cost coefficient therefore had no allocation effect in these pools. This is a logged counterfactual ranking analysis, not a causal retraining experiment or permission to retune coefficients after seeing results. A future cost-weight study must be independently prespecified. All 24 search runs had zero numerical/memory failures and zero retries; controlled deadline stops are recorded separately. The largest final epoch/persistence wall overrun was 0.369 seconds, and every new training job started within its cap.
