# Current GPU verification — 8 November 2026

**Latest GPU rerun:** [77 tests passed on CPU and CUDA; 38 commands passed; all 80 integration training trajectories used CUDA](latest_gpu_verification.md). This record includes the latest implementation fixes and GPU component-continuation checks.

The latest [execution review](current_execution_review.md) records 74 passing tests per CPU/CUDA-configured suite, 30/30 bounded CLI commands and 76 CUDA training trajectories. A separate fixed-reference study completed three 200-epoch CUDA training/test runs, and a real-Gas search-selected architecture was also trained and tested. Current evidence is in `docs/latex/evidence/current_gpu_*`, `current_tests_*` and `final_*`. The measurements below are the earlier GPU record.

# Measured GPU validation

Executed on 2026-11-06 with an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB),
Python 3.14.4, and PyTorch 2.15.0+cu132. GPU access required execution outside the
local sandbox. No training implementation changes were needed; tests now accept
`BPSO_TEST_DEVICE=cuda` while retaining CPU as their default.

| Check | Result | Wall time including process startup |
|---|---|---:|
| Complete CPU test suite | 67 passed, no failures or skips | 10.295 s |
| Complete suite with CUDA model/training tests | 67 passed, no failures or skips | 9.244 s |
| Real Gas Pipeline seven-method comparison | Passed | 192.595 s |
| All tests and integration commands | 30 commands passed, no failures | 430.901 s |

The 28 integration commands exercised preparation, fixed training, BPSO search,
diagnostics, a seven-method comparison, reporting, and search resume for each of
four profiles. Gas Pipeline used the real local ARFF; ToN-IoT, Bot-IoT, and SWaT
used explicitly generated synthetic data. All 76 integration training runs
(140 recorded epochs) reported CUDA devices. Their synchronized training and
validation time totaled 314.780 seconds.

The comparisons covered BPSO, reduced BPSO, halving BPSO, Hyperband, DEHB, random,
and surrogate search. They used the existing small integration configs, one seed,
and a four-epoch cap per method. Actual resource use varies by method. Hyperband
on Gas stopped at the cap boundary with `budget_exhausted`, an expected budget
outcome rather than an execution failure. These results do not establish speedup
or comparative research performance.

| Fixed model profile | Synthetic | Held-out accuracy | Macro F1 |
|---|---|---:|---:|
| Gas Pipeline | No | 63.0334% | 0.096657 |
| ToN-IoT | Yes | 6.3333% | 0.011912 |
| Bot-IoT | Yes | 37.6667% | 0.109443 |
| SWaT | Yes | 24.0000% | 0.193548 |

Fixed models ran for only two epochs. The Gas model predicted only the majority
class and missed all seven attack classes. These measurements verify software
execution; they do not demonstrate a useful detector or full paper reproduction.

Reproduce with `python -m scripts.run_gpu_validation` after setting up the local
datasets. Original detailed artifacts are local under
`results/gpu_validation/20261106T124004Z/`, including `REPORT.md`, `test_cases.csv`,
JUnit XML, per-class metrics, confusion matrices, search ledgers, checkpoints, and
GPU timings. Those generated artifacts are excluded from Git. New runs create
their own timestamped directories.
