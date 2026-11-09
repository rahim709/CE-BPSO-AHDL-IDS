# Latest GPU verification — 8 November 2026

The latest implementation fixes have now been tested on the NVIDIA GeForce RTX 3050 Laptop GPU. The run took 503.16 seconds (about 8.4 minutes).

| Check | Outcome |
|---|---|
| Complete CPU suite | 77 passed, zero failures or skips |
| Complete CUDA-configured suite | 77 passed, zero failures or skips |
| Test-suite and workflow commands | 38 passed, zero failures |
| Recorded integration training trajectories | 80; all used CUDA |
| Saved final prediction archives | Eight independently re-scored; all checked metrics match |
| Search-selected architecture | Exported architecture matches trained architecture for all four profiles |

The component continuation tests now honor the selected device and assert the recorded device. CUDA regression coverage includes the configured promotion factor, incomplete-selection labelling and skipping final testing for incomplete comparison selections. CPU-oriented data preparation, scheduling, metrics and plotting retain their normal execution paths.

For real Gas and synthetic ToN-IoT, Bot-IoT and SWaT, the run executed preparation, fixed training/testing, search, diagnostics, comparison of all seven search controls, reports, search resume, search-selected training/testing and fixed-training resume. Re-scoring also checked synthetic flags, exact epoch counts and recorded CUDA devices for all eight final models.

These are bounded integration budgets. The three earlier 200-epoch reference results remain unchanged and were not retrained in this verification. Original ToN-IoT, Bot-IoT and SWaT benchmarks and full paper-scale search remain outside these software checks.

Full local logs: `results/gpu_validation/20261108T105751Z/`. Portable evidence: `docs/latex/evidence/latest_gpu_*`, including test XML, command records, hardware, trajectory measurements, prediction audit and hashes of the implementation tested. The complete command-by-command record is [GPU report](latex/evidence/latest_gpu_REPORT.md).
