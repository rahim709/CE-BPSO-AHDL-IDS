# Submission verification

| Record | Tests | Failures | Scope |
|---|---:|---:|---|
| [Package CPU](../../results/verification/package_cpu.xml) | 110 | 0 | Fresh test run from the exported code |
| [Final study CPU](../../results/ua_amf_online_20261108/tests_cpu.xml) | 110 | 0 | Archived final study test run |
| [Final study targeted CUDA](../../results/ua_amf_online_20261108/tests_cuda.xml) | 55 | 0 | Archived targeted CUDA-configured test run |
| [Additional online CUDA](../../results/ua_amf_online_20261108/tests_online_cuda_final.xml) | 5 | 0 | Archived additional CUDA-configured test run |
| [Detector CPU](../../docs/latex/evidence/nmri_tests_cpu.xml) | 80 | 0 | Archived detector study test run |
| [Detector CUDA](../../docs/latex/evidence/nmri_tests_cuda.xml) | 80 | 0 | Archived detector study CUDA-configured test run |

The package CPU suite was rerun from this exported source. GPU records are preserved from the completed experiments; no new GPU training or test run was performed during export. The suites overlap and their counts should not be summed.

Saved predictions, metric ledgers and feature cutoffs were checked from the exported files: 43 archive-free runs / 948 predictions, 108 efficiency pilot runs / 1,184 predictions, and 54 continuation runs / 1,069 predictions. All checks passed.

The standalone Overleaf project compiled with pdfLaTeX and BibTeX, with no unresolved references or overfull text boxes. Final comparison plots rebuilt from the included measurements.

The report and README editorial update did not rerun these suites. Its separate checks are recorded in [documentation checks](../../docs/reports/documentation_checks.json).
