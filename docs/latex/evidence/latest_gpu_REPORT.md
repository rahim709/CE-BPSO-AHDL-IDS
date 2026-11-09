# GPU validation measurements

Bounded integration checks; real Gas Pipeline and three synthetic profiles; not paper reproduction

GPU: NVIDIA GeForce RTX 3050 Laptop GPU

| Test suite | Passed | Failed/errors | Skipped |
|---|---:|---:|---:|
| cpu | 77 | 0 | 0 |
| cuda | 77 | 0 | 0 |

| CLI/test case | Status | Wall seconds |
|---|---|---:|
| pytest_cpu | PASS | 9.792 |
| pytest_cuda | PASS | 10.292 |
| gas_pipeline_prepare | PASS | 4.627 |
| gas_pipeline_train | PASS | 4.378 |
| gas_pipeline_search | PASS | 36.910 |
| gas_pipeline_diagnostics | PASS | 68.392 |
| gas_pipeline_compare | PASS | 208.659 |
| gas_pipeline_report | PASS | 2.673 |
| gas_pipeline_search_resume | PASS | 2.275 |
| gas_pipeline_selected_train | PASS | 21.527 |
| gas_pipeline_train_resume | PASS | 3.475 |
| ton_iot_prepare | PASS | 2.422 |
| ton_iot_train | PASS | 3.575 |
| ton_iot_search | PASS | 4.727 |
| ton_iot_diagnostics | PASS | 5.732 |
| ton_iot_compare | PASS | 15.813 |
| ton_iot_report | PASS | 2.773 |
| ton_iot_search_resume | PASS | 2.220 |
| ton_iot_selected_train | PASS | 4.227 |
| ton_iot_train_resume | PASS | 3.325 |
| bot_iot_prepare | PASS | 2.272 |
| bot_iot_train | PASS | 3.576 |
| bot_iot_search | PASS | 4.829 |
| bot_iot_diagnostics | PASS | 5.581 |
| bot_iot_compare | PASS | 14.303 |
| bot_iot_report | PASS | 2.524 |
| bot_iot_search_resume | PASS | 2.271 |
| bot_iot_selected_train | PASS | 4.232 |
| bot_iot_train_resume | PASS | 3.375 |
| swat_prepare | PASS | 2.321 |
| swat_train | PASS | 3.576 |
| swat_search | PASS | 4.630 |
| swat_diagnostics | PASS | 5.580 |
| swat_compare | PASS | 13.899 |
| swat_report | PASS | 2.576 |
| swat_search_resume | PASS | 2.272 |
| swat_selected_train | PASS | 4.126 |
| swat_train_resume | PASS | 3.328 |

| Dataset | Synthetic | Test accuracy | Macro F1 | Train + validation seconds | Inference ms/sample |
|---|---|---:|---:|---:|---:|
| gas_pipeline | False | 0.630334 | 0.096657 | 1.315 | 0.000748 |
| ton_iot | True | 0.063333 | 0.011912 | 0.326 | 0.008744 |
| bot_iot | True | 0.376667 | 0.109443 | 0.318 | 0.008430 |
| swat | True | 0.240000 | 0.193548 | 0.344 | 0.008982 |

Per-test timings: `test_cases.csv`. Per-class metrics and confusion matrices: `<dataset>/train/`.
Search ledgers, checkpoints and synchronized CUDA timings are retained under each dataset.
Small, two-epoch fixed models are software checks; these scores do not establish detector quality.
