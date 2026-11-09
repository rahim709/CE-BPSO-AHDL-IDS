# False-alarm optimization results

## Method and measured results

The balanced-loss reference falsely flagged 42.70% of normal test records. An exploratory follow-up compared unweighted and square-root inverse-frequency weighting using the same hybrid architecture, Min–Max preprocessing, Adam settings and a fixed 200-epoch endpoint. Candidate training used seed 42 and directories containing only training and validation arrays. Unweighted and square-root-weighted loss produced validation false-positive rates of 2.0685% and 2.0930%, with attack detection of 90.9104% and 91.3007%, respectively. Square-root weighting had the higher validation macro-F1, 0.8552 versus 0.8480.

Neither candidate satisfied the initial target of at most 1% false alarms and at least 90% attack detection. This failed target was retained in the evidence. An explicitly revised exploratory limit of 3% false alarms, with the same attack-detection constraint, was recorded before new test predictions. Square-root weighting was then selected by validation macro-F1. Three final runs used paired seeds 10042, 10043 and 10044 and exactly 200 epochs each on GPU. No model parameters, thresholds or endpoints were changed after reading those new test results.

| Test measure | Earlier balanced loss | Selected square-root weighting |
|---|---:|---:|
| Normal false-positive rate | 42.70% | 1.90% |
| Attack detection recall | 98.48% | 90.96% |
| Accuracy | 72.40% | 95.44% |
| Macro-F1 | 0.8454 | 0.8551 |

Values are means across three training seeds on the same historical test rows. Normal false alarms decreased by 40.80 percentage points. Attack detection also decreased, so the correction has a measurable sensitivity trade-off. A lower false-alarm rate does not imply every attack class improved.

| Class | Earlier mean recall | Corrected mean recall | Corrected mean precision |
|---|---:|---:|---:|
| Normal | 0.5730 | 0.9810 | 0.9487 |
| NMRI | 0.9439 | 0.0054 | 1.0000 |
| CMRI | 0.9845 | 0.9845 | 0.9383 |
| MSCI | 0.9615 | 0.9615 | 0.9677 |
| MPCI | 0.9804 | 0.9804 | 0.9810 |
| MFCI | 0.9652 | 0.9652 | 1.0000 |
| DoS | 0.9646 | 0.9600 | 0.9944 |
| Recon | 1.0000 | 1.0000 | 1.0000 |

This correction was motivated by known historical errors and validated on the existing development split. It does not constitute blind independent confirmation, a new search algorithm or achievement of the original 1% false-alarm target. All three final prediction archives independently reproduce their reported metrics.

## Reproduction

The selected preset is `configs/gas_low_false_alarm.yaml`. Prepare its dedicated data directory before training:

```bash
python -m src.cli prepare --config configs/gas_low_false_alarm.yaml
python -m src.cli train --config configs/gas_low_false_alarm.yaml --output results/low_false_alarm_repeat
```

The full study is recorded in `scripts.optimize_false_alarms` and `scripts.evaluate_low_false_alarm`. The original model remains available as a baseline. The CPU regression suite passed 77 tests, and all five new 200-epoch training runs used CUDA.
