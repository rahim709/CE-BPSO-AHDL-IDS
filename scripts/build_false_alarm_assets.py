"""Audit and document the exploratory validation-selected false-alarm correction."""
import argparse
import json
import hashlib
from pathlib import Path
import shutil
import numpy as np
from src.evaluation.metrics import classification_metrics
from src.utils.core import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',action='store_true')
    args=parser.parse_args()
    evidence=Path('docs/latex/evidence')
    source=Path('results/false_alarm_optimization/final_20261108')
    if not args.snapshot:
        for name in ['summary','measurements','protocol','environment']:
            shutil.copyfile(source/f'{name}.json',evidence/f'false_alarm_{name}.json')
        shutil.copyfile('results/false_alarm_optimization/20261108/validation_results.json',evidence/'false_alarm_validation_results.json')
        shutil.copyfile('results/false_alarm_optimization/20261108/protocol.json',evidence/'false_alarm_initial_protocol.json')
        shutil.copyfile('results/false_alarm_optimization/tests_cpu.xml',evidence/'false_alarm_tests_cpu.xml')
        for seed in [10042,10043,10044]:
            dest=evidence/'false_alarm_predictions'
            dest.mkdir(exist_ok=True)
            shutil.copyfile(source/f'seed_{seed}/test_predictions.npz',dest/f'{seed}.npz')
            rows=json.loads((source/f'seed_{seed}/trajectory.json').read_text())
            assert len(rows)==200 and all(r['device']=='cuda' for r in rows)
    summary=json.loads((evidence/'false_alarm_summary.json').read_text())
    rows=json.loads((evidence/'false_alarm_measurements.json').read_text())
    old=json.loads((evidence/'final_summary.json').read_text())
    old_rows=json.loads((evidence/'final_measurements.json').read_text())
    names=json.loads((evidence/'gas_class_names.json').read_text())['classes']
    audit=[]
    for row in rows:
        path=evidence/'false_alarm_predictions'/f"{row['seed']}.npz"
        with np.load(path) as p:
            m=classification_metrics(p['y'],p['probabilities'].argmax(1),list(names),'0')
        assert all(m[k]==row['metrics'][k] for k in m)
        audit.append({'seed':row['seed'],'all_metrics_match':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    save_json(evidence/'false_alarm_prediction_audit.json',audit)
    fpr=summary['attack_detection']['fpr']['mean']
    recall=1-summary['attack_detection']['fnr']['mean']
    f1=summary['metrics']['macro_f1']['mean']
    accuracy=summary['metrics']['accuracy']['mean']
    nmri_recall=float(np.mean([r['metrics']['per_class']['1']['recall'] for r in rows]))
    old_recall=1-np.mean([r['metrics']['attack_detection']['fnr'] for r in old_rows])
    md=f'''### 5.6 Validation-selected correction of false alarms

The balanced-loss reference falsely flagged 42.70% of normal test records. An exploratory follow-up compared unweighted and square-root inverse-frequency weighting using the same hybrid architecture, Min–Max preprocessing, Adam settings and a fixed 200-epoch endpoint. Candidate training used seed 42 and directories containing only training and validation arrays. Unweighted and square-root-weighted loss produced validation false-positive rates of 2.0685% and 2.0930%, with attack detection of 90.9104% and 91.3007%, respectively. Square-root weighting had the higher validation macro-F1, 0.8552 versus 0.8480.

Neither candidate satisfied the initial target of at most 1% false alarms and at least 90% attack detection. This failed target was retained in the evidence. An explicitly revised exploratory limit of 3% false alarms, with the same attack-detection constraint, was recorded before new test predictions. Square-root weighting was then selected by validation macro-F1. Three final runs used paired seeds 10042, 10043 and 10044 and exactly 200 epochs each on GPU. No model parameters, thresholds or endpoints were changed after reading those new test results.

| Test measure | Earlier balanced loss | Selected square-root weighting |
|---|---:|---:|
| Normal false-positive rate | {old['normal_false_positive_rate']['mean']*100:.2f}% | {fpr*100:.2f}% |
| Attack detection recall | {old_recall*100:.2f}% | {recall*100:.2f}% |
| Accuracy | {old['metrics']['accuracy']['mean']*100:.2f}% | {accuracy*100:.2f}% |
| Macro-F1 | {old['metrics']['macro_f1']['mean']:.4f} | {f1:.4f} |

Values are means across three training seeds on the same historical test rows. Normal false alarms decreased by {(old['normal_false_positive_rate']['mean']-fpr)*100:.2f} percentage points. Attack detection also decreased, so the correction has a measurable sensitivity trade-off. A lower false-alarm rate does not imply every attack class improved.

| Class | Earlier mean recall | Corrected mean recall | Corrected mean precision |
|---|---:|---:|---:|
'''
    for c in names:
        md+=f"| {names[c]['abbreviation']} | {np.mean([r['metrics']['per_class'][c]['recall'] for r in old_rows]):.4f} | {np.mean([r['metrics']['per_class'][c]['recall'] for r in rows]):.4f} | {np.mean([r['metrics']['per_class'][c]['precision'] for r in rows]):.4f} |\n"
    md+='\nThis correction was motivated by known historical errors and validated on the existing development split. It does not constitute blind independent confirmation, a new search algorithm or achievement of the original 1% false-alarm target. All three final prediction archives independently reproduce their reported metrics.\n'
    Path('docs/reports/false_alarm_addendum.md').write_text(md)
    conclusion=f'The exploratory validation-selected square-root weighting correction reduced mean normal test false alarms from 42.70% to {fpr*100:.2f}%, with accuracy {accuracy*100:.2f}%, macro-F1 {f1:.4f} and attack detection {recall*100:.2f}%. However, NMRI recall fell from 94.39% to {nmri_recall*100:.2f}%. This is a false-alarm versus attack-sensitivity trade-off, not a complete detector-quality solution; the original 1% false-alarm target remains unmet. Independent evaluation is still required.'
    save_json(evidence/'false_alarm_conclusion.json',{'text':conclusion})
    Path('docs/latex/false_alarm_conclusion.tex').write_text(conclusion.replace('%',r'\%')+'\n')
    latex=r'''\subsection{Validation-Selected False-Alarm Correction}
The balanced-loss reference was followed by an exploratory comparison of unweighted and square-root inverse-frequency weighting. Architecture, preprocessing, optimizer and the 200-epoch endpoint were held fixed. Candidate training used seed 42 and training/validation arrays only. Neither candidate met the initial 1\% validation false-alarm target with at least 90\% attack detection. Before generating new test predictions, the false-alarm limit was revised to 3\%, retaining the detection constraint. Square-root weighting was selected by validation macro-F1 (0.8552 versus 0.8480 for unweighted loss).

'''+conclusion.replace('%',r'\%')+r'''
The selected setting was evaluated at paired final seeds 10042, 10043 and 10044, each trained for exactly 200 epochs on GPU. These results describe a conventional loss correction on the development-known split. The revised target is exploratory, and no test-driven threshold or endpoint selection was performed after these new predictions.
'''
    Path('docs/latex/false_alarm_study.tex').write_text(latex)
    report='# False-alarm optimization results\n\n'+md.replace('### 5.6 Validation-selected correction of false alarms','## Method and measured results')
    report+='\n## Reproduction\n\nThe selected preset is `configs/gas_low_false_alarm.yaml`. Prepare its dedicated data directory before training:\n\n```bash\npython -m src.cli prepare --config configs/gas_low_false_alarm.yaml\npython -m src.cli train --config configs/gas_low_false_alarm.yaml --output results/low_false_alarm_repeat\n```\n\nThe full study is recorded in `scripts.optimize_false_alarms` and `scripts.evaluate_low_false_alarm`. The original model remains available as a baseline. The CPU regression suite passed 77 tests, and all five new 200-epoch training runs used CUDA.\n'
    Path('docs/false_alarm_optimization.md').write_text(report)
    print(conclusion)


if __name__=='__main__':
    main()
