"""Snapshot and report the measured NMRI/false-alarm specialist experiment."""
import argparse
import json
import shutil
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.evaluation.metrics import classification_metrics
from src.utils.core import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',action='store_true');args=parser.parse_args()
    e=Path('docs/latex/evidence');base=Path('results/nmri_tradeoff');final=base/'final_20261108'
    if not args.snapshot:
        for name in ['summary','measurements','protocol','environment','pipeline_seed_10042']:
            shutil.copyfile(final/f'{name}.json',e/f'nmri_{name}.json')
        for name in ['results','protocol','training_ambiguity','selection']:
            shutil.copyfile(base/'20261108'/f'{name}.json',e/f'nmri_validation_{name}.json')
        for name in ['tests_cpu.xml','tests_cuda.xml']:
            shutil.copyfile(base/name,e/f'nmri_{name}')
        dest=e/'nmri_predictions';dest.mkdir(exist_ok=True)
        for seed in [10042,10043,10044]:shutil.copyfile(final/f'predictions_{seed}.npz',dest/f'{seed}.npz')
    summary=json.loads((e/'nmri_summary.json').read_text());rows=json.loads((e/'nmri_measurements.json').read_text())
    trials=json.loads((e/'nmri_validation_results.json').read_text())
    audits=[]
    for row in rows:
        path=e/'nmri_predictions'/f"{row['seed']}.npz"
        with np.load(path) as p:m=classification_metrics(p['y'],p['prediction'],[str(i) for i in range(8)],'0')
        assert all(m[k]==row['metrics'][k] for k in m)
        audits.append({'seed':row['seed'],'all_metrics_match':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    save_json(e/'nmri_prediction_audit.json',audits)
    fpr=summary['fpr']['mean']*100;recall=summary['nmri_recall']['mean']*100
    accuracy=summary['metrics']['accuracy']['mean']*100;f1=summary['metrics']['macro_f1']['mean'];attack=summary['attack_recall']['mean']*100
    paragraph=f'A subsequent validation-selected normal/NMRI specialist increased mean NMRI test recall from 0.54% to {recall:.2f}%, while normal false alarms rose from 1.90% to {fpr:.2f}%. Accuracy was {accuracy:.2f}%, macro-F1 {f1:.4f}, and overall attack detection {attack:.2f}%. This improves the observed trade-off, but most NMRI attacks are still missed. The same training-fitted specialist was shared across three hybrid seeds; these are not three independent specialist fits or independent-dataset replicates.'
    save_json(e/'nmri_conclusion.json',{'text':paragraph})
    md='''### 5.7 Dedicated NMRI specialist and remaining ambiguity

Further validation experiments compared the existing hybrid score with Extra Trees, histogram gradient boosting and a neural specialist. The specialists were fitted only on normal and NMRI training records. The neural specialist used GPU training for 100 epochs; the tree methods used CPU. Only predictions assigned to normal or NMRI by the fixed hybrid were reconsidered; other attack decisions were preserved. Threshold calibration included false alarms already assigned to other classes, and threshold ties remained normal.

No tested method reached the initial target of at least 50% NMRI recall with at most 5% false alarms and at least 90% overall attack detection. An exploratory follow-up was therefore explicitly recorded: select the highest validation macro-F1 among settings with at most 3% false alarms, at least 10% NMRI recall and at least 90% attack detection. Histogram gradient boosting was selected from validation data; its seed-42 validation NMRI recall was 18.44% at 2.95% false alarms. Before new test predictions, thresholds were calibrated separately for each of the three existing hybrid models using validation data only.

'''+paragraph+'''

| Test operating point | Normal false alarms | NMRI recall | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Original balanced hybrid | 42.70% | 94.39% | 72.40% | 0.8454 |
| Square-root-weighted hybrid | 1.90% | 0.54% | 95.44% | 0.8551 |
'''+f'| Hybrid with NMRI specialist | {fpr:.2f}% | {recall:.2f}% | {accuracy:.2f}% | {f1:.4f} |\n'+'''
The training-only ambiguity audit found that 1,374 of 1,657 NMRI records shared their processed non-time predictor state with normal records; those states also contained 26,843 normal training records. Duration was excluded from that audit. This demonstrates substantial overlap, but it is not a proof that the full feature set cannot distinguish the classes. The small candidate comparison also cannot prove that no better model exists.

All three saved test prediction archives reproduce the reported metrics. A saved inference pipeline reproduces all 19,404 test decisions for the prespecified first hybrid seed. The revised CPU and CUDA-configured regression suites each pass 80 tests. The same previously examined split was reused, so all follow-up findings remain exploratory and require independent confirmation.

![Figure 5. Validation NMRI recall versus measured normal false-alarm rate for the tested routing policies.](../latex/figures/nmri_tradeoff.png)
'''
    Path('docs/reports/nmri_addendum.md').write_text(md)
    latex=r'\subsection{Dedicated NMRI Specialist}'+'\n'+paragraph.replace('%',r'\%')+r'''

The specialist was trained on normal/NMRI training rows and only reconsidered normal or NMRI hybrid predictions. Extra Trees, histogram gradient boosting and a GPU-trained neural specialist were compared on validation. No tested method achieved 50\% NMRI recall below 5\% false alarms. An explicitly revised exploratory criterion selected histogram gradient boosting by validation macro-F1 under 3\% false alarms, at least 10\% NMRI recall and 90\% overall attack detection. All three validation thresholds were fixed before new test predictions. Training-only analysis found 1,374 of 1,657 NMRI records shared their non-time predictor state with normal records. Duration was excluded, so this is evidence of overlap rather than a full-feature impossibility result.
'''
    Path('docs/latex/nmri_study.tex').write_text(latex)
    table=r'''\begin{tabular}{lrrrr}
\toprule
Setting & FPR (\%) & NMRI (\%) & Acc. (\%) & F1 \\
\midrule
Balanced & 42.70 & 94.39 & 72.40 & 0.8454 \\
Sqrt loss & 1.90 & 0.54 & 95.44 & 0.8551 \\
'''+f'Specialist & {fpr:.2f} & {recall:.2f} & {accuracy:.2f} & {f1:.4f} '+r'\\'+r'''
\bottomrule
\end{tabular}
'''
    Path('docs/latex/tables/nmri_operating_points.tex').write_text(table)
    with Path('docs/latex/nmri_study.tex').open('a') as stream:
        stream.write(r'''
\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}
\input{tables/nmri_operating_points}
\caption{Observed test operating points. NMRI denotes class recall, and FPR is the normal false-positive rate. Means use three hybrid seeds; the specialist is shared.}
\end{table}
''')
    Path('docs/latex/nmri_conclusion.tex').write_text(paragraph.replace('%',r'\%')+'\n')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(5.5,3.3))
    for name in ['existing_hybrid_score','extra_trees','hist_gradient_boosting','neural_specialist']:
        values=[r for r in trials if r['method']==name and 'metrics' in r]
        ax.plot([r['metrics']['attack_detection']['fpr']*100 for r in values],[r['metrics']['per_class']['1']['recall']*100 for r in values],marker='o',label=name.replace('_',' '))
    ax.set(xlabel='Normal false-positive rate (%)',ylabel='NMRI recall (%)',ylim=(0,55));ax.legend(fontsize=8);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(f'docs/latex/figures/nmri_tradeoff.{ext}',dpi=300)
    plt.close(fig)
    report='# NMRI recall and false-alarm balance\n\n'+md.replace('### 5.7 Dedicated NMRI specialist and remaining ambiguity','## Experiment and results').replace('../latex/figures/','latex/figures/')
    report+='''
## Run the frozen pipeline

The first hybrid seed is selected for a reproducible example, not because it had the best test score. The pipeline verifies hashes of the existing hybrid checkpoint, specialist and preprocessing artifacts. CSV inputs are transformed with the saved training-fitted preprocessor; NPZ inputs must contain already transformed features in the original order. No labels are used by the prediction command.

```bash
python -m scripts.predict_nmri_specialist \\
  --pipeline results/nmri_tradeoff/final_20261108/pipeline_seed_10042.json \\
  --input path/to/new_records.csv --output results/new_predictions.npz --device cuda
```

The hybrid runs on GPU and the tree specialist on CPU. The manifest references local trained artifacts; a plain source checkout requires recreating those artifacts first. The loss-only configuration does not automatically enable this second-stage classifier. Experiments are reproducible using `scripts.nmri_tradeoff_study` and `scripts.evaluate_nmri_specialist`, with fresh output directories and the earlier frozen hybrid runs. `scripts.build_nmri_assets --snapshot` re-scores bundled predictions and rebuilds the study figures without training data. Independent capture/release evaluation and substantially better NMRI sensitivity remain necessary before deployment claims.
'''
    Path('docs/nmri_balance.md').write_text(report)
    print(paragraph)


if __name__=='__main__':main()
