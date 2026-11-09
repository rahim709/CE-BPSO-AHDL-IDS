"""Verify saved pilot predictions and generate research tables/addendum."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.metrics import classification_metrics
from src.utils.core import save_json
from scripts.ua_amf_study import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/ua_amf_multiseed_20261108')
    args = parser.parse_args()
    root = Path(args.input)
    summarize(root)
    frame = pd.read_csv(root / 'seed_results.csv')
    checks = []
    meta = json.loads((root / 'data/metadata.json').read_text())
    for path in root.rglob('validation_epoch_2.npz'):
        trajectory = path.parent / 'trajectory.json'
        rows = json.loads(trajectory.read_text())
        expected = next(r['macro_f1'] for r in rows if r['epoch'] == 2)
        with np.load(path) as data:
            actual = classification_metrics(data['y'], data['probabilities'].argmax(1), meta['classes'])['macro_f1']
        checks.append({'path': str(path), 'matches': bool(np.isclose(actual, expected, rtol=0, atol=1e-12))})
    if not all(c['matches'] for c in checks):
        raise ValueError('Saved prediction mismatch')
    for path in root.glob('runs/*/seed_*/search_summary.json'):
        s = json.loads(path.read_text())
        events = json.loads((path.parent / 'evaluations.json').read_text())
        assert s['charged_epochs'] == sum(e['charged_epochs'] for e in events) <= 24
        if s['method'] == 'ua_amf_bpso' and s['best']:
            assert s['best']['completed_budget'] == 2
            assert s['best']['fitness'] == np.mean(s['best']['observed_seed_fitness'])
    save_json(root / 'prediction_audit.json', {'checked': len(checks), 'all_match': True, 'records': checks})
    grouped = frame.groupby('variant').agg(n=('seed', 'count'), f1=('assessment_macro_f1', 'mean'),
                 f1_sd=('assessment_macro_f1', 'std'), seconds=('wall_seconds', 'mean'),
                 time_sd=('wall_seconds', 'std'), epochs=('charged_epochs', 'mean'), gpu=('gpu_seconds', 'mean'),
                 memory=('peak_gpu_memory_bytes', 'max'), unique=('unique_architectures', 'mean'), promoted=('promoted_candidates', 'mean'))
    baseline = grouped.loc['bpso', 'seconds']
    grouped['saving_percent'] = 100 * (baseline - grouped.seconds) / baseline
    grouped.to_csv(root / 'comparison_table.csv')
    table = ['| Method | Seeds | Assessment macro-F1 | Search s (mean ± SD) | Epochs | Unique | GPU s | Peak GPU MiB | Saving vs BPSO |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    tex = []
    for variant, r in grouped.iterrows():
        sd = f'{r.time_sd:.2f}' if pd.notna(r.time_sd) else 'NA'
        table.append(f'| {variant} | {int(r.n)} | {r.f1:.6f} | {r.seconds:.2f} ± {sd} | {r.epochs:.1f} | {r['unique']:.1f} | {r.gpu:.2f} | {r.memory / 2**20:.2f} | {r.saving_percent:.1f}% |')
        label = variant.replace('_', r'\_')
        tex.append(f'{label} & {r.f1:.4f} & {r.seconds:.2f} & {r.saving_percent:.1f}' + chr(92) * 2)
    (root / 'comparison_table.tex').write_text(r'\begin{tabular}{lrrr}\toprule' + '\n' + r'Method & Macro-F1 & Search s & Saving (\%)\\\midrule' + '\n' + '\n'.join(tex) + '\n' + r'\bottomrule\end{tabular}' + '\n')
    promotions = []
    for path in root.glob('runs/*/seed_*/promotions.json'):
        for record in json.loads(path.read_text()):
            ranks = np.argsort(-np.asarray(record['means']), kind='stable')[:len(record['selected_indices'])]
            promotions.append({'variant': path.parent.parent.name, 'seed': int(path.parent.name.split('_')[-1]),
                  'generation': record['generation'], 'rung': record['budget'], 'n': len(record['means']),
                  'selected': len(record['selected_indices']), 'max_sem': max(record['sem']),
                  'differs_from_mean_rank': set(ranks.tolist()) != set(record['selected_indices'])})
    pd.DataFrame(promotions).to_csv(root / 'promotion_analysis.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    if promotions:
        pdata = pd.DataFrame(promotions).groupby('variant').differs_from_mean_rank.mean()
        ax.bar(pdata.index, pdata.values)
    ax.set(ylabel='Fraction of promotion decisions differing from mean rank', ylim=(0, 1))
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(root / 'promotion_analysis.png', dpi=160)
    plt.close(fig)
    assessments = json.loads((root / 'validation_assessments.json').read_text())
    collapse = sum(r['collapsed_to_single_class'] for r in assessments)
    assessment_cost = float(frame.assessment_wall_seconds.sum())
    full = grouped.loc['ua_amf_bpso']
    controls = json.loads((root / 'validation_controls.json').read_text())
    control_lines = [f"- {family}: mean validation macro-F1 {np.mean([r['macro_f1'] for r in controls if r['family'] == family]):.6f}." for family in sorted({r['family'] for r in controls})]
    report = f'''# Measured UA-AMF-BPSO feasibility results

Scope: real Gas Pipeline, all eight classes, 1,536 parent-training rows and 768 parent-validation rows, original training-fitted column-L2 preprocessing, full 73-bit space, cumulative budgets 1/2 epochs. These are deliberately short learning-failure/engineering pilots, not a full-data detection benchmark. Device: {json.loads((root / 'environment.json').read_text())['gpu'] or 'CPU'} (see protocol for actual training device).

''' + '\n'.join(table) + f'''

Every run's epoch ledger was checked against its summary. {len(checks)} saved two-epoch validation prediction archives reproduce their recorded macro-F1. The separate common-assessment cost totals {assessment_cost:.3f} wall seconds, excluded from reported search savings. Independent assessment initializations are shared across methods and search seeds, not independent datasets.

{collapse}/{len(assessments)} selected-architecture assessment models collapse to one predicted class. Full-method mean assessment macro-F1 is {full.f1:.6f}; measured saving versus BPSO is {full.saving_percent:.2f}%. Negative saving means slower. These low and often tied scores do not establish competitive intrusion detection or preserved useful performance. Any faster time at this endpoint can reflect selected architecture cost, caching, ties, scheduler choices, and runtime variability.

The ordinary halving and replicated fixed/uncertainty/budget/full variants are included in the table and seed CSV. Promotion differences and SEM magnitudes are in `promotion_analysis.csv`; they describe allocation behavior, not evidence of beneficial uncertainty calibration. The cost model is an uncalibrated local epoch-cost extrapolation. No significance test is justified by these small, dependent, near-degenerate outcomes. The complete search candidate universe was not continued to a common reference, so search selection regret remains null. A separate three-architecture 3/10/30 diagnostic cohort completed 270 epochs: all nine terminal models tied at macro-F1 0.096645 and collapsed to one class. Its zero regret is degenerate and does not demonstrate useful selection. See `results/ua_amf_cohort_20261108/cohort_audit.json`.

Conventional fixed-protocol, two-epoch validation controls:
''' + '\n'.join(control_lines) + '''

Full precision, recall, macro-F1, accuracy, normal FPR, class recalls, confusion matrices and parameter counts for every assessment are in `validation_assessments.json`; the same validation metrics for MLP/CNN are in `validation_controls.json`. These are **validation** metrics, not final IDS test results. CPU runs report zero GPU use; CUDA logs report synchronized execution time and allocated peak memory, not energy or kernel-profiler time. Search methods are serial within each study. The CPU multiseed study overlapped briefly with CUDA correctness/pilot jobs; cross-study timing contamination is a limitation and these timings cannot support a clean causal speedup claim.

No expensive full-scale search or new final-test run was launched: this pilot did not establish useful detection performance or uncertainty benefit. Historical test partitions remain development-known. A new independent capture/release is unavailable in this study. Final architecture selection and independent final models for the proposed method remain planned.

The next justified experiment is to establish learning and rank separation on a prespecified common full-training candidate cohort at 3/10/30 epochs with replicated seeds. Measure all candidates at the terminal reference budget to obtain uncensored regret, early-late rank stability, SEM behavior and cost-prediction error. Then freeze a matched-GPU-time protocol and randomized method order, validate a practical F1 noninferiority margin, and run sufficiently many search seeds. Only proceed to final detector comparisons after that gate; include MLP/CNN and independent-capture evaluation. This implementation is research infrastructure, not yet evidence for an IEEE optimization-method superiority claim.
'''
    (root / 'REPORT.md').write_text(report)
    print('\n'.join(table))
    print('Prediction archives verified:', len(checks))


if __name__ == '__main__':
    main()
