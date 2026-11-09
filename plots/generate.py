"""Rebuild final comparison figures from the included measurements, without training."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('plots/final'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        for extension in ['pdf', 'png']:
            fig.savefig(args.output / f'{name}.{extension}', dpi=300)
        plt.close(fig)

    evidence = Path('docs/latex/evidence')
    rows = []
    for label, filename in [('Balanced hybrid', 'final_summary.json'),
                            ('Weighted hybrid', 'false_alarm_summary.json'),
                            ('NMRI specialist', 'nmri_summary.json')]:
        summary = json.loads((evidence / filename).read_text())
        rows.append((label, summary))
    # Read the original aggregate schemas without changing the measurements.
    metrics = []
    for label, summary in rows:
        metric = summary['metrics']
        if label == 'Balanced hybrid':
            fpr = summary['normal_false_positive_rate']['mean']
            recall = summary['per_class_recall']['1']['mean']
        elif label == 'Weighted hybrid':
            fpr = summary['attack_detection']['fpr']['mean']
            measurements = json.loads((evidence / 'false_alarm_measurements.json').read_text())
            recall = np.mean([r['metrics']['confusion_matrix'][1][1] /
                              sum(r['metrics']['confusion_matrix'][1]) for r in measurements])
        else:
            fpr, recall = summary['fpr']['mean'], summary['nmri_recall']['mean']
        metrics.append([metric['accuracy']['mean'] * 100, metric['macro_f1']['mean'] * 100,
                        fpr * 100, recall * 100])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    values = np.asarray(metrics)
    for index, (axis, title) in enumerate(zip(axes.flat, ['Accuracy (%)', 'Macro-F1 (%)',
                                                        'Normal false alarms (%)', 'NMRI recall (%)'])):
        axis.bar([label for label, _ in rows], values[:, index])
        axis.set_ylabel(title)
        axis.tick_params(axis='x', rotation=15)
        axis.set_ylim(0, max(values[:, index]) * 1.18)
        for position, value in enumerate(values[:, index]):
            axis.text(position, value, f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    save(fig, 'detector_comparison')

    study = Path('results/ua_amf_online_20261108')
    comparison = pd.read_csv(study / 'comparison.csv')
    labels = {'bpso': 'BPSO', 'bpso_sh': 'BPSO-SH', 'dehb': 'DEHB', 'offline_ua': 'Offline UA',
              'online_prediction': 'Online prediction', 'online_uncertainty': 'Online uncertainty',
              'online_full': 'Archive-free full'}
    quality = comparison[comparison.phase == 'quality']
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(quality.variant.map(labels), quality.gpu_mean, yerr=quality.gpu_sd, capsize=3)
    axis.set_ylabel('Actual target-or-cap GPU elapsed (s)')
    axis.tick_params(axis='x', rotation=20)
    for index, row in enumerate(quality.itertuples()):
        axis.text(index, row.gpu_mean + row.gpu_sd + .3, f'{row.reached}/{row.seeds} targets',
                  ha='center', fontsize=8)
    save(fig, 'search_target_cost')

    compute = comparison[comparison.phase == 'compute']
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(compute.variant.map(labels), compute.f1_mean, yerr=compute.f1_sd, capsize=3)
    axis.set_ylabel('Terminal validation macro-F1 at 20 GPU seconds')
    axis.set_ylim(0, 1.1)
    axis.tick_params(axis='x', rotation=20)
    save(fig, 'search_fixed_compute')

    anytime = pd.read_csv(study / 'anytime.csv')
    fig, axis = plt.subplots(figsize=(10, 5))
    for method, group in anytime.groupby('variant'):
        stats = group.groupby('gpu_budget').macro_f1.agg(['mean', 'std', 'count'])
        stats = stats[stats['count'] == 3]
        axis.errorbar(stats.index, stats['mean'], yerr=stats['std'].fillna(0),
                      marker='o', capsize=2, label=labels[method])
    axis.set(xlabel='Actual GPU elapsed budget (s)', ylabel='Terminal validation macro-F1', ylim=(0, 1.1))
    axis.legend(fontsize=8)
    save(fig, 'search_anytime')
    print(f'Final graphs written to {args.output}')


if __name__ == '__main__':
    main()
