"""Verify and snapshot final-reference measurements; render submission assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.build_submission_assets import EVIDENCE, ROOT, figure, table
from src.evaluation.metrics import classification_metrics
from src.utils.core import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/submission_final_evaluation/20261108'))
    parser.add_argument('--snapshot', action='store_true')
    args = parser.parse_args()
    if not args.snapshot:
        summary = json.loads((args.input / 'summary.json').read_text())
        assert summary['complete'] and summary['all_prediction_metrics_verified']
        protocol = args.input / 'protocol.json'
        assert hashlib.sha256(protocol.read_bytes()).hexdigest() == summary['protocol_sha256']
        for name in ['summary', 'measurements', 'protocol', 'commands', 'environment']:
            shutil.copyfile(args.input / f'{name}.json', EVIDENCE / f'final_{name}.json')
        shutil.copyfile(args.input / 'processed/metadata.json', EVIDENCE / 'final_data_metadata.json')
        rows = json.loads((args.input / 'measurements.json').read_text())
        curves = {}
        for row in rows:
            run = Path(row['run'])
            dest = EVIDENCE / 'final_predictions' / f'seed_{row["seed"]}.npz'
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(run / 'test_predictions.npz', dest)
            curves[str(row['seed'])] = [
                {'epoch': r['epoch'], 'macro_f1': r['macro_f1'], 'loss': r['loss']}
                for r in json.loads((run / 'trajectory.json').read_text())]
        save_json(EVIDENCE / 'final_learning_curves.json', curves)
    summary = json.loads((EVIDENCE / 'final_summary.json').read_text())
    rows = json.loads((EVIDENCE / 'final_measurements.json').read_text())
    meta = json.loads((EVIDENCE / 'final_data_metadata.json').read_text())
    curves = json.loads((EVIDENCE / 'final_learning_curves.json').read_text())
    assert len(rows) == summary['n'] == 3
    assert hashlib.sha256((EVIDENCE / 'final_protocol.json').read_bytes()).hexdigest() == summary['protocol_sha256']
    for row in rows:
        path = EVIDENCE / 'final_predictions' / f'seed_{row["seed"]}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['prediction_sha256']
        with np.load(path) as p:
            verified = classification_metrics(p['y'], p['probabilities'].argmax(1), meta['classes'], '0')
        assert all(verified[k] == row['metrics'][k] for k in verified)
    for key in summary['metrics']:
        values = [r['metrics'][key] for r in rows]
        assert np.isclose(summary['metrics'][key]['mean'], np.mean(values), atol=1e-12, rtol=0)
        assert np.isclose(summary['metrics'][key]['sample_sd'], np.std(values, ddof=1), atol=1e-12, rtol=0)
    table('final_evaluation', 'rrrrr', 'Seed & Val. F1 & Test acc. & Test F1 & FPR', [
        f"{r['seed']} & {r['validation_macro_f1']:.4f} & {r['metrics']['accuracy']:.4f} & {r['metrics']['macro_f1']:.4f} & {r['metrics']['attack_detection']['fpr']:.4f}"
        for r in rows])
    names = json.loads((EVIDENCE / 'gas_class_names.json').read_text())['classes']
    table('final_class_recall', 'llrr', 'Code & Class & Mean recall & SD', [
        f"{c} & {names[c]['abbreviation']} & {summary['per_class_recall'][c]['mean']:.4f} & {summary['per_class_recall'][c]['sample_sd']:.4f}"
        for c in meta['classes']])
    class_summary = {c: {key: {'mean': float(np.mean([r['metrics']['per_class'][c][key] for r in rows])),
                                'sample_sd': float(np.std([r['metrics']['per_class'][c][key] for r in rows], ddof=1))}
                         for key in ['precision', 'recall', 'f1']}
                     for c in meta['classes']}
    save_json(EVIDENCE / 'final_class_summary.json', class_summary)
    table('final_class_metrics', 'lrrrr', 'Class & Precision & Recall & F1 & Support', [
        f"{names[c]['abbreviation']} & {class_summary[c]['precision']['mean']:.4f} & {class_summary[c]['recall']['mean']:.4f} & {class_summary[c]['f1']['mean']:.4f} & {rows[0]['metrics']['per_class'][c]['support']:,}"
        for c in meta['classes']])
    macros = {'FinalTestFOne': summary['metrics']['macro_f1']['mean'],
              'FinalTestFOneSD': summary['metrics']['macro_f1']['sample_sd'],
              'FinalTestAccuracy': summary['metrics']['accuracy']['mean'],
              'FinalTestFPR': summary['normal_false_positive_rate']['mean']}
    (ROOT / 'final_macros.tex').write_text('% Generated from verified fixed-endpoint test predictions.\n' +
        '\n'.join('\\newcommand{\\' + k + '}{' + f'{v:.4f}' + '}' for k, v in macros.items()) + '\n')
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for seed, curve in curves.items():
        ax.plot([r['epoch'] for r in curve], [r['macro_f1'] for r in curve], label=seed)
    ax.set(xlabel='Training epoch', ylabel='Validation macro-F1', ylim=(0, 1))
    ax.legend(title='Final training seed', fontsize=8)
    figure(fig, 'final_learning_curves')
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    values = [summary['per_class_recall'][c] for c in meta['classes']]
    ax.bar(range(8), [v['mean'] for v in values], yerr=[v['sample_sd'] for v in values], capsize=2)
    ax.set(xticks=range(8), xticklabels=[names[c]['abbreviation'] for c in meta['classes']],
           ylabel='Mean test recall', ylim=(0, 1.12))
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    figure(fig, 'final_class_recall')
    matrices = np.array([r['metrics']['confusion_matrix'] for r in rows], dtype=float)
    normalized = matrices / matrices.sum(axis=2, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    im = ax.imshow(normalized.mean(0), vmin=0, vmax=1, cmap='Blues')
    labels = [names[c]['abbreviation'] for c in meta['classes']]
    ax.set(xticks=range(8), yticks=range(8), xticklabels=labels, yticklabels=labels,
           xlabel='Predicted class', ylabel='True class')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.colorbar(im, ax=ax, label='Mean fraction within true class')
    figure(fig, 'final_confusion')
    print('Verified all three final prediction archives and rebuilt final evaluation assets.')


if __name__ == '__main__':
    main()
