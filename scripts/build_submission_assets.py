"""Generate submission tables/figures from audited saved measurements only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.evaluation.metrics import classification_metrics
from src.utils.core import save_json

ROOT = Path('docs/latex')
EVIDENCE = ROOT / 'evidence'


def table(name: str, columns: str, header: str, rows: list[str]) -> None:
    (ROOT / 'tables' / f'{name}.tex').write_text(
        '\\begin{tabular}{' + columns + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n'
        + '\n'.join(row + ' \\\\' for row in rows) + '\n\\bottomrule\n\\end{tabular}\n')


def figure(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(ROOT / 'figures' / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', action='store_true', help='Use bundled evidence; no historical run directories required')
    args = parser.parse_args()
    for folder in ['tables', 'figures', 'evidence']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    source = Path('results/learning_sanity/20261105T114230Z')
    if not args.snapshot:
        for name in ['results.json', 'results.csv', 'protocol.json', 'environment.json']:
            shutil.copyfile(source / name, EVIDENCE / ('learning_' + name))
    metrics = {}
    for name in ['control', 'balanced_reference']:
        folder = Path('results/learning_sanity/final_verification') / name
        if not args.snapshot:
            for file in ['test_metrics.json', 'test_predictions.npz', 'per_class.csv', 'confusion_matrix.csv', 'environment.json', 'resolved_config.json', 'trajectory.json']:
                shutil.copyfile(folder / file, EVIDENCE / f'{name}_{file}')
        metrics[name] = json.loads((EVIDENCE / f'{name}_test_metrics.json').read_text())
    if args.snapshot:
        integration = Path(json.loads((EVIDENCE / 'verification_summary.json').read_text())['integration_directory'])
    else:
        shutil.copyfile('results/submission_review/latex_review_tests.xml', EVIDENCE / 'tests_cpu.xml')
        integration = Path(json.loads(Path('results/software_validation/latest.json').read_text())['directory'])
        for file in ['verification_summary.json', 'environment.json', 'resolved_config.json']:
            shutil.copyfile(integration / file, EVIDENCE / ('integration_' + file))
        shutil.copyfile('docs/gas_pipeline_source.json', EVIDENCE / 'gas_pipeline_source.json')
    metadata = json.loads((EVIDENCE / 'data_metadata.json').read_text())
    classes = metadata['classes']
    naming = json.loads((EVIDENCE / 'gas_class_names.json').read_text())
    assert naming['raw_sha256'] == metadata['sources'][0]['sha256']
    names = naming['classes']
    assert set(names) == set(classes)
    for c in classes:
        assert names[c]['count'] == sum(metadata['splits'][s]['classes'][c] for s in ['train', 'val', 'test'])
    short_names = [names[c]['abbreviation'] for c in classes]
    table('splits', 'llrrr', 'Code & Class & Train & Validation & Test', [
        f"{c} & {names[c]['abbreviation']} & {metadata['splits']['train']['classes'][c]:,} & {metadata['splits']['val']['classes'][c]:,} & {metadata['splits']['test']['classes'][c]:,}" for c in classes]
        + ['Total & & 58,211 & 19,404 & 19,404'])
    table('class_mapping', 'lllr', 'Code & Class & Description & Count', [
        f"{c} & {names[c]['abbreviation']} & {names[c]['name']} & {names[c]['count']:,}" for c in classes])
    table('class_names', 'll', 'Class (code) & Meaning', [
        f"{names[c]['abbreviation']} ({c}) & {names[c]['name']}" for c in classes])
    learning = json.loads((EVIDENCE / 'learning_results.json').read_text())
    rows = [r for r in learning if not r['synthetic_data']]
    table('gas_ablation', 'llrr', 'Preset & Loss & Macro-F1 & Classes', [
        f"{r['model'].capitalize()} & {r['class_weighting']} & {r['validation_macro_f1']:.6f} & {r['predicted_class_count']}/8" for r in rows])
    controls = json.loads((EVIDENCE / 'validation_controls.json').read_text())
    for name, saved in metrics.items():
        with np.load(EVIDENCE / f'{name}_test_predictions.npz') as p:
            measured = classification_metrics(p['y'], p['probabilities'].argmax(1), classes)
        for key in ['macro_f1', 'accuracy', 'confusion_matrix']:
            assert np.allclose(saved[key], measured[key], atol=1e-12, rtol=0), (name, key)
    prediction_dir = EVIDENCE / 'validation_predictions'
    prediction_dir.mkdir(exist_ok=True)
    for row in controls['models']:
        path = prediction_dir / Path(row['predictions']).name
        if not args.snapshot:
            shutil.copyfile(row['predictions'], path)
        with np.load(path) as p:
            measured = classification_metrics(p['y'], p['prediction'], classes, '0')
        for key in ['macro_f1', 'accuracy', 'confusion_matrix']:
            assert np.allclose(row['metrics'][key], measured[key], atol=1e-12, rtol=0), (path, key)
    table('tabular_controls', 'llrrr', 'Method & Features & Accuracy & Macro-F1 & FPR', [
        f"Majority & --- & {controls['majority']['accuracy']:.6f} & {controls['majority']['macro_f1']:.6f} & {float(controls['majority']['training_majority_class'] != metadata['config']['normal_label']):.6f}"] + [
        f"{'RF' if r['method'] == 'random_forest' else 'LR'} & {r['features'].replace('_', ' ')} & {r['metrics']['accuracy']:.6f} & {r['metrics']['macro_f1']:.6f} & {r['metrics']['attack_detection']['fpr']:.6f}" for r in controls['models']])
    labels = [('accuracy', 'Accuracy'), ('macro_precision', 'Macro precision'), ('macro_recall', 'Macro recall'),
              ('macro_f1', 'Macro-F1'), ('weighted_f1', 'Weighted-F1'), ('parameters', 'Parameters'),
              ('predicted_class_count', 'Predicted classes'), ('zero_recall_class_count', 'Zero-recall classes'),
              ('training_and_validation_seconds', 'Training + validation (s)')]
    table('heldout', 'lrr', 'Measure & Tiny / none & Reference / balanced', [
        label + ' & ' + ' & '.join(f'{metrics[n][key]:,}' if key in ['parameters', 'predicted_class_count', 'zero_recall_class_count']
                                  else f'{metrics[n][key]:.6f}' for n in metrics)
        for key, label in labels])
    table('class_recall', 'llrr', 'Code & Class & Control recall & Reference recall', [
        c + ' & ' + names[c]['abbreviation'] + ' & ' + ' & '.join(f"{metrics[n]['per_class'][c]['recall']:.6f}" for n in metrics) for c in classes])
    overlap = json.loads((EVIDENCE / 'leakage_audit.json').read_text())['overlaps']
    table('overlap', 'lrr', 'Unique predictor hashes & All 26 & Without time', [
        f"{label} & {overlap['all_features'][key]:,} & {overlap['without_time'][key]:,}" for key, label in [
        ('predictor_overlap_train_val', 'Train--validation'), ('predictor_overlap_train_test', 'Train--test'),
        ('predictor_overlap_val_test', 'Validation--test'), ('predictor_hashes_with_conflicting_labels', 'Conflicting labels (corpus)')]])
    tests = list(ET.parse(EVIDENCE / 'tests_cpu.xml').iter('testcase'))
    save_json(EVIDENCE / 'verification_summary.json', {'cpu_tests': len(tests),
        'cpu_failed': sum(c.find('failure') is not None or c.find('error') is not None for c in tests),
        'integration_directory': str(integration), 'new_test_evaluations': 0,
        'new_validation_controls': len(controls['models']),
        'historical_test_note': 'Existing predictions re-scored only; no retraining or held-out tuning'})
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    ax.bar(np.arange(4), [r['validation_macro_f1'] for r in rows], color=['#708090','#708090','#1b7a78','#1b7a78'])
    ax.set(xticks=np.arange(4), xticklabels=['Tiny\nnone', 'Tiny\nbalanced', 'Reference\nnone', 'Reference\nbalanced'],
           ylabel='Validation macro-F1', ylim=(0, .3))
    figure(fig, 'gas_ablation')
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    positions = np.arange(8)
    for offset, (name, display) in enumerate([('control','Tiny / none'), ('balanced_reference','Reference / balanced')]):
        ax.bar(positions + (offset-.5)*.36, [metrics[name]['per_class'][c]['recall'] for c in classes], .36, label=display)
    ax.set(xticks=positions, xticklabels=short_names, xlabel='Gas Pipeline class', ylabel='Held-out class recall', ylim=(0,1.12))
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    ax.legend(fontsize=8, loc='upper center')
    figure(fig, 'class_recall')
    fig, axes = plt.subplots(1,2,figsize=(6.8,3))
    for ax, name, title in zip(axes, metrics, ['Tiny / none','Reference / balanced']):
        cm = np.array(metrics[name]['confusion_matrix'],dtype=float)
        cm /= cm.sum(axis=1,keepdims=True)
        im = ax.imshow(cm, vmin=0,vmax=1,cmap='Blues')
        ax.set(xticks=positions, yticks=positions, xticklabels=short_names, yticklabels=short_names,
               xlabel='Predicted class', ylabel='True class', title=title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, pad=.03, label='Row fraction')
    # Fixed margins avoid tight_layout's shared-colorbar incompatibility.
    fig.savefig(ROOT / 'figures' / 'confusion.pdf',bbox_inches='tight')
    fig.savefig(ROOT / 'figures' / 'confusion.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.8, 3))
    for name, label in [('control','Tiny / none'),('balanced_reference','Reference / balanced')]:
        trajectory = json.loads((EVIDENCE / f'{name}_trajectory.json').read_text())
        ax.plot([r['epoch'] for r in trajectory], [r['macro_f1'] for r in trajectory], marker='o', label=label)
    ax.set(xlabel='Epoch', ylabel='Validation macro-F1', xticks=[1,3,5,7,10])
    ax.legend(fontsize=8)
    figure(fig,'learning_curves')
    fig, ax = plt.subplots(figsize=(3.5,2.2))
    views = ['all','without_time','time_only']
    for i, (method, label) in enumerate([('random_forest','Random forest'),('logistic_regression','Logistic regression')]):
        vals = [next(r['metrics']['macro_f1'] for r in controls['models'] if r['features']==v and r['method']==method) for v in views]
        ax.bar(np.arange(3)+(i-.5)*.35, vals,.35,label=label)
    ax.axhline(controls['majority']['macro_f1'],color='gray',linestyle='--',label='Majority')
    ax.set(xticks=np.arange(3),xticklabels=['All features','Without time','Time only'],ylabel='Validation macro-F1',ylim=(0,1))
    ax.legend(fontsize=8)
    figure(fig,'tabular_controls')
    print('Submission assets regenerated from saved measurements.')


if __name__ == '__main__':
    main()
