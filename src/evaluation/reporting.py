"""Tables, independent-seed uncertainty and standalone vector/raster figures."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from src.utils.core import save_json


def _figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def seed_summary(rows: list[dict], metrics: list[str]) -> list[dict]:
    """Group only matched experiment protocols and unique random seeds."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    summaries = []
    for (method, data, protocol), group in frame.groupby(['method', 'data_fingerprint', 'protocol_fingerprint']):
        if group['seed'].duplicated().any():
            raise ValueError('Duplicate seed in the same method/data/protocol: report independent experiment roots separately')
        for metric in metrics:
            if metric not in group:
                continue
            values = group[metric].dropna().to_numpy(dtype=float)
            n = len(values)
            half = float(t.ppf(.975, n-1)*values.std(ddof=1)/np.sqrt(n)) if n > 1 else None
            summaries.append({'method': method, 'data_fingerprint': data, 'protocol_fingerprint': protocol,
                              'metric': metric, 'n': n, 'mean': float(values.mean()) if n else None,
                              'ci95_half_width': half})
    return summaries


def generate_reports(root: str | Path, output: str | Path, max_trajectory_figures: int = 24) -> dict:
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    searches, finals, diagnostics = [], [], []
    search_by_directory = {}
    for filename in sorted(root.rglob('search_summary.json')):
        row = json.loads(filename.read_text())
        search_by_directory[filename.parent] = row
        searches.append({'run': str(filename.parent), 'method': row['method'], 'seed': row['seed'],
                         'data_fingerprint': row['data_fingerprint'],
                         'protocol_fingerprint': row.get('protocol_fingerprint', 'legacy_unspecified'),
                         'charged_epochs': row['charged_epochs'], 'wall_seconds': row['wall_seconds'],
                         'gpu_seconds': row['gpu_seconds'], 'status': row['status'],
                         'synthetic_data': row.get('synthetic_data', False),
                         'selected_budget': row['best']['completed_budget'] if row['best'] else None,
                         'collapsed_to_single_class': row['best'].get('collapsed_to_single_class') if row['best'] else None,
                         'predicted_class_count': row['best'].get('predicted_class_count') if row['best'] else None,
                         'majority_baseline_macro_f1': row['best'].get('majority_baseline_macro_f1') if row['best'] else None,
                         'validation_macro_f1': row['best']['fitness'] if row['best'] else None})
        events = json.loads((filename.parent / 'evaluations.json').read_text())
        good = [e for e in events if e['fitness'] is not None]
        if good:
            highest = max(e['completed_budget'] for e in good)
            cumulative, incumbent, xs, ys, accuracies = 0., -np.inf, [], [], []
            for e in events:
                cumulative += e['wall_seconds']
                if e['completed_budget'] == highest and e['fitness'] is not None:
                    if e['fitness'] > incumbent:
                        incumbent = e['fitness']
                        accuracy = e.get('validation_accuracy')
                        if accuracy is None:
                            trajectory = filename.parent / 'candidates' / e['candidate'] / 'trajectory.json'
                            if trajectory.exists():
                                accuracy = next(r['metrics']['accuracy'] for r in json.loads(trajectory.read_text())
                                                if r['epoch'] == e['completed_budget'])
                    xs.append(e.get('search_wall_seconds', cumulative))
                    ys.append(incumbent)
                    accuracies.append(accuracy)
            fig, axes = plt.subplots(1, 2, figsize=(9, 3))
            axes[0].step(xs, ys, where='post')
            axes[0].set(ylabel=f'Best validation macro-F1 ({highest} epochs)')
            if all(a is not None for a in accuracies):
                axes[1].step(xs, accuracies, where='post')
            axes[1].set(ylabel='Validation accuracy of F1-selected incumbent')
            for ax in axes:
                ax.set_xlabel('Search wall seconds (legacy: evaluation wall)')
            _figure(fig, output / f'convergence_{len(searches)}')
    for filename in sorted(root.rglob('test_metrics.json')):
        row = json.loads(filename.read_text())
        search = search_by_directory.get(filename.parent.parent)
        final = {'run': str(filename.parent), **{k: v for k, v in row.items() if isinstance(v, (float, int, str))}}
        final.update({'method': search['method'] if search else 'fixed_model',
                      'data_fingerprint': row['data_fingerprint'],
                      'protocol_fingerprint': search.get('protocol_fingerprint', 'legacy_unspecified') if search else str(filename.parent),
                      'seed': search['seed'] if search else row['seed'],
                      'final_seed': row['seed'], 'search_wall_seconds': search['wall_seconds'] if search else None,
                      'search_epochs': search['charged_epochs'] if search else None})
        finals.append(final)
        cm = np.array(row['confusion_matrix'])
        classes = list(row['per_class'])
        fig, ax = plt.subplots(figsize=(max(5, len(classes)*.65), max(4, len(classes)*.55)))
        im = ax.imshow(cm, cmap='Blues')
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
               xlabel='Predicted class', ylabel='True class')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.colorbar(im, ax=ax)
        _figure(fig, output / f'confusion_{len(finals)}')
    if finals:
        comparable = pd.DataFrame(finals).dropna(subset=['search_wall_seconds'])
        if len(comparable):
            for data, group in comparable.groupby('data_fingerprint'):
                fig, ax = plt.subplots(figsize=(5, 3))
                for method, points in group.groupby('method'):
                    ax.scatter(points['search_wall_seconds'], points['accuracy'], label=method)
                ax.set(xlabel='Search wall seconds', ylabel='Final held-out accuracy')
                ax.legend(fontsize=8)
                _figure(fig, output / f'final_accuracy_compute_{str(data)[:10]}')
    for filename in sorted(root.rglob('ranking_diagnostics.json')):
        diagnostics.extend({'run': str(filename.parent), **r} for r in json.loads(filename.read_text())['rankings'])
    trajectories = []
    for filename in sorted(root.rglob('trajectory.json')):
        rows = json.loads(filename.read_text())
        if not rows:
            continue
        if len(trajectories) < max_trajectory_figures:
            fig, ax = plt.subplots(figsize=(5, 3))
            for c in rows[0]['per_class_recall']:
                ax.plot([r['epoch'] for r in rows], [r['per_class_recall'][c] for r in rows], label=c)
            ax.set(xlabel='Cumulative epoch', ylabel='Validation class recall', ylim=(-.02, 1.02))
            ax.legend(fontsize=7)
            _figure(fig, output / f'class_trajectory_{len(trajectories)}')
        trajectories.append({'run': str(filename.parent), 'epochs': len(rows),
                             'training_seconds': sum(r['training_seconds'] for r in rows),
                             'validation_seconds': sum(r['validation_seconds'] for r in rows)})
    pd.DataFrame(searches).to_csv(output / 'search_results.csv', index=False)
    pd.DataFrame(finals).to_csv(output / 'final_results.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output / 'ranking_results.csv', index=False)
    pd.DataFrame(trajectories).to_csv(output / 'cost_breakdown.csv', index=False)
    summaries = seed_summary(searches, ['validation_macro_f1'])
    final_summaries = seed_summary(finals, ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1'])
    pd.DataFrame(summaries).to_csv(output / 'seed_summaries.csv', index=False)
    pd.DataFrame(final_summaries).to_csv(output / 'final_seed_summaries.csv', index=False)
    save_json(output / 'summary.json', {'searches': searches, 'finals': finals, 'rankings': diagnostics,
                                       'seed_summaries': summaries, 'final_seed_summaries': final_summaries})
    return {'search_runs': len(searches), 'final_runs': len(finals), 'ranking_rows': len(diagnostics)}
