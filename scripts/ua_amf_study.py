"""Bounded real Gas Pipeline pilot; never loads historical test arrays.

The reduced-row pilot cannot establish full-data NAS performance. All methods
use identical row IDs, training-fitted preprocessing, search space and compute caps.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.optimization.search import run_search
from src.evaluation.metrics import classification_metrics
from src.training.engine import train_to
from src.utils.core import digest, environment, read_config, save_json

VARIANTS = {
    'bpso': ('bpso', {}), 'bpso_sh': ('bpso_sh', {}), 'random': ('random', {}),
    'hyperband': ('hyperband', {}), 'dehb': ('dehb', {}),
    'replicated_fixed': ('ua_amf_bpso', {'uncertainty_beta': 0., 'cost_weight': 0., 'restart_fraction': 0.}),
    'uncertainty_only': ('ua_amf_bpso', {'cost_weight': 0., 'restart_fraction': 0.}),
    'budget_only': ('ua_amf_bpso', {'uncertainty_beta': 0.}),
    'ua_amf_bpso': ('ua_amf_bpso', {}),
}


def pilot_data(source, output, train_rows, val_rows):
    meta = json.loads((source / 'metadata.json').read_text())
    provenance = {}
    for split, count in [('train', train_rows), ('val', val_rows)]:
        with np.load(source / f'{split}.npz') as data:
            x, y = data['x'], data['y']
        indices, _ = train_test_split(np.arange(len(y)), train_size=count, stratify=y, random_state=271828)
        indices.sort()
        provenance[split] = {'source_indices': indices.tolist(), 'source_sha256': hashlib.sha256((source / f'{split}.npz').read_bytes()).hexdigest(),
                             'class_counts': np.bincount(y[indices], minlength=len(meta['classes'])).tolist()}
        if len(np.unique(y[indices])) != len(meta['classes']):
            raise ValueError('Pilot must retain all eight classes')
        np.savez_compressed(output / f'{split}.npz', x=x[indices], y=y[indices])
    meta['parent_fingerprint'] = meta['fingerprint']
    meta['pilot_subsampling'] = provenance
    meta['fingerprint'] = digest({'parent': meta['parent_fingerprint'], 'subsampling': provenance})
    save_json(output / 'metadata.json', meta)


def summarize(output):
    rows, metric_rows = [], []
    meta = json.loads((output / 'data/metadata.json').read_text())
    for path in sorted(output.glob('runs/*/seed_*/search_summary.json')):
        s = json.loads(path.read_text())
        best = s['best']
        promotions = path.parent / 'promotions.json'
        records = json.loads(promotions.read_text()) if promotions.exists() else []
        assessments = list(path.parent.glob('assessment/seed_*/trajectory.json'))
        values = [json.loads(p.read_text())[-1]['macro_f1'] for p in assessments]
        events = json.loads((path.parent / 'evaluations.json').read_text())
        promoted = {e['candidate'] for e in events if e.get('completed_budget', 0) > 1 and e.get('charged_epochs', 0) < e.get('requested_budget', 0)}
        for trajectory in assessments:
            with np.load(trajectory.parent / 'validation_epoch_2.npz') as pred:
                metrics = classification_metrics(pred['y'], pred['probabilities'].argmax(1), meta['classes'], '0')
            save_json(trajectory.parent / 'validation_metrics.json', metrics)
            metric_rows.append({'variant': path.parent.parent.name, 'search_seed': s['seed'], 'assessment_seed': int(trajectory.parent.name.split('_')[-1]),
                               'parameters': json.loads(trajectory.read_text())[-1]['parameters'], **metrics})
        row = {k: s.get(k) for k in ['seed', 'wall_seconds', 'charged_epochs', 'gpu_seconds', 'unique_architectures', 'peak_gpu_memory_bytes', 'selection_complete', 'scheduler_and_setup_seconds']}
        row.update(variant=path.parent.parent.name, validation_macro_f1=best['fitness'] if best else None,
                   assessment_macro_f1=float(np.mean(values)) if values else None,
                   promoted_candidates=(sum(len(r['selected_indices']) for r in records) if records else len(promoted)),
                   assessment_wall_seconds=sum(json.loads((p.parent / 'cost.json').read_text())['wall_seconds'] for p in assessments),
                   selection_regret=None)
        rows.append(row)
    save_json(output / 'validation_assessments.json', metric_rows)
    controls = []
    for path in sorted(output.glob('controls/*/seed_*/trajectory.json')):
        row = json.loads(path.read_text())[-1]
        with np.load(path.parent / 'validation_epoch_2.npz') as pred:
            metrics = classification_metrics(pred['y'], pred['probabilities'].argmax(1), meta['classes'], '0')
        controls.append({'family': path.parent.parent.name, 'seed': int(path.parent.name.split('_')[-1]), 'parameters': row['parameters'], **metrics})
    save_json(output / 'validation_controls.json', controls)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    times = frame[frame.variant == 'bpso'].set_index('seed').wall_seconds.to_dict()
    frame['saving_vs_bpso_percent'] = [100 * (times[r.seed] - r.wall_seconds) / times[r.seed] for r in frame.itertuples()]
    frame.to_csv(output / 'seed_results.csv', index=False)
    save_json(output / 'seed_results.json', frame.astype(object).where(pd.notnull(frame), None).to_dict('records'))
    # JSON serializer disallows NaN; CSV preserves missing measurements explicitly.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, group in frame.groupby('variant'):
        ax.scatter(group.wall_seconds, group.assessment_macro_f1, label=variant)
    ax.set(xlabel='Measured search wall time (s)', ylabel='Common validation assessment macro-F1', title='Reduced-row, two-epoch CPU pilot')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / 'f1_vs_time.png', dpi=160)
    plt.close(fig)
    frame.groupby('variant').agg({'wall_seconds': ['mean', 'std'], 'assessment_macro_f1': ['mean', 'std'], 'charged_epochs': 'mean'}).to_csv(output / 'comparison.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/ua_amf_pilot')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--train-rows', type=int, default=1536)
    parser.add_argument('--val-rows', type=int, default=768)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'seeds': args.seeds, 'train_rows': args.train_rows, 'val_rows': args.val_rows, 'device': args.device,
                'epochs': 2, 'epoch_cap': 24, 'seconds_cap': 60, 'variants': VARIANTS,
                'scope': 'real-data reduced-row feasibility pilot; no test evaluation',
                'assessment': 'selected architecture only; validation; seeds 90042,90043,90044; two epochs; separate cost ledger',
                'variant_order': 'seeded permutation with seed + 777; fixed before observations',
                'gate': 'No full-scale study unless full method improves assessed F1 or reduces measured search time without F1 loss versus bpso_sh; exploratory only'}
    if (output / 'protocol.json').exists() and json.loads((output / 'protocol.json').read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('Use a fresh output directory for a changed protocol')
    save_json(output / 'protocol.json', protocol)
    save_json(output / 'environment.json', environment())
    save_json(output / 'source_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for root in ['src', 'scripts', 'configs'] for p in Path(root).rglob('*') if p.suffix in {'.py', '.yaml'}})
    data = output / 'data'
    data.mkdir(exist_ok=True)
    if not (data / 'metadata.json').exists():
        pilot_data(Path('datasets/processed/gas_pipeline'), data, args.train_rows, args.val_rows)
    base = read_config('configs/ua_amf.yaml')
    base['data']['processed'] = str(data)
    base['training'].update(device=args.device, threads=1)
    base['search'].update(epochs=2, min_epochs=1, budgets=[1, 2], population=3, generations=1000, candidates=10000, brackets=1000)
    base['compute'] = {'max_epochs': 24, 'max_seconds': 60}
    try:
        for seed in args.seeds:
            for variant in np.random.default_rng(seed + 777).permutation(list(VARIANTS)):
                method, overrides = VARIANTS[variant]
                cfg = copy.deepcopy(base)
                cfg['seed'] = seed
                cfg['search'].update(method=method, **overrides)
                folder = output / 'runs' / variant / f'seed_{seed}'
                save_json(folder / 'resolved_config.json', cfg)
                summary_path = folder / 'search_summary.json'
                result = json.loads(summary_path.read_text()) if summary_path.exists() else run_search(cfg, folder)
                print(variant, seed, result['charged_epochs'], round(result['wall_seconds'], 3), result['best']['fitness'] if result['best'] else None, flush=True)
                if result['selection_complete']:
                    for assess_seed in [90042, 90043, 90044]:
                        start = time.perf_counter()
                        assessment = folder / 'assessment' / f'seed_{assess_seed}'
                        row = train_to(result['best']['architecture'], data, assessment, 2, assess_seed, cfg['training'])
                        if not (assessment / 'cost.json').exists():
                            save_json(assessment / 'cost.json', {'wall_seconds': time.perf_counter() - start, 'epochs': 2, 'parameters': row['parameters'],
                                      'normal_fpr': row['metrics'].get('false_positive_rate'), 'metrics': row['metrics']})
            # Conventional detector controls on validation only, protocol fixed above.
            for family in ['mlp', 'cnn']:
                arch = dict(base['architecture'], model_family=family, batch_size=256, optimizer='adam', learning_rate=.001)
                train_to(arch, data, output / 'controls' / family / f'seed_{seed}', 2, seed, base['training'])
    finally:
        summarize(output)


if __name__ == '__main__':
    main()
