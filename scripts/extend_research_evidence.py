"""Frozen repeated-seed validation study: budgets, normalization and neural controls.

Uses train/validation only. Alternate data directories deliberately omit test.npz.
This bounded preset cohort does not replace the planned 25-architecture NAS study.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer
import torch
from src.data.preprocessing import load_files
from src.evaluation.metrics import classification_metrics
from src.evaluation.ranking import ranking_metrics
from src.models.controls import make_model
from src.training.engine import train_to, load_partition
from src.utils.core import digest, environment, read_config, save_json


def prepare_views(cfg, output):
    data = Path(cfg['data']['processed'])
    metadata = json.loads((data / 'metadata.json').read_text())
    saved = joblib.load(data / 'preprocessor.joblib')
    raw, sources = load_files(cfg['data']['files'])
    assert sources == metadata['sources']
    ids = json.loads((data / 'split_indices.json').read_text())
    base = [saved['transform'].transform(raw.loc[ids[s], saved['features']].replace([np.inf, -np.inf], np.nan))
            for s in ['train', 'val']]
    targets = [load_partition(data, s)[1] for s in ['train', 'val']]
    result = {}
    for normalization in cfg['extension']['normalizations']:
        path = output / 'data' / normalization
        path.mkdir(parents=True, exist_ok=True)
        if normalization == 'column_l2':
            arrays = [load_partition(data, s)[0] for s in ['train', 'val']]
            transform = None
        else:
            transform = (MinMaxScaler() if normalization == 'minmax' else Normalizer()).fit(base[0])
            arrays = [transform.transform(x).astype(np.float32) for x in base]
        meta = copy.deepcopy(metadata)
        meta.pop('fingerprint')
        meta['normalization'] = normalization
        meta['config']['normalization'] = normalization
        meta['extension'] = {'source_fingerprint': metadata['fingerprint'], 'test_array_available': False,
                             'normalization_fit': 'training only; original fitted imputer retained'}
        meta['fingerprint'] = digest(meta)
        save_json(path / 'metadata.json', meta)
        save_json(path / 'split_indices.json', {s: ids[s] for s in ['train', 'val']})
        for s, x, y in zip(['train', 'val'], arrays, targets):
            assert np.isfinite(x).all()
            np.savez_compressed(path / f'{s}.npz', x=x, y=y)
        if transform is not None:
            joblib.dump(transform, path / 'normalizer.joblib')
        assert not (path / 'test.npz').exists()
        result[normalization] = path
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/research_extension'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('Requested CUDA is unavailable; no silent device substitution.')
    cfg = read_config('configs/research_extension.yaml')
    cfg['training']['device'] = args.device
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text())['config'] != cfg:
        raise SystemExit('Output belongs to another protocol; choose a new directory.')
    reference = cfg['architecture']
    cohort = {'small': copy.deepcopy(reference), 'reference': copy.deepcopy(reference),
              'wide': copy.deepcopy(reference), 'stacked': copy.deepcopy(reference)}
    cohort['small']['cnn'][0]['filters'] = 8
    cohort['small']['rnn'][0]['units'] = 16
    cohort['wide']['cnn'][0]['filters'] = 32
    cohort['wide']['rnn'][0]['units'] = 64
    cohort['stacked']['cnn'].append(dict(reference['cnn'][0], filters=32, pool=1))
    protocol = {'config': cfg, 'cohort': cohort, 'test_access': False,
                'scope': 'four fixed hybrid presets, three training seeds; not full-space NAS',
                'long_budget': 'reference column-L2 all seeds; reference Min-Max seed42; chosen before new scores',
                'component_controls': 'MLP64/32, CNN16/flatten/32, GRU32; capacities and readouts differ',
                'comparisons': 'same fixed split; epochs matched, compute not equalized; exploratory only'}
    save_json(path, protocol)
    save_json(args.output / 'environment.json', environment())
    views, metadata = prepare_views(cfg, args.output)
    settings = cfg['extension']
    training = cfg['training']
    rows = []
    jobs = [(name, 'column_l2', seed, arch) for seed in settings['seeds'] for name, arch in cohort.items()]
    jobs += [('reference', view, seed, reference) for view in ['minmax', 'row_l2'] for seed in settings['seeds']]
    jobs += [(family, 'minmax', seed, dict(reference, model_family=family))
             for family in settings['component_families'] for seed in settings['seeds']]
    long_jobs = [('reference', 'column_l2', seed, reference) for seed in settings['long_column_seeds']]
    long_jobs += [('reference', 'minmax', seed, reference) for seed in settings['long_minmax_seeds']]
    def execute(name, view, seed, arch, budget):
        run = args.output / 'runs' / view / name / f'seed_{seed}'
        began = time.perf_counter()
        row = train_to(arch, views[view], run, budget, seed, training)
        with np.load(run / f'validation_epoch_{budget}.npz') as p:
            metrics = classification_metrics(p['y'], p['probabilities'].argmax(1), metadata['classes'], '0')
        assert abs(metrics['macro_f1'] - row['macro_f1']) < 1e-12
        item = {'preset': name, 'normalization': view, 'seed': seed, 'epoch': budget,
                'metrics': metrics, 'parameters': sum(p.numel() for p in make_model(arch, 26, 8).parameters()),
                'training_validation_seconds': row['elapsed_seconds'],
                'invocation_wall_seconds': time.perf_counter() - began, 'run': str(run),
                'prediction_sha256': hashlib.sha256((run / f'validation_epoch_{budget}.npz').read_bytes()).hexdigest()}
        rows.append(item)
        save_json(args.output / 'measurements.json', rows)
        print(f'{name}/{view}/seed{seed}/epoch{budget}: F1={metrics["macro_f1"]:.6f}', flush=True)
    for i, (name, view, seed, arch) in enumerate(jobs):
        for budget in settings['budgets']:
            execute(name, view, seed, arch, budget)
        save_json(args.output / 'progress.json', {'completed_short_runs': i+1, 'total_short_runs': len(jobs),
                                                 'long_runs_completed': 0, 'state': 'running'})
    for i, (name, view, seed, arch) in enumerate(long_jobs):
        execute(name, view, seed, arch, settings['long_epochs'])
        save_json(args.output / 'progress.json', {'completed_short_runs': len(jobs), 'total_short_runs': len(jobs),
                                                 'long_runs_completed': i+1, 'state': 'running'})
    ranks = []
    for seed in settings['seeds']:
        for early in [3, 10]:
            def scores(epoch):
                return np.array([next(r['metrics']['macro_f1'] for r in rows if r['preset']==name and
                    r['normalization']=='column_l2' and r['seed']==seed and r['epoch']==epoch) for name in cohort])
            ranks.append({'seed': seed, 'early_epoch': early, 'late_epoch': 30,
                          'scope': 'complete four-preset cohort only', **ranking_metrics(scores(early), scores(30), k=1)})
    save_json(args.output / 'ranking_results.json', ranks)
    curves = {}
    for name, view, seed, _ in jobs:
        run = args.output / 'runs' / view / name / f'seed_{seed}'
        curves[f'{name}/{view}/{seed}'] = [{'epoch': r['epoch'], 'macro_f1': r['macro_f1'],
            'loss': r['loss'], 'elapsed_seconds': r['elapsed_seconds'], 'per_class_recall': r['per_class_recall']}
            for r in json.loads((run / 'trajectory.json').read_text())]
    save_json(args.output / 'learning_curves.json', curves)
    save_json(args.output / 'progress.json', {'completed_short_runs': len(jobs), 'total_short_runs': len(jobs),
                                             'long_runs_completed': len(long_jobs), 'state': 'complete'})
    print('Completed validation-only extension; no test arrays were available.', flush=True)


if __name__ == '__main__':
    main()
