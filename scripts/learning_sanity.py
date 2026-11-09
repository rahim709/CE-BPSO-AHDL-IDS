"""Validation-only factorial check: architecture preset, training budget, class weighting.

Never opens test.npz and never chooses settings using held-out test performance.
Synthetic observations remain marked; no paper reproduction or NAS claim follows.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import pandas as pd
from src.training.engine import train_to
from src.utils.core import environment, read_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 2:
        raise ValueError('Need at least two epochs to audit initial versus later behavior')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = Path('results/learning_sanity') / stamp
    root.mkdir(parents=True)
    latest = json.loads(Path('results/synthetic_datasets/latest.json').read_text())
    synthetic_root = Path(latest['directory'])
    data = {'gas_pipeline_real': Path('datasets/processed/gas_pipeline'),
            **{f'{name}_synthetic': synthetic_root / name / 'processed' for name in ['ton_iot', 'bot_iot', 'swat']}}
    reference = read_config('configs/learning_sanity.yaml')
    tiny = copy.deepcopy(read_config('configs/debug.yaml')['architecture'])
    tiny['batch_size'] = reference['architecture']['batch_size']
    architectures = {'tiny': tiny, 'reference': reference['architecture']}
    rows = []
    save_json(root / 'environment.json', environment())
    save_json(root / 'protocol.json', {'epochs': args.epochs, 'seed': args.seed, 'architectures': architectures,
              'training': reference['training'], 'weightings': ['none', 'balanced'], 'test_access': False,
              'interpretation': 'Learning sanity comparison only. Different model sizes have different compute costs.'})
    for dataset, directory in data.items():
        metadata = json.loads((directory / 'metadata.json').read_text())
        for model_name, architecture in architectures.items():
            for weighting in ['none', 'balanced']:
                training = dict(reference['training'], class_weighting=weighting)
                run = root / dataset / f'{model_name}_{weighting}'
                save_json(run / 'resolved_config.json', {'architecture': architecture, 'training': training,
                          'data': metadata['config'], 'data_fingerprint': metadata['fingerprint'], 'seed': args.seed})
                began = time.perf_counter()
                row = train_to(architecture, directory, run, args.epochs, args.seed, training)
                trajectory = json.loads((run / 'trajectory.json').read_text())
                early = next(r for r in trajectory if r['epoch'] == 2)
                item = {'dataset': dataset, 'synthetic_data': bool(metadata['config'].get('synthetic', False)),
                        'model': model_name, 'class_weighting': weighting, 'epochs': args.epochs,
                        'validation_macro_f1': row['macro_f1'], 'epoch2_macro_f1': early['macro_f1'],
                        'validation_accuracy': row['metrics']['accuracy'],
                        'validation_macro_recall': row['metrics']['macro_recall'],
                        'predicted_class_count': row['metrics']['predicted_class_count'],
                        'zero_recall_class_count': len(row['metrics']['zero_recall_classes']),
                        'collapsed_to_single_class': row['metrics']['collapsed_to_single_class'],
                        'majority_baseline_macro_f1': row['majority_baseline']['macro_f1'],
                        'beats_majority_macro_f1': row['beats_majority_macro_f1'],
                        'wall_seconds': time.perf_counter()-began, 'run': str(run),
                        'per_class_recall': row['per_class_recall']}
                rows.append(item)
                save_json(root / 'results.json', rows)
                pd.DataFrame([{k:v for k,v in r.items() if k != 'per_class_recall'} for r in rows]).to_csv(root / 'results.csv', index=False)
                print(f"{dataset}/{model_name}/{weighting}: val macro-F1={row['macro_f1']:.4f}, predicted classes={row['metrics']['predicted_class_count']}", flush=True)
    save_json('results/learning_sanity/latest.json', {'directory': str(root), 'results': str(root / 'results.json'),
              'validation_only': True, 'runs_completed': len(rows)})
    print(f'Completed{len(rows)} validation-only checks:{root}', flush=True)


if __name__ == '__main__':
    main()
