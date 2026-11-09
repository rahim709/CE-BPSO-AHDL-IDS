"""Validation-only conventional controls and an evidence audit for the LaTeX review.

Run from the repository root. Historical test predictions are checked, never used
to select or fit the new controls. No new held-out model evaluation is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data.preprocessing import load_files, overlap_diagnostics
from src.evaluation.metrics import classification_metrics, majority_baseline
from src.training.engine import load_partition
from src.utils.core import environment, read_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/submission_review/validation_controls'))
    parser.add_argument('--evidence-output', type=Path, default=Path('docs/latex/evidence'))
    args = parser.parse_args()
    out = args.evidence_output
    out.mkdir(parents=True, exist_ok=True)
    runs = args.output
    if (runs / 'controls.json').exists():
        raise SystemExit('Controls already exist; preserve the record or use a fresh directory.')
    runs.mkdir(parents=True, exist_ok=True)
    cfg = read_config('configs/gas_pipeline.yaml')['data']
    data = Path(cfg['processed'])
    meta = json.loads((data / 'metadata.json').read_text())
    raw, sources = load_files(cfg['files'])
    assert sources == meta['sources'], 'Raw data changed since frozen preprocessing'
    ids = json.loads((data / 'split_indices.json').read_text())
    parts = [raw.loc[ids[n]] for n in ['train', 'val', 'test']]
    features, target, classes = meta['features'], cfg['target'], meta['classes']
    overlaps = {'all_features': overlap_diagnostics(parts, features, target),
                'without_time': overlap_diagnostics(parts, [f for f in features if f != 'time'], target)}
    # The source paper defines time as query-response duration, not a timestamp.
    # Sorting it measures duration coverage, never chronological generalization.
    ordered = raw.sort_values('time', kind='stable')
    cuts = [0, int(.6 * len(raw)), int(.8 * len(raw)), len(raw)]
    interval_coverage = {n: ordered.iloc[cuts[i]:cuts[i+1]][target].astype(str).value_counts().to_dict()
                  for i, n in enumerate(['train', 'val', 'test'])}
    save_json(out / 'leakage_audit.json', {'overlaps': overlaps, 'interval_sorted_class_coverage': interval_coverage,
              'note': 'Duration-sorted coverage only. Morris and Gao (2014), Sec. 5.2, defines time as query-response interval; actual capture chronology is unavailable. No chronological training was run.'})
    x, y = load_partition(data, 'train')
    vx, vy = load_partition(data, 'val')
    encoded = meta['encoded_features']
    assert x.shape[1] == len(encoded) == 26
    indices = {'all': np.arange(x.shape[1]),
               'without_time': np.array([i for i, n in enumerate(encoded) if n != 'numeric__time']),
               'time_only': np.array([encoded.index('numeric__time')])}
    record = {'scope': 'real Gas validation only; one frozen split and seed; no test evaluation',
              'seed': 42, 'data_fingerprint': meta['fingerprint'], 'environment': environment(),
              'preprocessing': 'frozen train-only column L2; LR additionally train-fitted median and standard scaling',
              'models': [], 'majority': majority_baseline(y, vy, classes)}
    for view, columns in indices.items():
        for name in ['random_forest', 'logistic_regression']:
            if name == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                                random_state=42, n_jobs=1)
            else:
                model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
            began = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(x[:, columns], y)
            fit_seconds = time.perf_counter() - began
            began = time.perf_counter()
            prediction = model.predict(vx[:, columns])
            predict_seconds = time.perf_counter() - began
            metrics = classification_metrics(vy, prediction, classes, '0')
            path = runs / f'{name}_{view}.npz'
            np.savez_compressed(path, y=vy, prediction=prediction)
            record['models'].append({'method': name, 'features': view, 'feature_count': len(columns),
                 'fit_seconds': fit_seconds, 'validation_predict_seconds': predict_seconds,
                 'warnings': [str(w.message) for w in caught],
                 'parameters': {k: repr(v) for k, v in model.get_params(deep=True).items()},
                 'metrics': metrics, 'predictions': str(path)})
            print(name, view, metrics['macro_f1'], flush=True)
    save_json(runs / 'controls.json', record)
    save_json(out / 'validation_controls.json', record)
    audits = []
    for path in sorted(Path('results').rglob('test_predictions.npz')):
        metric_path = path.parent / 'test_metrics.json'
        if not metric_path.exists():
            continue
        saved = json.loads(metric_path.read_text())
        with np.load(path) as p:
            measured = classification_metrics(p['y'], p['probabilities'].argmax(1), list(saved['per_class']))
        keys = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1', 'confusion_matrix']
        matches = {k: bool(np.allclose(saved[k], measured[k], atol=1e-12, rtol=0)) for k in keys}
        audits.append({'run': str(path.parent), 'matches': matches,
                       'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    save_json(out / 'historical_prediction_audit.json', audits)
    assert all(all(a['matches'].values()) for a in audits), 'Historical metric mismatch'
    inventory = []
    for directory in ['src', 'scripts', 'configs', 'tests', 'docs', 'results']:
        for path in sorted(Path(directory).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in {'.pyc', '.tmp'}:
                continue
            if path == out / 'inventory.json':
                continue
            payload = path.read_bytes()
            entry = {'path': str(path), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            if path.suffix == '.json':
                try:
                    json.loads(payload)
                    entry['json_valid'] = True
                except (ValueError, UnicodeDecodeError):
                    entry['json_valid'] = False
            inventory.append(entry)
    save_json(out / 'inventory.json', inventory)
    save_json(out / 'data_metadata.json', meta)
    print(f'Audited {len(audits)} held-out prediction files and inventoried {len(inventory)} files.')


if __name__ == '__main__':
    main()
