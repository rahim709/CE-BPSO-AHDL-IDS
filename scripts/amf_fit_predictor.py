"""Fit and independently assess an architecture-grouped learning-curve predictor."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from src.optimization.curve_predictor import CurvePredictor, curve_features
from src.utils.core import digest, save_json


def predictor_context(cfg, meta):
    return digest({'data_fingerprint': meta['fingerprint'], 'training': cfg['training'], 'space': cfg['space']})


def fit_archive(root):
    if (root / 'curve_predictor.joblib').exists() and (root / 'predictor_report.json').exists():
        print('Using frozen predictor and its saved held-out assessment; choose a fresh study directory to refit.', flush=True)
        return CurvePredictor.load(root / 'curve_predictor.joblib')
    began = time.perf_counter()
    cfg = json.loads((root / 'resolved_config.json').read_text())
    meta = json.loads((root / 'data/metadata.json').read_text())
    endpoints = pd.read_csv(root / 'reliability_endpoints.csv')
    cohort = {r['index']: r for r in json.loads((root / 'cohort_definitions.json').read_text())}
    records = []
    for (index, seed), group in endpoints.groupby(['index', 'seed']):
        if 30 not in group.budget.values:
            continue
        entry = cohort[index]
        trajectory = json.loads((Path(group.iloc[0].candidate_path) / 'trajectory.json').read_text())
        target = next(r['macro_f1'] for r in trajectory if r['epoch'] == 30)
        partition = 'fit' if index < 12 else 'calibration' if index < 18 else 'test'
        for budget in [3, 10]:
            records.append({'architecture_id': entry['architecture_id'], 'index': int(index), 'seed': int(seed), 'partition': partition,
                            'budget': budget, 'features': curve_features(entry['architecture'], trajectory, budget).tolist(), 'target': target})
    model = CurvePredictor().fit(records, predictor_context(cfg, meta))
    model.save(root / 'curve_predictor.joblib')
    save_json(root / 'predictor_records.json', records)
    report, predictions = [], []
    for budget in [3, 10]:
        test = [r for r in records if r['partition'] == 'test' and r['budget'] == budget]
        if not test:
            continue
        means, radius, low, high = model.predict_features([r['features'] for r in test], budget)
        y = np.array([r['target'] for r in test])
        covered = (low <= y) & (y <= high)
        group_coverage = {}
        for row, mean, error, lo, hi, hit in zip(test, means, radius, low, high, covered):
            group_coverage.setdefault(row['architecture_id'], []).append(bool(hit))
            predictions.append({'index': row['index'], 'seed': row['seed'], 'budget': budget, 'target': row['target'],
                                'prediction': float(mean), 'radius': float(error), 'lower': float(lo), 'upper': float(hi), 'covered': bool(hit)})
        report.append({'budget': budget, 'test_architectures': len(group_coverage), 'test_seed_observations': len(test),
                       'nominal_coverage': .8, 'empirical_observation_coverage': float(covered.mean()),
                       'empirical_architecture_all_seeds_coverage': float(np.mean([all(v) for v in group_coverage.values()])),
                       'mean_interval_width': float(np.mean(high - low)), 'mae': float(np.mean(np.abs(y - means))),
                       'rmse': float(np.sqrt(np.mean((y - means)**2))),
                       'fit_architectures': model.models[budget]['fit_architectures'],
                       'calibration_architectures': model.models[budget]['calibration_architectures']})
    pd.DataFrame(predictions).to_csv(root / 'predictor_heldout_predictions.csv', index=False)
    result = {'calibration': report, 'forest_fit_wall_seconds': model.fit_wall_seconds,
              'total_build_wall_seconds': time.perf_counter() - began,
              'context': model.context, 'archive_architectures': sorted(model.archive_architectures),
              'interpretation': 'Held-out architectures only. Nominal 80% group split-conformal construction; observed coverage is descriptive with seven held-out architectures, not a population calibration guarantee under BPSO shift.'}
    save_json(root / 'predictor_report.json', result)
    print(json.dumps(result, indent=2))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/ua_amf_prediction_20261108')
    args = parser.parse_args()
    fit_archive(Path(args.input))


if __name__ == '__main__':
    main()
