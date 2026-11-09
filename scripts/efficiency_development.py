"""Reuse historical validation trajectories, group-CV tune on fit IDs only.

All replay costs are historical proxies, not new measured search performance.
The entire historical archive cost remains explicitly charged in reports.
"""
import copy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from src.optimization.curve_predictor import CurvePredictor, curve_features
from src.utils.core import digest, save_json

ROOT = Path('results/ua_amf_efficiency_20261108')
SOURCE = Path('results/ua_amf_prediction_20261108')


def main():
    start = time.perf_counter()
    if (ROOT / 'frozen_development.json').exists():
        print('Using frozen development configuration.')
        return
    config = json.loads((SOURCE / 'resolved_config.json').read_text())
    meta = json.loads((Path(config['data']['processed']) / 'metadata.json').read_text())
    definitions = {r['index']: r for r in json.loads((SOURCE / 'cohort_definitions.json').read_text())}
    endpoints = pd.read_csv(SOURCE / 'reliability_endpoints.csv')
    records = []
    for (index, seed), group in endpoints.groupby(['index', 'seed']):
        if 30 not in group.budget.values:
            continue
        trajectory = json.loads((Path(group.iloc[0].candidate_path) / 'trajectory.json').read_text())
        sparse = [dict(r, macro_f1=r['macro_f1'] if r['epoch'] in [3,10,30] else None) for r in trajectory]
        target = next(r['macro_f1'] for r in trajectory if r['epoch'] == 30)
        for b in [3,10]:
            features = curve_features(definitions[index]['architecture'], sparse, b)
            features[-1] = 0.
            records.append({'index': int(index), 'seed': int(seed), 'architecture_id': definitions[index]['architecture_id'],
                            'partition': 'fit' if index < 12 else 'calibration' if index < 18 else 'test',
                            'budget': b, 'target': target, 'features': features.tolist(),
                            'remaining_cost_proxy': sum(r['training_seconds'] + r['validation_seconds'] for r in trajectory if r['epoch']>b)})
    cv = []
    for fold in range(3):
        data = []
        for r in records:
            if r['index'] >= 12:
                continue
            row = dict(r)
            block = row['index'] // 4
            row['partition'] = 'test' if block == fold else 'calibration' if block == (fold+1)%3 else 'fit'
            data.append(row)
        model = CurvePredictor().fit(data, 'development-only')
        for r in data:
            if r['partition'] != 'test':
                continue
            mean, radius, low, high = model.predict_features(r['features'], r['budget'])
            cv.append(dict(r, prediction=float(mean[0]), radius=float(radius[0])))
    scores = []
    for beta in [.1,.25,.5]:
        for weight in [.02,.05,.1,.2]:
            regrets, normalized_cost = [], []
            for budget in [3,10]:
                for seed in [42,43]:
                    group = [r for r in cv if r['budget']==budget and r['seed']==seed]
                    costs = np.array([r['remaining_cost_proxy'] for r in group])
                    values = np.array([r['prediction'] + beta*r['radius'] for r in group]) - weight*costs/np.median(costs)
                    index = int(np.argmax(values))
                    regrets.append(max(r['target'] for r in group)-group[index]['target'])
                    normalized_cost.append(costs[index]/np.median(costs))
            # Fixed development objective: regret plus .02 per median cost unit.
            scores.append({'beta': beta, 'cost_weight': weight, 'objective': float(np.mean(regrets)+.02*np.mean(normalized_cost)),
                           'mean_regret': float(np.mean(regrets)), 'normalized_cost': float(np.mean(normalized_cost))})
    chosen = min(scores, key=lambda r:(r['objective'],r['beta'],r['cost_weight']))
    context = digest({'data_fingerprint': meta['fingerprint'], 'training': config['training'], 'space': 'paper'})
    model = CurvePredictor().fit(records, context)
    model.time_invariant = True
    model.save(ROOT / 'sparse_predictor.joblib')
    calibration = []
    for b in [3,10]:
        held = [r for r in records if r['partition']=='test' and r['budget']==b]
        m,u,lo,hi = model.predict_features([r['features'] for r in held],b)
        y=np.array([r['target'] for r in held])
        calibration.append({'budget': b, 'observations': len(held), 'architectures': len(set(r['architecture_id'] for r in held)),
                            'coverage': float(np.mean((lo<=y)&(y<=hi))), 'mean_width':float(np.mean(hi-lo)), 'mae':float(np.mean(abs(m-y)))})
    pd.DataFrame(scores).to_csv(ROOT / 'development_grid.csv',index=False)
    save_json(ROOT / 'development_cv_predictions.json',cv)
    save_json(ROOT / 'sparse_calibration.json',calibration)
    old = json.loads((SOURCE/'reliability_summary.json').read_text())
    frozen = {'beta':chosen['beta'],'cost_weight':chosen['cost_weight'], 'initial_probes':2, 'probe_interval':2,
              'min_improvement':.01, 'selected_on':'Only archived architecture indices0..11, architecture-disjoint 3-fold CV; no new benchmark seeds or test outcomes.',
              'development_wall_seconds':time.perf_counter()-start,'new_archive_gpu_seconds':0.,
              'historical_archive_gpu_seconds':old['gpu_seconds'],'historical_archive_wall_seconds':old['wall_seconds'],
              'proxy_warning':'Historical replay costs are development proxies and are not reported as executed search timing.',
              'profile_acceptance':'Final weights must match original FP32; choose combined runtime only after profile passes.'}
    save_json(ROOT/'frozen_development.json',frozen)
    print(json.dumps(frozen,indent=2))


if __name__=='__main__':
    main()
