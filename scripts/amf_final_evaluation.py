"""Frozen BPSO/UA winner evaluation on the development-known historical test.

Uses dedicated initialization seeds and a common 30-epoch endpoint. No post-test
choice of architecture, loss, endpoint or decision threshold is permitted here.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from src.training.engine import train_to, final_evaluate
from src.utils.core import digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='results/ua_amf_prediction_20261108')
    parser.add_argument('--benchmark', default='results/ua_amf_prediction_benchmark_20261108')
    parser.add_argument('--output', default='results/ua_amf_prediction_final_20261108')
    args = parser.parse_args()
    study, benchmark, root = Path(args.study), Path(args.benchmark), Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((study / 'resolved_config.json').read_text())
    selectors = []
    for variant in ['bpso', 'ua_amf_bpso']:
        for seed in [501, 502, 503]:
            folder = benchmark / variant / f'seed_{seed}'
            if not (folder / 'search_summary.json').exists():
                raise ValueError('Complete all benchmark searches before freezing final protocol')
            events = json.loads((folder / 'evaluations.json').read_text())
            eligible = [e for e in events if e['failure'] is None and e['fitness'] is not None and e['completed_budget'] == 30 and e['search_wall_seconds'] <= 30]
            winner = max(eligible, key=lambda e: e['fitness']) if eligible else None
            selectors.append({'variant': variant, 'search_seed': seed, 'architecture': winner['architecture'] if winner else None,
                              'selection_validation_f1': winner['fitness'] if winner else None})
    protocol = {'selection': selectors, 'training_seeds': [1501, 1502, 1503], 'epochs': 30, 'training': cfg['training'],
                'train_data': str(study / 'data'), 'max_stage_seconds': 180,
                'test_status': 'previously examined historical Gas Pipeline test split; not a blind independent test',
                'choice_rule': 'highest actually observed 30-epoch validation macro-F1 completed within 30 search wall seconds; stable event tie break',
                'contextual_baselines': 'existing MLP/CNN/GRU/CNN-GRU results preserved; different training-row/budget protocols are not matched comparisons'}
    if (root / 'frozen_protocol.json').exists() and json.loads((root / 'frozen_protocol.json').read_text()) != protocol:
        raise ValueError('Different final protocol requires new directory')
    # Freeze all selections and settings before reading this stage's test array.
    save_json(root / 'frozen_protocol.json', protocol)
    historic = Path('results/submission_final_evaluation/20261108/processed')
    parent = Path('results/research_extension/data/minmax')
    checks = {}
    for split in ['train', 'val']:
        with np.load(historic / f'{split}.npz') as left, np.load(parent / f'{split}.npz') as right:
            checks[split] = bool(np.array_equal(left['x'], right['x']) and np.array_equal(left['y'], right['y']))
    if not all(checks.values()):
        raise ValueError('Historical test preprocessing compatibility cannot be established')
    data = root / 'data'
    if not (data / 'metadata.json').exists():
        data.mkdir(exist_ok=True)
        for split in ['train', 'val']:
            shutil.copy2(study / 'data' / f'{split}.npz', data / f'{split}.npz')
        shutil.copy2(historic / 'test.npz', data / 'test.npz')
        meta = json.loads((study / 'data/metadata.json').read_text())
        original_meta = json.loads((historic / 'metadata.json').read_text())
        assert meta['classes'] == original_meta['classes']
        meta['historical_test_sha256'] = hashlib.sha256((data / 'test.npz').read_bytes()).hexdigest()
        meta['fingerprint'] = digest({'pilot': meta['fingerprint'], 'historical_test': meta['historical_test_sha256']})
        save_json(data / 'metadata.json', meta)
    save_json(root / 'preprocessing_compatibility.json', checks)
    state_path = root / 'stage_time.json'
    previous = json.loads(state_path.read_text())['elapsed_seconds'] if state_path.exists() else 0.
    start = time.perf_counter()
    deadline = start + max(0., 180 - previous)
    results = []
    try:
        for selector in selectors:
            if selector['architecture'] is None:
                results.append(dict(selector, status='no_common_fidelity_winner'))
                continue
            for seed in [1501, 1502, 1503]:
                key = digest({'architecture': selector['architecture'], 'seed': seed})
                folder = root / 'models' / key
                metrics_path = folder / 'test_metrics.json'
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text())
                else:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('Final-stage cap reached')
                    train_to(selector['architecture'], data, folder, 30, seed, cfg['training'], deadline=deadline)
                    metrics = final_evaluate(selector['architecture'], data, folder, 30, seed, cfg['training'], normal_label='0')
                results.append({'variant': selector['variant'], 'search_seed': selector['search_seed'], 'training_seed': seed,
                                'architecture': selector['architecture'], 'model_directory': str(folder), 'status': 'completed', **metrics})
    except TimeoutError as exc:
        save_json(root / 'stop.json', {'reason': str(exc), 'partial': True})
    finally:
        save_json(root / 'stage_time.json', {'elapsed_seconds': previous + time.perf_counter() - start})
        save_json(root / 'results.json', results)
    print('Completed final model references:', len(results), flush=True)


if __name__ == '__main__':
    main()
