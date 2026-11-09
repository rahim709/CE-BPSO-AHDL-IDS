"""Run a prespecified reference from preparation through fixed-endpoint testing.

No settings or endpoints are chosen using these test scores. The existing test
partition is development-known; this study cannot create an independent holdout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from src.evaluation.metrics import classification_metrics
from src.utils.core import environment, read_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable; no silent substitution.')
    if args.output.exists():
        raise SystemExit('Preserve existing evidence; choose a fresh output directory.')
    args.output.mkdir(parents=True)
    cfg = read_config('configs/submission_evaluation.yaml')
    cfg['training']['device'] = args.device
    cfg['data']['processed'] = str(args.output / 'processed')
    config = args.output / 'config.json'
    save_json(config, cfg)
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
        'model_origin': 'fixed CNN16/GRU32 reference from prior validation; not NAS-selected',
        'endpoint': '200 epochs for every final seed; no early stopping or test-driven selection',
        'partition_status': 'same historical random-stratified split; test outcomes already known',
        'scope': 'repeated final-seed reference evaluation; not independent release or search comparison',
        'seed_interpretation': 'training variability on one fixed split; not independent dataset replication',
    }
    save_json(args.output / 'protocol.json', protocol)
    protocol_hash = hashlib.sha256((args.output / 'protocol.json').read_bytes()).hexdigest()
    save_json(args.output / 'environment.json', environment())
    env = dict(os.environ, MPLCONFIGDIR='/tmp/bpso-mpl', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    commands = []

    def execute(label: str, arguments: list[str]) -> None:
        command = [sys.executable, '-m', 'src.cli', *arguments]
        began = time.perf_counter()
        log = args.output / (label + '.log')
        with log.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        commands.append({'label': label, 'command': command, 'exit_code': result.returncode,
                         'seconds': time.perf_counter() - began, 'log': str(log)})
        save_json(args.output / 'commands.json', commands)
        if result.returncode:
            raise RuntimeError(f'{label} failed; inspect {log}')
        print(f'{label}: PASS ({commands[-1]["seconds"]:.1f}s)', flush=True)

    execute('prepare', ['prepare', '--config', str(config)])
    # Verify the new preparation retained the original source and row identities.
    original = Path(read_config('configs/gas_pipeline.yaml')['data']['processed'])
    prepared = Path(cfg['data']['processed'])
    metadata = json.loads((prepared / 'metadata.json').read_text())
    assert metadata['sources'] == json.loads((original / 'metadata.json').read_text())['sources']
    assert json.loads((prepared / 'split_indices.json').read_text()) == json.loads((original / 'split_indices.json').read_text())
    rows = []
    for seed in cfg['evaluation']['seeds']:
        seed_cfg = dict(cfg, final_seed=seed)
        seed_config = args.output / f'config_seed_{seed}.json'
        save_json(seed_config, seed_cfg)
        run = args.output / f'seed_{seed}'
        execute(f'train_test_{seed}', ['train', '--config', str(seed_config), '--output', str(run)])
        metrics = json.loads((run / 'test_metrics.json').read_text())
        with np.load(run / 'test_predictions.npz') as predictions:
            verified = classification_metrics(predictions['y'], predictions['probabilities'].argmax(1), metadata['classes'], '0')
        for key in verified:
            assert verified[key] == metrics[key], (seed, key)
        trajectory = json.loads((run / 'trajectory.json').read_text())
        assert len(trajectory) == cfg['final_epochs'] == metrics['epochs']
        assert metrics['seed'] == seed and not metrics['synthetic_data']
        assert {r['device'] for r in trajectory} == {args.device}
        rows.append({'seed': seed, 'metrics': metrics, 'validation_macro_f1': trajectory[-1]['macro_f1'],
                     'prediction_sha256': hashlib.sha256((run / 'test_predictions.npz').read_bytes()).hexdigest(),
                     'run': str(run)})
        save_json(args.output / 'measurements.json', rows)
    assert hashlib.sha256((args.output / 'protocol.json').read_bytes()).hexdigest() == protocol_hash

    def stats(values):
        return {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1))}

    summary = {'protocol_sha256': protocol_hash, 'n': len(rows), 'epochs': cfg['final_epochs'],
               'scope': protocol['scope'], 'partition_status': protocol['partition_status'],
               'metrics': {key: stats([r['metrics'][key] for r in rows]) for key in
                           ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1']},
               'normal_false_positive_rate': stats([r['metrics']['attack_detection']['fpr'] for r in rows]),
               'per_class_recall': {c: stats([r['metrics']['per_class'][c]['recall'] for r in rows]) for c in metadata['classes']},
               'all_prediction_metrics_verified': True,
               'test_evaluations': len(rows), 'complete': True}
    save_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
