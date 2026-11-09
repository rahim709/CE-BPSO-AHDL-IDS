"""Prespecified real-data 25-architecture reliability study, resumable and bounded."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.optimization.encoding import BinaryCodec
from src.optimization.evaluator import Evaluator, BudgetExhausted
from src.evaluation.ranking import ranking_metrics
from src.utils.core import digest, environment, read_config, save_json


def prepare_view(root):
    target = root / 'data'
    if (target / 'metadata.json').exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    source = Path('results/research_extension/data/minmax')
    meta = json.loads((source / 'metadata.json').read_text())
    provenance = {}
    for split, n in [('train', 8192), ('val', 4096)]:
        with np.load(source / f'{split}.npz') as data:
            x, y = data['x'], data['y']
        indices, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=39173)
        indices.sort()
        assert len(np.unique(y[indices])) == 8
        np.savez_compressed(target / f'{split}.npz', x=x[indices], y=y[indices])
        provenance[split] = {'parent_indices': indices.tolist(), 'parent_sha256': hashlib.sha256((source / f'{split}.npz').read_bytes()).hexdigest()}
        meta['splits'][split] = {'rows': n, 'classes': {c: int((y[indices] == i).sum()) for i, c in enumerate(meta['classes'])}}
    meta['parent_fingerprint'] = meta['fingerprint']
    meta['pilot'] = {'provenance': provenance, 'scaler_fit': 'full parent training only; shared by every method', 'test_available': False}
    meta['splits'].pop('test', None)
    meta['fingerprint'] = digest({'parent': meta['fingerprint'], 'provenance': provenance})
    save_json(target / 'metadata.json', meta)


def study_config(root, device):
    cfg = read_config('configs/gas_pipeline.yaml')
    cfg['data']['processed'] = str(root / 'data')
    cfg['data']['normalization'] = 'minmax'
    cfg['training'].update(device=device, threads=1, class_weighting='balanced', gradient_clip=1., prediction_epochs=[3, 10, 30])
    cfg['compute'] = {'max_epochs': 2250, 'max_seconds': 480, 'max_gpu_seconds': 420}
    return cfg


def extract(root):
    cfg = json.loads((root / 'resolved_config.json').read_text())
    codec = BinaryCodec('paper')
    evaluator = Evaluator(cfg, root / 'cohort', codec)
    cohort = json.loads((root / 'cohort_definitions.json').read_text())
    records, reports = [], []
    for entry in cohort:
        for seed in [42, 43, 44]:
            path = evaluator.candidate_path(entry['architecture'], seed) / 'trajectory.json'
            if not path.exists():
                continue
            rows = json.loads(path.read_text())
            for budget in [3, 10, 30]:
                row = next((r for r in rows if r['epoch'] == budget), None)
                if row:
                    records.append({'index': entry['index'], 'architecture_id': entry['architecture_id'], 'seed': seed, 'budget': budget,
                                    'macro_f1': row['macro_f1'], 'parameters': row['parameters'],
                                    'gpu_seconds': row['gpu_seconds'], 'elapsed_seconds': row['elapsed_seconds'],
                                    'peak_gpu_memory_bytes': row['peak_gpu_memory_bytes'], 'candidate_path': str(path.parent)})
    frame = pd.DataFrame(records)
    frame.to_csv(root / 'reliability_endpoints.csv', index=False)
    if not frame.empty:
        for seed, group in frame.groupby('seed'):
            matrix = group.pivot(index='index', columns='budget', values='macro_f1')
            for early in [3, 10]:
                if early not in matrix or 30 not in matrix:
                    continue
                paired = matrix[[early, 30]].dropna()
                if len(paired) >= 2:
                    reports.append({'seed': int(seed), 'early_budget': early, 'reference_budget': 30,
                                    'cohort_complete': len(paired) == 25,
                                    **ranking_metrics(paired[early].values, paired[30].values, k=5)})
    summary = {'requested_architectures': 25, 'requested_seeds': [42, 43, 44], 'requested_budgets': [3, 10, 30],
               'completed_30_epoch_trajectories': sum(r['budget'] == 30 for r in records), 'requested_terminal_trajectories': 75,
               'charged_epochs': evaluator.used_epochs, 'wall_seconds': evaluator.previous_wall,
               'gpu_seconds': sum(e['gpu_seconds'] for e in evaluator.events), 'rankings': reports,
               'failed_evaluations': sum(bool(e['failure']) for e in evaluator.events),
               'scope': 'Prespecified 8192/4096 real-data pilot, full 73-bit space; parent training-fitted MinMax and balanced loss. No test array.'}
    save_json(root / 'reliability_summary.json', summary)
    print(json.dumps(summary), flush=True)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/ua_amf_prediction_20261108')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.extract_only:
        extract(root)
        return
    protocol = {'architecture_count': 25, 'cohort_seed': 271828, 'training_seeds': [42, 43, 44], 'budgets': [3, 10, 30],
                'device': args.device, 'train_rows': 8192, 'val_rows': 4096, 'max_wall_seconds': 480, 'max_gpu_seconds': 420,
                'data_choice': 'MinMax/balanced fixed from pre-existing validation evidence, before current measurements; old column-L2 studies retained',
                'execution_order': 'seed, architecture, ascending cumulative rung; partial results retained at cap',
                'predictor_split': 'architecture index 0..11 fit, 12..17 calibration, 18..24 held-out; every seed of an architecture stays in one partition',
                'prediction': 'forest mean, separate split-conformal absolute residual radius at nominal 80%; no own terminal target in features',
                'score': 'predicted F1 + beta*interval_radius - lambda*(predicted extra GPU seconds / 60); beta=1 lambda=0.05 fixed before data',
                'benchmark': 'seeds 501,502,503; per-run wall cap 30s; max fidelity30; anytime wall limits10/20/30; targets0.2,0.4,0.6; no tuning on held-out predictor architectures or IDS test'}
    if (root / 'protocol.json').exists() and json.loads((root / 'protocol.json').read_text()) != protocol:
        raise ValueError('Changed protocol requires new output')
    save_json(root / 'protocol.json', protocol)
    save_json(root / 'environment.json', environment())
    save_json(root / 'launch_source_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').rglob('*.py')})
    prepare_view(root)
    cfg = study_config(root, args.device)
    save_json(root / 'resolved_config.json', cfg)
    codec, rng = BinaryCodec('paper'), np.random.default_rng(271828)
    cohort, seen = [], set()
    while len(cohort) < 25:
        bits = codec.sample(rng)
        architecture = codec.decode(bits)
        key = digest(architecture)
        if key not in seen:
            seen.add(key)
            cohort.append({'index': len(cohort), 'architecture_id': key, 'bits': codec.encode(architecture).tolist(), 'architecture': architecture})
    save_json(root / 'cohort_definitions.json', cohort)
    evaluator = Evaluator(cfg, root / 'cohort', codec)
    try:
        for seed in [42, 43, 44]:
            for entry in cohort:
                for budget in [3, 10, 30]:
                    score = evaluator.evaluate(np.asarray(entry['bits']), budget, seed)
                    print(seed, entry['index'], budget, score, flush=True)
                    if not np.isfinite(score):
                        break
    except BudgetExhausted as exc:
        save_json(root / 'stop.json', {'reason': str(exc), 'partial': True})
    finally:
        evaluator.persist()
        extract(root)


if __name__ == '__main__':
    main()
