"""Actual-time, fixed-protocol controls and predictive UA-AMF ablations.

Run after amf_reliability and amf_fit_predictor. Does not open IDS test data.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from src.optimization.search import run_search
from src.utils.core import environment, save_json

VARIANTS = {
    'bpso': ('bpso', {}), 'bpso_sh': ('bpso_sh', {}), 'random': ('random', {}),
    'hyperband': ('hyperband', {}), 'dehb': ('dehb', {}),
    'bpso_prediction': ('ua_amf_bpso', {'uncertainty_beta': 0., 'cost_weight': 0.}),
    'bpso_prediction_uncertainty': ('ua_amf_bpso', {'uncertainty_beta': 1., 'cost_weight': 0.}),
    'ua_amf_bpso': ('ua_amf_bpso', {'uncertainty_beta': 1., 'cost_weight': .05}),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', default='results/ua_amf_prediction_20261108')
    parser.add_argument('--output', default='results/ua_amf_prediction_benchmark_20261108')
    parser.add_argument('--seeds', nargs='+', type=int, default=[501, 502, 503])
    args = parser.parse_args()
    archive, root = Path(args.study), Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((archive / 'resolved_config.json').read_text())
    model = archive / 'curve_predictor.joblib'
    if not model.exists():
        raise ValueError('Fit and assess the predictor before running comparisons')
    protocol = {'archive': str(archive), 'seeds': args.seeds, 'variants': VARIANTS, 'budgets': [3, 10, 30],
                'max_seconds_per_run': 30, 'anytime_limits': [10, 20, 30], 'f1_targets': [.2, .4, .6],
                'predictor_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
                'variant_order': 'seeded permutation, seed+991; serial methods and no concurrent GPU jobs',
                'archive_cost': 'separate one-time reliability GPU/wall plus predictor fitting cost; show unamortized totals',
                'final_validation': 'strict-30-second selected architecture; common30 epochs; seeds1501,1502,1503; no IDS test tuning/evaluation'}
    canonical = json.loads(json.dumps(protocol))
    if (root / 'protocol.json').exists() and json.loads((root / 'protocol.json').read_text()) != canonical:
        raise ValueError('Changed benchmark protocol: choose new output')
    save_json(root / 'protocol.json', protocol)
    save_json(root / 'environment.json', environment())
    save_json(root / 'source_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').rglob('*.py')})
    for seed in args.seeds:
        for variant in np.random.default_rng(seed + 991).permutation(list(VARIANTS)):
            method, overrides = VARIANTS[variant]
            run_cfg = copy.deepcopy(cfg)
            run_cfg['seed'] = seed
            run_cfg['compute'] = {'max_epochs': 1000000, 'max_seconds': 30, 'max_gpu_seconds': 30}
            run_cfg['search'].update(method=method, population=6, generations=100000, candidates=100000, brackets=10000,
                                     epochs=30, min_epochs=3, budgets=[3, 10, 30])
            if method == 'ua_amf_bpso':
                run_cfg['search'].update(promotion_policy='learning_curve', predictor_bundle=str(model),
                                         predictor_sha256=protocol['predictor_sha256'], restart_fraction=.2,
                                         cost_unit_seconds=60., **overrides)
            folder = root / variant / f'seed_{seed}'
            resolved = folder / 'resolved_config.json'
            if resolved.exists() and json.loads(resolved.read_text()) != run_cfg:
                raise ValueError('Run directory config differs')
            save_json(resolved, run_cfg)
            if (folder / 'search_summary.json').exists():
                continue
            result = run_search(run_cfg, folder)
            print(variant, seed, result['wall_seconds'], result['charged_epochs'], result['best']['fitness'] if result['best'] else None, flush=True)
    print('All bounded searches completed. Run scripts.amf_analyze for verification and reports.', flush=True)


if __name__ == '__main__':
    main()
