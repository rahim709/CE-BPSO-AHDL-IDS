"""Fixed-quality stops and actual GPU-budget tests, with a shared fast backend."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from src.optimization.search import run_search
from src.training.runtime import clear_caches
from src.utils.core import environment, save_json

ROOT=Path('results/ua_amf_efficiency_20261108')
SOURCE=Path('results/ua_amf_prediction_20261108')
RUNTIME={'vectorized':True,'resident':True,'cache_data':True,'cache_model':True,
         'defer_scalar_sync':True,'validation_epochs':[3,10,30]}


def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT))
    parser.add_argument('--development-root',default=str(ROOT))
    parser.add_argument('--phase',choices=['quality','compute','all'],default='all')
    args=parser.parse_args()
    ROOT=Path(args.output)
    development_root=Path(args.development_root)
    ROOT.mkdir(parents=True,exist_ok=True)
    dev=json.loads((development_root/'frozen_development.json').read_text())
    profile=json.loads((development_root/'profile_results_completed.json').read_text())
    combined=[r for r in profile if r['profile']=='combined']
    if len(combined)!=3 or not all(r['exact_weights'] for r in combined):
        raise ValueError('Combined runtime did not pass all development exact-weight checks')
    variants={
        'bpso':('bpso',{}),'bpso_sh':('bpso_sh',{}),
        'prediction_only':('ua_amf_bpso',{'uncertainty_beta':0.,'cost_weight':0.}),
        'ua_no_cost':('ua_amf_bpso',{'uncertainty_beta':dev['beta'],'cost_weight':0.}),
        'ua_no_uncertainty':('ua_amf_bpso',{'uncertainty_beta':0.,'cost_weight':dev['cost_weight']}),
        'ua_full':('ua_amf_bpso',{'uncertainty_beta':dev['beta'],'cost_weight':dev['cost_weight']}),
        'hyperband':('hyperband',{}),'dehb':('dehb',{}),
        'fixed_prediction':('ua_amf_bpso',{'promotion_policy':'learning_curve','uncertainty_beta':0.,'cost_weight':0.}),
    }
    protocol={'seeds':[601,602,603],'quality_targets':[.4,.5,.6],'gpu_cap':8.,'wall_safety_cap':20.,
              'fixed_compute_gpu_limits':[2.,4.,8.],'fidelity':[3,10,30],'population':6,
              'variants':variants,'runtime':RUNTIME,'minimum_acceptable_f1_difference':-.02,
              'coefficient_source':'frozen_development.json; coefficients fixed before fresh benchmark seeds',
              'model_sha256':hashlib.sha256((development_root/'sparse_predictor.joblib').read_bytes()).hexdigest(),
              'censoring':'Every seed retained; misses censored at common cap. Restricted time includes misses; do not average successes alone.',
              'historical_archive_gpu_seconds':dev['historical_archive_gpu_seconds'],
              'total_new_search_gpu_cap':864.,'final_model_rule':'Freeze full and SH strict-cap winners from compute study; independent seed1601 at30 epochs; no tuning on test.',
              'order':'Seeded permutation per experiment/target/seed, serial GPU jobs. AMP not used.'}
    canonical=json.loads(json.dumps(protocol))
    path=ROOT/'benchmark_protocol.json'
    if path.exists() and json.loads(path.read_text())!=canonical:
        raise ValueError('Changed protocol requires a fresh output root')
    if not path.exists():
        save_json(path,protocol)
        save_json(ROOT/'benchmark_environment.json',environment())
        save_json(ROOT/'benchmark_sources.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').rglob('*.py')})
    base=json.loads((SOURCE/'resolved_config.json').read_text())
    phases=['quality','compute'] if args.phase=='all' else [args.phase]
    for experiment in phases:
        for target in (protocol['quality_targets'] if experiment=='quality' else [None]):
            target_name=f'target_{target:.1f}' if target is not None else 'fixed'
            for seed in protocol['seeds']:
                for variant in np.random.default_rng(seed+int((target or 0)*1000)+511).permutation(list(variants)):
                    method,overrides=variants[variant]
                    cfg=copy.deepcopy(base)
                    cfg['seed']=seed
                    cfg['runtime']=RUNTIME
                    cfg['compute']={'max_epochs':1000000,'max_seconds':20.,'max_gpu_seconds':8.,'gpu_budget_mode':'execution'}
                    cfg['search'].update(method=method,population=6,generations=100000,candidates=100000,brackets=10000,
                                         min_epochs=3,epochs=30,budgets=[3,10,30],stop_f1=target)
                    if method=='ua_amf_bpso':
                        cfg['search'].update(promotion_policy='dynamic',predictor_bundle=str(development_root/'sparse_predictor.joblib'),
                                             predictor_sha256=protocol['model_sha256'],restart_fraction=.2,
                                             initial_probes=dev['initial_probes'],probe_interval=dev['probe_interval'],
                                             min_improvement=dev['min_improvement'])
                        cfg['search'].update(overrides)
                    folder=ROOT/experiment/target_name/variant/f'seed_{seed}'
                    if (folder/'search_summary.json').exists():
                        continue
                    save_json(folder/'resolved_config.json',cfg)
                    clear_caches()
                    result=run_search(cfg,folder)
                    print(experiment,target,variant,seed,result['target_reached'],round(result['gpu_seconds'],3),
                          result['best']['fitness'] if result['best'] else None,flush=True)
    print('Requested bounded phase complete.',flush=True)


if __name__=='__main__':
    main()
