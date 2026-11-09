"""Fresh, matched archive-free comparison. All calibration GPU work is in-search."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from src.optimization.search import run_search
from src.training.runtime import clear_caches
from src.utils.core import environment, read_config, save_json

ROOT=Path('results/ua_amf_online_20261108')
RUNTIME={'vectorized':True,'resident':True,'cache_data':True,'cache_model':True,
         'defer_scalar_sync':True,'validation_epochs':[3,10,30]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT))
    parser.add_argument('--diagnostic',action='store_true')
    args=parser.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    base=read_config('configs/gas_pipeline.yaml')
    base['data'].update(processed='results/ua_amf_prediction_20261108/data',normalization='minmax')
    base['training'].update(device='cuda',threads=1,class_weighting='balanced',gradient_clip=1.,prediction_epochs=[3,10,30])
    base['runtime']=RUNTIME
    base['search'].update(population=6,generations=100000,candidates=100000,brackets=10000,
                           epochs=30,min_epochs=3,budgets=[3,10,30])
    old=read_config('configs/ua_amf_efficient.yaml')
    variants={'bpso':('bpso',None),'bpso_sh':('bpso_sh',None),'dehb':('dehb',None),
              'offline_ua':('ua_amf_bpso','offline'),
              'online_prediction':('ua_amf_bpso','prediction'),
              'online_uncertainty':('ua_amf_bpso','uncertainty'),
              'online_full':('ua_amf_bpso','full')}
    protocol={'seeds':[601,602,603],'phases':['quality','compute'],'target':.6,'gpu_cap':20.,'wall_safety_cap':50.,
              'fixed_compute_gpu_limits':[5.,10.,20.],'fidelities':[3,10,30],'variants':variants,'runtime':RUNTIME,
              'online_fit':'CPU only, search-generated terminal labels only; no extra label GPU allocation.',
              'gates':{'min_fit':6,'min_calibration':4,'min_out_of_time_checks':4,'max_mae':.08,'relative_error_improvement':.1,
                       'minimum_absolute_error_improvement':.005,'minimum_spearman':.3,'min_target_range':.02,
                       'nominal_coverage':.8,'minimum_empirical_coverage':.75,'maximum_mean_width':.3},
              'online_coefficients':'fixed generic beta=.1 lambda=.05, median-normalized measured cost; not read from historical development.',
              'primary_cost':'actual GPU to first terminal target or cap; all misses retained/right-censored',
              'historical_offline_archive_gpu_seconds':403.8190207847874,'archive_free_external_gpu_seconds':0.,
              'diagnostic':{'seed':601,'gpu_cap':120.,'wall_cap':150.,'purpose':'activation/cold-start diagnostic only, not a matched comparison'},
              'maximum_primary_gpu_seconds':840.,'method_order':'seeded permutation; serial methods; no GPU test jobs concurrently'}
    canonical=json.loads(json.dumps(protocol))
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=canonical:
        raise ValueError('Use a fresh output for a changed protocol')
    if not (root/'protocol.json').exists():
        save_json(root/'protocol.json',protocol)
        save_json(root/'environment.json',environment())
        save_json(root/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').rglob('*.py')})
    experiments=[('diagnostic',601,'online_full')] if args.diagnostic else [(phase,seed,variant) for phase in protocol['phases'] for seed in protocol['seeds']
        for variant in np.random.default_rng(seed+(733 if phase=='quality' else 911)).permutation(list(variants))]
    for phase,seed,variant in experiments:
        method,mode=variants[variant]
        cfg=copy.deepcopy(base)
        cfg['seed']=seed
        cfg['compute']={'max_epochs':1000000,'max_gpu_seconds':120. if phase=='diagnostic' else 20.,
                        'max_seconds':150. if phase=='diagnostic' else 50.,'gpu_budget_mode':'execution'}
        cfg['search'].update(method=method,stop_f1=.6 if phase=='quality' else None)
        if mode=='offline':
            cfg['search'].update({k:old['search'][k] for k in ['promotion_policy','predictor_bundle','predictor_sha256','initial_probes',
                                  'probe_interval','min_improvement','uncertainty_beta','cost_weight']})
        elif mode:
            cfg['search'].update(promotion_policy='online',online={'mode':mode})
            assert 'predictor_bundle' not in cfg['search']
        folder=root/phase/variant/f'seed_{seed}'
        resolved=folder/'resolved_config.json'
        if resolved.exists() and json.loads(resolved.read_text())!=cfg:
            raise ValueError('Existing run has a different configuration')
        save_json(resolved,cfg)
        if (folder/'search_summary.json').exists():continue
        clear_caches()
        result=run_search(cfg,folder)
        print(phase,variant,seed,round(result['gpu_seconds'],3),result['target_reached'],
              result['best']['fitness'] if result['best'] else None,flush=True)
    print('Bounded archive-free experiment phase complete.',flush=True)


if __name__=='__main__':main()
