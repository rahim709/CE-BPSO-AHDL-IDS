"""Explicit, charged continuation of the 8-second pilot to a 20-second horizon.

Old experiment directories remain untouched. Imported ledgers retain all spent
compute, including failed partial-fidelity attempts; identities are migrated with
provenance because only the resource allowance and output directory change.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from src.optimization.search import run_search
from src.training.runtime import clear_caches
from src.utils.core import digest, save_json

ROOT=Path('results/ua_amf_efficiency_20261108')


def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot-root',default=str(ROOT))
    parser.add_argument('--phase',choices=['quality','compute','all'],default='all')
    args=parser.parse_args()
    ROOT=Path(args.pilot_root)
    protocol=json.loads((ROOT/'benchmark_protocol.json').read_text())
    followup=ROOT/'followup20'
    followup.mkdir(exist_ok=True)
    declaration={'reason':'8-second target0.4 pilot had low attainment in standard/SH controls; exploratory feasibility extension, not a confirmatory replication.',
                 'seeds':protocol['seeds'],'variants':list(protocol['variants']), 'gpu_cap_total':20.,'wall_cap_total':50.,
                 'quality_target':.6,'phases':['quality','compute'],'max_additional_gpu_seconds':648.,
                 'unchanged':['algorithm','coefficients','training','data','seeds','initial swarm'],
                 'accounting':'All original epochs/GPU/wall retained. No resetting spent compute. Original outputs remain read-only.'}
    if not (followup/'protocol.json').exists():
        save_json(followup/'protocol.json',declaration)
        save_json(followup/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').rglob('*.py')})
    for phase in (['quality','compute'] if args.phase=='all' else [args.phase]):
        target_name='target_0.6' if phase=='quality' else 'fixed'
        for seed in protocol['seeds']:
            for variant in np.random.default_rng(seed+2381).permutation(list(protocol['variants'])):
                old=ROOT/phase/target_name/variant/f'seed_{seed}'
                if not (old/'search_summary.json').exists():
                    raise ValueError(f'Finish original bounded pilot before continuation: {old}')
                dest=followup/phase/variant/f'seed_{seed}'
                if (dest/'search_summary.json').exists():
                    continue
                cfg=json.loads((old/'resolved_config.json').read_text())
                cfg['compute'].update(max_gpu_seconds=20.,max_seconds=50.)
                if not (dest/'evaluations.json').exists():
                    import_started=time.perf_counter()
                    dest.mkdir(parents=True,exist_ok=True)
                    shutil.copytree(old/'candidates',dest/'candidates',dirs_exist_ok=True)
                    meta=json.loads((Path(cfg['data']['processed'])/'metadata.json').read_text())
                    identity=digest({'config':cfg,'data':meta['fingerprint']})
                    events=json.loads((old/'evaluations.json').read_text())
                    for event in events:
                        event['original_identity']=event['identity']
                        event['identity']=identity
                        event['imported_from']=str(old)
                    save_json(dest/'evaluations.json',events)
                    save_json(dest/'search_identity.json',{'identity':identity})
                    oldwall=json.loads((old/'wall_state.json').read_text())
                    import_seconds=time.perf_counter()-import_started
                    save_json(dest/'wall_state.json',{'seconds':oldwall['seconds']+import_seconds})
                    save_json(dest/'import_provenance.json',{'source':str(old),'source_ledger_sha256':hashlib.sha256((old/'evaluations.json').read_bytes()).hexdigest(),
                              'original_epochs':sum(e['charged_epochs'] for e in events),'original_gpu_seconds':sum(e['gpu_seconds'] for e in events),
                              'original_wall_seconds':oldwall['seconds'],'import_wall_seconds':import_seconds,'changes':{'max_gpu_seconds':[8.,20.],'max_seconds':[20.,50.]}})
                save_json(dest/'resolved_config.json',cfg)
                clear_caches()
                result=run_search(cfg,dest)
                print(phase,variant,seed,result['target_reached'],round(result['gpu_seconds'],3),result['best']['fitness'] if result['best'] else None,flush=True)
    print('Charged continuation complete.',flush=True)


if __name__=='__main__':
    main()
