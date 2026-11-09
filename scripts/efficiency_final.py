"""Freeze every fixed-compute method before dedicated historical-test evaluation."""
import json
from pathlib import Path
import time
import numpy as np
from src.training.engine import train_to,final_evaluate
from src.training.runtime import clear_caches
from src.utils.core import digest,save_json

ROOT=Path('results/ua_amf_efficiency_20261108')
SOURCE=Path('results/ua_amf_prediction_20261108')


def main():
    followup=ROOT/'followup20'
    table=__import__('pandas').read_csv(followup/'comparison.csv')
    compute=table[table.experiment=='compute']
    # Evaluate all optimized variants and controls, avoiding post-hoc selection of
    # only favorable methods. One dedicated seed per search-seed selection.
    selections=[]
    for method in sorted(compute.variant):
        for seed in [601,602,603]:
            path=followup/'compute'/method/f'seed_{seed}/strict_winner.json'
            selections.append({'variant':method,'search_seed':seed,'architecture':json.loads(path.read_text())['architecture'] if path.exists() else None})
    protocol={'selections':selections,'training_seed':1601,'epochs':30,'max_new_stage_wall_seconds':180,
              'data':'Same 8192-row training subset, historical development-known test; preprocessing compatibility established in prior final stage.',
              'test_status':'Not blind or independent. No test tuning.','choice_rule':'Best observed30-epoch validation candidate within20 measured GPU seconds.'}
    out=ROOT/'final'
    out.mkdir(exist_ok=True)
    if (out/'frozen_protocol.json').exists() and json.loads((out/'frozen_protocol.json').read_text()) != protocol:
        raise ValueError('Final choices changed; preserve old results and use a fresh protocol directory')
    save_json(out/'frozen_protocol.json',protocol)
    data=Path('results/ua_amf_prediction_final_20261108/data')
    config=json.loads((SOURCE/'resolved_config.json').read_text())
    previous_wall=json.loads((out/'stage_time.json').read_text())['wall_seconds'] if (out/'stage_time.json').exists() else 0.
    started=time.perf_counter()
    deadline=started+max(0.,180-previous_wall)
    results=[]
    try:
        for row in selections:
            if row['architecture'] is None:
                results.append(dict(row,status='no_common_fidelity_winner'))
                continue
            key=digest({'architecture':row['architecture'],'seed':1601})
            folder=out/'models'/key
            if not (folder/'test_metrics.json').exists():
                if time.perf_counter()>=deadline:raise TimeoutError('Final cap reached')
                clear_caches()
                # Final models use the existing dense-validation FP32 protocol,
                # ensuring every method has the same final training procedure.
                train_to(row['architecture'],data,folder,30,1601,config['training'],deadline=deadline)
                metrics=final_evaluate(row['architecture'],data,folder,30,1601,config['training'],normal_label='0')
            else:
                metrics=json.loads((folder/'test_metrics.json').read_text())
            results.append(dict(row,status='completed',model_directory=str(folder),**metrics))
    except TimeoutError as exc:
        save_json(out/'stop.json',{'partial':True,'reason':str(exc)})
    finally:
        save_json(out/'results.json',results)
        save_json(out/'stage_time.json',{'wall_seconds':previous_wall+time.perf_counter()-started})
    print('Final evaluations',len(results),flush=True)


if __name__=='__main__':main()
