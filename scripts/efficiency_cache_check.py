"""Measure real continuation setup work, separate from detector evidence."""
import json
from pathlib import Path
import time
from unittest.mock import patch
import torch
from src.training.engine import train_to
from src.training import engine
from src.training.runtime import clear_caches
from src.utils.core import save_json

ROOT=Path('results/ua_amf_efficiency_20261108')


def main():
    if (ROOT/'continuation_cache_check.json').exists():
        print('Using saved continuation mechanism check; do not overwrite timing records.')
        return
    cfg=json.loads(Path('results/ua_amf_prediction_20261108/resolved_config.json').read_text())
    arch=json.loads(Path('results/ua_amf_prediction_20261108/cohort_definitions.json').read_text())[0]['architecture']
    runtime=json.loads((ROOT/'benchmark_protocol.json').read_text())['runtime']
    records=[]
    for name,settings in [('legacy',None),('optimized',runtime)]:
        clear_caches()
        folder=ROOT/'cache_check'/name
        began=time.perf_counter()
        with patch('src.training.engine.make_model',wraps=engine.make_model) as construction,patch('torch.load',wraps=torch.load) as loading:
            for budget in [3,10,30]:
                row=train_to(arch,cfg['data']['processed'],folder,budget,983,cfg['training'],runtime=settings,deadline=began+30)
            records.append({'profile':name,'model_constructions':construction.call_count,'checkpoint_loads':loading.call_count,
                            'wall_seconds':time.perf_counter()-began,'gpu_seconds':row['gpu_seconds'],'macro_f1':row['macro_f1']})
    a=torch.load(ROOT/'cache_check/legacy/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(ROOT/'cache_check/optimized/last.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(a['model'][key],b['model'][key]) for key in a['model'])
    save_json(ROOT/'continuation_cache_check.json',{'records':records,'exact_weights':True,'scope':'One development architecture and seed; engineering mechanism check, not IDS accuracy evidence.'})
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
