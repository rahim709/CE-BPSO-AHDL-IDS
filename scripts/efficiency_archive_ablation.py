"""Small-archive CPU refits using actual historical trajectories; no new GPU labels."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from src.optimization.curve_predictor import CurvePredictor,curve_features
from src.utils.core import digest,save_json

ROOT=Path('results/ua_amf_efficiency_20261108')
SOURCE=Path('results/ua_amf_prediction_20261108')


def main():
    if (ROOT/'archive_size_ablation.json').exists():
        print('Using saved archive-size ablation.')
        return
    definitions={r['index']:r for r in json.loads((SOURCE/'cohort_definitions.json').read_text())}
    endpoints=pd.read_csv(SOURCE/'reliability_endpoints.csv')
    events=json.loads((SOURCE/'cohort/evaluations.json').read_text())
    output=[]
    for fit_ids,cal_ids in [(list(range(12)),list(range(12,18))), (list(range(8)),list(range(12,16))), (list(range(4)),list(range(12,16)))]:
        start=time.perf_counter()
        records=[]
        for (index,seed),group in endpoints.groupby(['index','seed']):
            if 30 not in group.budget.values or index not in fit_ids+cal_ids+list(range(18,25)):
                continue
            trajectory=json.loads((Path(group.iloc[0].candidate_path)/'trajectory.json').read_text())
            sparse=[dict(r,macro_f1=r['macro_f1'] if r['epoch'] in [3,10,30] else None) for r in trajectory]
            target=next(r['macro_f1'] for r in trajectory if r['epoch']==30)
            for b in [3,10]:
                features=curve_features(definitions[index]['architecture'],sparse,b)
                features[-1]=0.
                records.append({'architecture_id':definitions[index]['architecture_id'],'budget':b,'features':features,
                                'partition':'fit' if index in fit_ids else 'calibration' if index in cal_ids else 'test','target':target})
        model=CurvePredictor().fit(records,'archive-ablation')
        charged={definitions[i]['architecture_id'] for i in fit_ids+cal_ids}
        cost=sum(e['gpu_seconds'] for e in events if digest(e['architecture']) in charged)
        for b in [3,10]:
            held=[r for r in records if r['partition']=='test' and r['budget']==b]
            m,u,l,h=model.predict_features([r['features'] for r in held],b)
            y=np.array([r['target'] for r in held])
            output.append({'fit_architectures':len(fit_ids),'calibration_architectures':len(cal_ids),'budget':b,
                           'heldout_architectures':7,'coverage':float(np.mean((l<=y)&(y<=h))),
                           'mean_width':float(np.mean(h-l)),'mae':float(np.mean(abs(m-y))),
                           'new_gpu_seconds':0.,'historical_fit_cal_gpu_seconds':cost,'cpu_build_seconds':time.perf_counter()-start,
                           'used_in_benchmark':len(fit_ids)==12})
    pd.DataFrame(output).to_csv(ROOT/'archive_size_ablation.csv',index=False)
    save_json(ROOT/'archive_size_ablation.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
