"""Freeze specialist and validation thresholds before GPU-backed test inference."""
import argparse
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import torch
from src.models.controls import make_model
from src.training.engine import load_partition,predict
from src.evaluation.metrics import classification_metrics
from src.evaluation.specialist import calibrate_specialist,route_specialist
from src.utils.core import read_config,save_json,environment


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise SystemExit('Use a fresh directory')
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    args.output.mkdir(parents=True)
    study=Path('results/nmri_tradeoff/20261108')
    trials=json.loads((study/'results.json').read_text())
    feasible=[r for r in trials if r.get('metrics') and r['metrics']['attack_detection']['fpr']<=.03 and r['metrics']['per_class']['1']['recall']>=.10 and r['metrics']['attack_detection']['fnr']<=.10]
    selected=max(feasible,key=lambda r:r['metrics']['macro_f1'])
    assert selected['method']=='hist_gradient_boosting'
    specialist=joblib.load(study/'hist_gradient_boosting.joblib')
    shutil.copyfile(study/'hist_gradient_boosting.joblib',args.output/'specialist.joblib')
    source=Path('results/false_alarm_optimization/final_20261108')
    data=Path('results/submission_final_evaluation/20261108/processed')
    vx,vy=load_partition(data,'val');score=specialist.predict_proba(vx)[:,1]
    choices={}
    for seed in [10042,10043,10044]:
        with np.load(source/f'seed_{seed}/validation_epoch_200.npz') as p:
            assert np.array_equal(vy,p['y']);base=p['probabilities'].argmax(1)
        choices[str(seed)]=calibrate_specialist(base,score,vy,.03)
        choices[str(seed)]['validation_metrics']=classification_metrics(vy,route_specialist(base,score,choices[str(seed)]['threshold']),[str(i) for i in range(8)],'0')
    save_json(args.output/'protocol.json',{'selected':selected,'thresholds':choices,'seeds':[10042,10043,10044],
        'selection':'frozen method from validation seed42; per-model thresholds calibrated on validation before all new test predictions',
        'revision':'Initial 50% NMRI recall at <=5% FPR was not reached. Exploratory follow-up chooses >=10% NMRI recall at <=3% FPR, >=90% overall attack detection, then macro-F1.',
        'scope':'same fixed training-fitted specialist across three existing hybrid seeds; not three independent specialist fits; development-known split'})
    save_json(args.output/'environment.json',environment())
    # Test partition is opened only after every calibration choice is persisted.
    x,y=load_partition(data,'test');test_score=specialist.predict_proba(x)[:,1]
    cfg=read_config('configs/gas_low_false_alarm.yaml');rows=[]
    for seed in [10042,10043,10044]:
        model=make_model(cfg['architecture'],x.shape[1],8).cuda()
        checkpoint=torch.load(source/f'seed_{seed}/last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model']);prob=predict(model,x,torch.device('cuda'))
        base=prob.argmax(1);prediction=route_specialist(base,test_score,choices[str(seed)]['threshold'])
        m=classification_metrics(y,prediction,[str(i) for i in range(8)],'0')
        np.savez_compressed(args.output/f'predictions_{seed}.npz',y=y,prediction=prediction,base_probabilities=prob,specialist_score=test_score)
        rows.append({'seed':seed,'metrics':m,'hybrid_inference_device':'cuda'})
        print(seed,'FPR',m['attack_detection']['fpr'],'NMRI recall',m['per_class']['1']['recall'],'accuracy',m['accuracy'],'F1',m['macro_f1'],flush=True)
    save_json(args.output/'measurements.json',rows)
    def stats(values):return {'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
    summary={'n':3,'metrics':{k:stats([r['metrics'][k] for r in rows]) for k in ['accuracy','macro_f1']},
        'fpr':stats([r['metrics']['attack_detection']['fpr'] for r in rows]),
        'attack_recall':stats([1-r['metrics']['attack_detection']['fnr'] for r in rows]),
        'nmri_recall':stats([r['metrics']['per_class']['1']['recall'] for r in rows]),'complete':True}
    save_json(args.output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
