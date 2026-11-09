"""Training-only specialist fits and validation-only NMRI/FPR trade-off audit."""
import argparse
import json
from pathlib import Path
import hashlib
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import classification_metrics
from src.training.engine import load_partition
from src.utils.core import save_json, seed_all, environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output
    if root.exists(): raise SystemExit('Use a fresh output directory')
    if not torch.cuda.is_available(): raise SystemExit('CUDA required for neural specialist')
    root.mkdir(parents=True)
    protocol={'seed':42,'neural_epochs':100,'fpr_caps':[.01,.03,.05,.10,.20],
        'success_criteria':{'maximum_fpr':.05,'minimum_nmri_recall':.50,'minimum_attack_detection':.90},
        'methods':['existing_hybrid_score','extra_trees','hist_gradient_boosting','neural_specialist'],
        'routing':'only replace normal/NMRI predictions; preserve six other predicted classes',
        'selection_data':'validation only; test arrays and predictions never opened',
        'scope':'exploratory development-known split; no confirmatory generalization claim'}
    save_json(root/'protocol.json',protocol)
    save_json(root/'environment.json',environment())
    data=Path('results/submission_final_evaluation/20261108/processed')
    x,y=load_partition(data,'train');vx,vy=load_partition(data,'val')
    mask=np.isin(y,[0,1]);bx,by=x[mask],y[mask]
    with np.load('results/false_alarm_optimization/20261108/validation/sqrt_balanced/validation_epoch_200.npz') as p:
        assert np.array_equal(vy,p['y']); probabilities=p['probabilities']
    base=probabilities.argmax(1);eligible=np.isin(base,[0,1]); normal=vy==0
    models={'extra_trees':ExtraTreesClassifier(n_estimators=200,min_samples_leaf=2,max_features=1.,class_weight='balanced',random_state=42,n_jobs=2),
            'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=31,l2_regularization=1,class_weight='balanced',random_state=42)}
    scores={'existing_hybrid_score':probabilities[:,1]/np.maximum(probabilities[:,0]+probabilities[:,1],1e-12)}
    for name,model in models.items():
        model.fit(bx,by);scores[name]=model.predict_proba(vx)[:,1]
        joblib.dump(model,root/f'{name}.joblib')
        print('Fitted',name,flush=True)
    seed_all(42,1)
    scaler=StandardScaler().fit(bx)
    tx=torch.tensor(scaler.transform(bx),dtype=torch.float32,device='cuda')
    ty=torch.tensor(by,dtype=torch.long,device='cuda')
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2)).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    counts=np.bincount(by,minlength=2)
    weights=torch.tensor(len(by)/(2*counts),dtype=torch.float32,device='cuda')
    for epoch in range(1,101):
        model.train();order=torch.randperm(len(by),device='cuda')
        for indices in order.split(512):
            opt.zero_grad();loss=nn.functional.cross_entropy(model(tx[indices]),ty[indices],weight=weights)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    model.eval()
    with torch.no_grad():
        scores['neural_specialist']=model(torch.tensor(scaler.transform(vx),dtype=torch.float32,device='cuda')).softmax(1)[:,1].cpu().numpy()
    torch.save(model.state_dict(),root/'neural_specialist.pt');joblib.dump(scaler,root/'neural_scaler.joblib')
    results=[]
    for name,score in scores.items():
        np.savez_compressed(root/f'{name}_validation.npz',y=vy,score=score,base_prediction=base)
        normal_scores=np.sort(score[normal & eligible])[::-1]
        fixed_fp=int((normal & ~eligible).sum())
        for cap in protocol['fpr_caps']:
            allowance=int(np.floor(cap*normal.sum()))-fixed_fp
            if allowance<0:
                results.append({'method':name,'cap':cap,'status':'fixed_other_class_false_alarms_exceed_cap'});continue
            threshold=float(normal_scores[min(allowance,len(normal_scores)-1)])
            pred=base.copy();pred[eligible]=np.where(score[eligible]>threshold,1,0)
            m=classification_metrics(vy,pred,[str(i) for i in range(8)],'0')
            feasible=m['attack_detection']['fpr']<=.05 and m['per_class']['1']['recall']>=.5 and m['attack_detection']['fnr']<=.1
            results.append({'method':name,'cap':cap,'threshold':threshold,'metrics':m,'feasible':feasible})
            print(name,cap,'FPR',round(m['attack_detection']['fpr'],4),'NMRI recall',round(m['per_class']['1']['recall'],4),'F1',round(m['macro_f1'],4),flush=True)
    save_json(root/'results.json',results)
    feasible=[r for r in results if r.get('feasible')]
    save_json(root/'selection.json',{'status':'feasible' if feasible else 'no_feasible_candidate',
        'selected':max(feasible,key=lambda r:r['metrics']['macro_f1']) if feasible else None,'test_access':False})
    # Training/validation-only ambiguity analysis; excluding duration is not a full-feature impossibility proof.
    import pandas as pd
    train_hash=pd.util.hash_pandas_object(pd.DataFrame(bx[:,:-1]),index=False).to_numpy()
    support=pd.crosstab(train_hash,by).reindex(columns=[0,1],fill_value=0)
    shared=(support[0]>0)&(support[1]>0)
    save_json(root/'training_ambiguity.json',{'normal_training_rows':int((by==0).sum()),'nmri_training_rows':int((by==1).sum()),
        'nmri_rows_in_non_time_states_also_seen_as_normal':int(support.loc[shared,1].sum()),
        'normal_rows_in_these_states':int(support.loc[shared,0].sum()),
        'note':'Hashes of processed 25 non-time predictors; duration excluded. Evidence of ambiguity, not proof of full-feature indistinguishability.'})
    print('Completed validation-only study; test data were not opened.',flush=True)


if __name__=='__main__':main()
