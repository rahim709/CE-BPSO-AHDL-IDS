"""Validation-only sensitivity study with identical non-time states kept together.

This grouping removes exact predictor-state overlap; it is not a capture/session
split and does not establish prospective temporal or cross-release generalization.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.data.preprocessing import load_files
from src.evaluation.metrics import classification_metrics
from src.training.engine import train_to
from src.utils.core import digest, environment, read_config, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/state_grouped_validation'))
    args = parser.parse_args()
    cfg = read_config('configs/research_extension.yaml')
    protocol = {'split_seed': 1729, 'seeds': [42,43,44], 'epochs': 30,
                'grouping': 'pandas hashes of all 25 non-time predictors, excluding labels',
                'input_features': 'all 26 including query-response duration', 'normalization': 'training-fitted Min-Max',
                'methods': ['random_forest','logistic_regression','reference_hybrid'], 'test_access': False,
                'scope': 'state-grouped sensitivity; no capture identities or temporal generalization'}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise SystemExit('Protocol mismatch; choose a fresh output directory')
    save_json(path, protocol)
    save_json(args.output / 'environment.json', environment())
    raw, sources = load_files(cfg['data']['files'])
    features = cfg['data']['features']
    group_features = [f for f in features if f!='time']
    groups = pd.util.hash_pandas_object(raw[group_features], index=False).to_numpy()
    pool, reserved = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=1729).split(raw,groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=1730).split(raw.iloc[pool],groups=groups[pool]))
    tr, va = pool[tr], pool[va]
    assert not set(groups[tr]) & set(groups[va])
    assert not set(groups[tr]) & set(groups[reserved])
    assert not set(groups[va]) & set(groups[reserved])
    classes = sorted(raw.iloc[tr]['result'].astype(str).unique())
    assert classes == [str(i) for i in range(8)], 'Training class coverage failed; no silent split-seed search.'
    y, vy = [np.array([classes.index(str(v)) for v in raw.iloc[indices]['result']],dtype=np.int64) for indices in [tr,va]]
    imputer = SimpleImputer(strategy='median',keep_empty_features=True).fit(raw.iloc[tr][features])
    unscaled = [imputer.transform(raw.iloc[indices][features]) for indices in [tr,va]]
    scaler = MinMaxScaler().fit(unscaled[0])
    x, vx = [scaler.transform(a).astype(np.float32) for a in unscaled]
    data = args.output / 'data'; data.mkdir(exist_ok=True)
    meta = {'classes':classes,'sources':sources,'features':features,'normalization':'minmax',
            'config':dict(cfg['data'],split='non_time_state_grouped',normalization='minmax'),
            'splits':{name:{'rows':len(indices),'classes':raw.iloc[indices]['result'].astype(str).value_counts().to_dict()}
                      for name,indices in [('train',tr),('val',va)]}}
    meta['fingerprint']=digest(meta)
    save_json(data/'metadata.json',meta)
    save_json(data/'split_indices.json',{'train':raw.index[tr].tolist(),'val':raw.index[va].tolist(),'reserved':raw.index[reserved].tolist()})
    save_json(args.output/'split_audit.json',{'train_rows':len(tr),'validation_rows':len(va),'reserved_rows':len(reserved),
             'unique_groups':len(set(groups)), 'train_validation_group_overlap':0,'train_reserved_group_overlap':0,
             'validation_reserved_group_overlap':0,'test_array_available':False})
    np.savez_compressed(data/'train.npz',x=x,y=y)
    np.savez_compressed(data/'val.npz',x=vx,y=vy)
    assert not (data/'test.npz').exists()
    rows=[]
    for seed in protocol['seeds']:
        for method in protocol['methods']:
            began=time.perf_counter()
            if method=='reference_hybrid':
                training=dict(cfg['training'],device='cpu',prediction_epochs=[3,10,30])
                run=args.output/'runs'/method/f'seed_{seed}'
                row=train_to(cfg['architecture'],data,run,30,seed,training)
                with np.load(run/'validation_epoch_30.npz') as p:
                    pred=p['probabilities'].argmax(1)
                fit_seconds=row['elapsed_seconds']
            else:
                model=(RandomForestClassifier(n_estimators=100,class_weight='balanced',n_jobs=1,random_state=seed)
                       if method=='random_forest' else make_pipeline(StandardScaler(),
                       LogisticRegression(class_weight='balanced',max_iter=1000,random_state=seed)))
                model.fit(x,y)
                fit_seconds=time.perf_counter()-began
                pred=model.predict(vx)
            metric=classification_metrics(vy,pred,classes,'0')
            dest=args.output/f'{method}_seed_{seed}_validation.npz'
            np.savez_compressed(dest,y=vy,prediction=pred)
            rows.append({'method':method,'seed':seed,'metrics':metric,'fit_or_train_validation_seconds':fit_seconds,
                         'wall_seconds':time.perf_counter()-began,'prediction_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
            save_json(args.output/'measurements.json',rows)
            print(f'{method}/seed{seed}: validation macro-F1={metric["macro_f1"]:.6f}',flush=True)
    save_json(args.output/'complete.json',{'completed':True,'runs':len(rows),'test_access':False})


if __name__=='__main__':
    main()
