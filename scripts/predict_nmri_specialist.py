"""Apply the frozen hybrid/specialist pipeline to CSV or processed NPZ inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from src.data.preprocessing import transform_new
from src.models.controls import make_model
from src.training.engine import predict
from src.evaluation.specialist import route_specialist


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA requested but unavailable')
    if args.output.exists():raise SystemExit('Use a fresh output file')
    cfg=json.loads(args.pipeline.read_text())
    for key in ['checkpoint','specialist','preprocessor']:
        assert hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest()==cfg[key+'_sha256'],key
    if args.input.suffix.lower()=='.csv':
        x=transform_new(cfg['preprocessor'],pd.read_csv(args.input))
    else:
        with np.load(args.input) as p:x=p['x']
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=cfg['features'] or not np.isfinite(x).all():raise ValueError('Invalid processed inputs')
    model=make_model(cfg['architecture'],cfg['features'],len(cfg['classes'])).to(args.device)
    checkpoint=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    probabilities=predict(model,x,torch.device(args.device))
    score=joblib.load(cfg['specialist']).predict_proba(x)[:,1]
    predictions=route_specialist(probabilities.argmax(1),score,cfg['threshold'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,prediction=predictions,base_probabilities=probabilities,
                        specialist_score=score,classes=np.asarray(cfg['classes']))
    print(f'Saved {len(predictions)} predictions; hybrid device={args.device}; specialist device=CPU.')


if __name__=='__main__':main()
