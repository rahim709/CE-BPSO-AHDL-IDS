"""Evaluate the validation-selected loss correction under a recorded revised target."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.training.engine import final_evaluate
from src.evaluation.metrics import classification_metrics
from src.utils.core import read_config, save_json, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    if args.output.exists():
        raise SystemExit('Use a fresh output')
    source = Path('results/false_alarm_optimization/20261108')
    candidates = json.loads((source/'validation_results.json').read_text())
    feasible = [r for r in candidates if r['metrics']['attack_detection']['fpr'] <= .03
                and r['metrics']['attack_detection']['fnr'] <= .10]
    if not feasible:
        raise SystemExit('No candidate meets the revised validation target')
    selected = max(feasible, key=lambda r: r['metrics']['macro_f1'])
    cfg = read_config('configs/gas_low_false_alarm.yaml')
    assert cfg['training']['class_weighting'] == selected['weighting']
    cfg['training']['device'] = 'cuda'
    cfg['data']['processed'] = 'results/submission_final_evaluation/20261108/processed'
    args.output.mkdir(parents=True)
    save_json(args.output/'protocol.json', {'config': cfg, 'selected_on_validation': selected,
        'maximum_validation_fpr': .03, 'minimum_validation_attack_recall': .90,
        'revision_reason': 'Neither candidate met the initial 1% target; revised exploratory 3% target before new test scores',
        'selection_source': str(source), 'endpoint': 200, 'seeds': [10042,10043,10044],
        'limitation': 'development-known historical split; exploratory correction, not blind confirmation'})
    save_json(args.output/'environment.json', environment())
    classes = json.loads((Path(cfg['data']['processed'])/'metadata.json').read_text())['classes']
    rows=[]
    for seed in [10042,10043,10044]:
        run=args.output/f'seed_{seed}'
        save_json(run/'resolved_config.json', dict(cfg, final_seed=seed))
        m=final_evaluate(cfg['architecture'],cfg['data']['processed'],run,200,seed,cfg['training'],'0')
        with np.load(run/'test_predictions.npz') as p:
            verified=classification_metrics(p['y'],p['probabilities'].argmax(1),classes,'0')
        assert all(verified[k]==m[k] for k in verified)
        rows.append({'seed':seed,'metrics':m})
        save_json(args.output/'measurements.json',rows)
        print(seed,m['accuracy'],m['macro_f1'],m['attack_detection'],flush=True)
    def stats(values):
        return {'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
    summary={'complete':True,'selected_weighting':selected['weighting'],'n':3,'epochs':200,
        'metrics':{k:stats([r['metrics'][k] for r in rows]) for k in ['accuracy','macro_f1','macro_precision','macro_recall']},
        'attack_detection':{k:stats([r['metrics']['attack_detection'][k] for r in rows]) for k in ['fpr','fnr']}}
    save_json(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
