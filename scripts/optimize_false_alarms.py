"""Validation-selected loss comparison with an explicit false-alarm constraint."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from src.training.engine import train_to, final_evaluate
from src.evaluation.metrics import classification_metrics
from src.utils.core import read_config, save_json, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    if args.output.exists():
        raise SystemExit('Use a fresh output directory')
    root = args.output
    root.mkdir(parents=True)
    source = Path('results/submission_final_evaluation/20261108/processed')
    cfg = read_config('configs/submission_evaluation.yaml')
    protocol = {'config': cfg, 'candidates': ['none', 'sqrt_balanced'],
                'validation_seed': 42, 'epochs': 200, 'maximum_validation_fpr': .01,
                'minimum_validation_attack_recall': .90,
                'selection': 'highest validation macro-F1 among feasible candidates; no final test if none feasible',
                'final_seeds': [10042, 10043, 10044],
                'limitation': 'exploratory correction motivated by known historical test errors; no blind generalization claim'}
    save_json(root/'protocol.json', protocol)
    save_json(root/'environment.json', environment())
    data = root/'validation_data'
    data.mkdir()
    for name in ['train.npz', 'val.npz', 'metadata.json']:
        shutil.copyfile(source/name, data/name)
    assert not (data/'test.npz').exists()
    classes = json.loads((data/'metadata.json').read_text())['classes']
    results = []
    for weighting in protocol['candidates']:
        training = dict(cfg['training'], class_weighting=weighting, device='cuda')
        run = root/'validation'/weighting
        train_to(cfg['architecture'], data, run, 200, 42, training)
        with np.load(run/'validation_epoch_200.npz') as p:
            m = classification_metrics(p['y'], p['probabilities'].argmax(1), classes, '0')
        feasible = m['attack_detection']['fpr'] <= .01 and m['attack_detection']['fnr'] <= .10
        results.append({'weighting': weighting, 'metrics': m, 'feasible': feasible})
        save_json(root/'validation_results.json', results)
        print(weighting, m['accuracy'], m['macro_f1'], m['attack_detection'], 'feasible:', feasible, flush=True)
    feasible = [r for r in results if r['feasible']]
    if not feasible:
        save_json(root/'selection.json', {'status': 'no_feasible_candidate', 'test_evaluations': 0})
        print('No feasible candidate; no test evaluation performed.', flush=True)
        return
    selected = max(feasible, key=lambda r: r['metrics']['macro_f1'])
    save_json(root/'selection.json', {'status': 'selected_before_testing', **selected})
    training = dict(cfg['training'], class_weighting=selected['weighting'], device='cuda')
    measurements = []
    for seed in protocol['final_seeds']:
        run = root/'final'/f'seed_{seed}'
        metrics = final_evaluate(cfg['architecture'], source, run, 200, seed, training, '0')
        with np.load(run/'test_predictions.npz') as p:
            verified = classification_metrics(p['y'], p['probabilities'].argmax(1), classes, '0')
        assert all(verified[k] == metrics[k] for k in verified)
        measurements.append({'seed': seed, 'metrics': metrics})
        save_json(root/'final_measurements.json', measurements)
        print('final', seed, metrics['accuracy'], metrics['macro_f1'], metrics['attack_detection'], flush=True)
    summary = {'selected_weighting': selected['weighting'], 'runs': 3, 'epochs': 200,
               'metrics': {k: {'mean': float(np.mean([r['metrics'][k] for r in measurements])),
                               'sd': float(np.std([r['metrics'][k] for r in measurements], ddof=1))}
                           for k in ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']},
               'attack_detection': {k: {'mean': float(np.mean([r['metrics']['attack_detection'][k] for r in measurements])),
                                         'sd': float(np.std([r['metrics']['attack_detection'][k] for r in measurements], ddof=1))}
                                   for k in ['fpr', 'fnr']}, 'complete': True}
    save_json(root/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
