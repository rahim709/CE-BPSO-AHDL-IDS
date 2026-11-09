"""Audit saved smoke validation predictions and recompute tie-aware rankings."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from src.evaluation.metrics import classification_metrics, majority_baseline
from src.evaluation.ranking import ranking_metrics
from src.utils.core import digest, save_json


def main() -> None:
    root = Path(json.loads(Path('results/synthetic_datasets/latest.json').read_text())['directory'])
    audit, rank_audit = [], []
    for name in ['ton_iot', 'bot_iot', 'swat', 'gas_pipeline_real']:
        source = Path('results/gas_real_smoke') if name == 'gas_pipeline_real' else root / name
        run = source / 'fixed_model' if name == 'gas_pipeline_real' else source / 'train'
        cfg = json.loads((run / 'resolved_config.json').read_text())
        data = Path(cfg['data']['processed'])
        meta = json.loads((data / 'metadata.json').read_text())
        epoch = json.loads((run / 'trajectory.json').read_text())[-1]['epoch']
        with np.load(run / f'validation_epoch_{epoch}.npz') as predictions:
            metrics = classification_metrics(predictions['y'], predictions['probabilities'].argmax(1), meta['classes'])
            with np.load(data / 'train.npz') as training:
                baseline = majority_baseline(training['y'], predictions['y'], meta['classes'])
        audit.append({'dataset': name, 'synthetic_data': bool(meta['config'].get('synthetic', False)),
                      'validation_epoch': epoch, 'validation_macro_f1': metrics['macro_f1'],
                      'prediction_counts': metrics['prediction_counts'],
                      'collapsed_to_single_class': metrics['collapsed_to_single_class'],
                      'majority_baseline': baseline, 'zero_recall_classes': metrics['zero_recall_classes'], 'source': str(run)})
        diag = source / 'diagnostics'
        cohort = json.loads((diag / 'cohort.json').read_text())
        events = json.loads((diag / 'evaluations.json').read_text())
        trajectories = []
        for candidate in cohort:
            event = next(e for e in events if digest(e['architecture']) == digest(candidate['architecture']))
            rows = json.loads((diag / 'candidates' / event['candidate'] / 'trajectory.json').read_text())
            trajectories.append({r['epoch']: r['macro_f1'] for r in rows})
        for early, late in [(1, 2), (2, 3)]:
            rank_audit.append({'dataset': name, 'synthetic_data': bool(meta['config'].get('synthetic', False)),
                               'early_epoch': early, 'late_epoch': late,
                               'scope': 'Complete two-candidate smoke cohort; insufficient for research inference',
                               **ranking_metrics(np.array([t[early] for t in trajectories]),
                                                 np.array([t[late] for t in trajectories]), k=1)})
    save_json('results/result_review/initial_validation_health.json', audit)
    save_json('results/result_review/corrected_smoke_rankings.json', rank_audit)
    print(f'Audited{len(audit)} validation runs and{len(rank_audit)} ranking comparisons; no retraining.')


if __name__ == '__main__':
    main()
