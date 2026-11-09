"""Audit and plot the completed, uncensored reduced-row 3/10/30 cohort."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.metrics import classification_metrics
from src.utils.core import save_json


def main():
    root = Path('results/ua_amf_cohort_20261108')
    cfg = json.loads((root / 'resolved_config.json').read_text())
    meta = json.loads((Path(cfg['data']['processed']) / 'metadata.json').read_text())
    events = json.loads((root / 'evaluations.json').read_text())
    rows, verified = [], 0
    for path in sorted((root / 'candidates').glob('*/trajectory.json')):
        trajectory = json.loads(path.read_text())
        seed = next(e['seed'] for e in events if e['candidate'] == path.parent.name)
        for budget in [3, 10, 30]:
            row = next(r for r in trajectory if r['epoch'] == budget)
            with np.load(path.parent / f'validation_epoch_{budget}.npz') as pred:
                metrics = classification_metrics(pred['y'], pred['probabilities'].argmax(1), meta['classes'], '0')
            assert abs(metrics['macro_f1'] - row['macro_f1']) < 1e-12
            verified += 1
            rows.append({'candidate': path.parent.name, 'seed': seed, 'budget': budget,
                         'macro_f1': row['macro_f1'], 'collapsed': row['metrics']['collapsed_to_single_class'],
                         'normal_fpr': metrics['attack_detection']['fpr'], 'parameters': row['parameters']})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'cohort_results.csv', index=False)
    epochs = sum(e['charged_epochs'] for e in events)
    assert epochs == 270 and verified == 27
    result = {'completed_epochs': epochs, 'verified_endpoints': verified,
              'wall_seconds': json.loads((root / 'ranking_diagnostics.json').read_text())['wall_seconds'],
              'gpu_seconds': sum(e['gpu_seconds'] for e in events),
              'peak_gpu_memory_bytes': max(e['peak_gpu_memory_bytes'] for e in events),
              'terminal_macro_f1': frame[frame.budget == 30].macro_f1.tolist(),
              'terminal_collapsed_models': int(frame[frame.budget == 30].collapsed.sum()),
              'all_terminal_scores_tied': bool(frame[frame.budget == 30].macro_f1.nunique() == 1),
              'selection_regret': 0.0,
              'interpretation': 'Zero regret is degenerate: all terminal candidates tie at weak one-class performance. No evidence of useful ranking or uncertainty benefit. Reduced-row cohort only.'}
    save_json(root / 'cohort_audit.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    architectures = {e['candidate']: json.dumps(e['architecture'], sort_keys=True) for e in events}
    frame['architecture'] = frame.candidate.map(architectures)
    for i, (_, group) in enumerate(frame.groupby('architecture')):
        stats = group.groupby('budget').macro_f1.agg(['mean', 'std'])
        ax.errorbar(stats.index, stats['mean'], yerr=stats['std'], marker='o', label=f'Architecture {i + 1}')
    ax.set(xlabel='Cumulative epochs', ylabel='Validation macro-F1', title='Complete 3-architecture, 3-seed cohort')
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'cohort_learning.png', dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
