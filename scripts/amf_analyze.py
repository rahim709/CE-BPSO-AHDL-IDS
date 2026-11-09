"""Re-score saved outputs, audit budgets, and build measured anytime comparisons."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.metrics import classification_metrics
from src.utils.core import digest, save_json


def json_rows(frame):
    return frame.astype(object).where(pd.notnull(frame), None).to_dict('records')


def verify_prediction(path, trajectory, classes):
    budget = int(path.stem.split('_')[-1])
    row = next(r for r in trajectory if r['epoch'] == budget)
    with np.load(path) as pred:
        metrics = classification_metrics(pred['y'], pred['probabilities'].argmax(1), classes, '0')
    if abs(metrics['macro_f1'] - row['macro_f1']) > 1e-12:
        raise ValueError(f'Saved prediction metric differs: {path}')
    return metrics


def analyze_cohort(study):
    endpoints = pd.read_csv(study / 'reliability_endpoints.csv')
    ranks = json.loads((study / 'reliability_summary.json').read_text())['rankings']
    pd.DataFrame([{k: r[k] for k in ['seed', 'early_budget', 'reference_budget', 'cohort_complete', 'n', 'spearman', 'kendall', 'top_k_overlap', 'selection_regret', 'selection_regret_range']} for r in ranks]).to_csv(study / 'reliability_rankings.csv', index=False)
    definitions = {r['index']: r for r in json.loads((study / 'cohort_definitions.json').read_text())}
    meta = json.loads((study / 'data/metadata.json').read_text())
    verified, late_bloomers = [], []
    for path in sorted((study / 'cohort/candidates').glob('*/trajectory.json')):
        trajectory = json.loads(path.read_text())
        for prediction in path.parent.glob('validation_epoch_*.npz'):
            verify_prediction(prediction, trajectory, meta['classes'])
            verified.append(str(prediction))
    for seed, group in endpoints.groupby('seed'):
        matrix = group.pivot(index='index', columns='budget', values='macro_f1')
        if 30 not in matrix:
            continue
        paired = matrix[[3, 10, 30]].dropna()
        top = set(paired[30].nlargest(5).index)
        early_ranks = paired[3].rank(ascending=False, method='min')
        for index in sorted(top):
            if early_ranks.loc[index] > max(1, len(paired) // 3):
                late_bloomers.append({'seed': int(seed), 'index': int(index), 'early_rank': float(early_ranks.loc[index]),
                                     'f1_3': paired.loc[index, 3], 'f1_10': paired.loc[index, 10], 'f1_30': paired.loc[index, 30],
                                     'cohort_complete': len(paired) == 25})
    pd.DataFrame(late_bloomers).to_csv(study / 'late_bloomers.csv', index=False)
    heldout = pd.read_csv(study / 'predictor_heldout_predictions.csv')
    selections = []
    for (seed, budget), group in heldout.groupby(['seed', 'budget']):
        group = group.copy()
        observed, costs = [], []
        for row in group.itertuples():
            item = endpoints[(endpoints['index'] == row.index) & (endpoints.seed == seed) & (endpoints.budget == budget)].iloc[0]
            trajectory = json.loads((Path(item.candidate_path) / 'trajectory.json').read_text())
            before = [r for r in trajectory if r['epoch'] <= budget]
            observed.append(item.macro_f1)
            costs.append(np.mean([r['training_seconds'] + r['validation_seconds'] for r in before]) * (30 - budget))
        policies = {'observed': np.asarray(observed), 'prediction': group.prediction.values,
                    'prediction_uncertainty': group.prediction.values + group.radius.values,
                    'full': group.prediction.values + group.radius.values - .05 * np.asarray(costs) / 60}
        for name, scores in policies.items():
            i = int(np.argmax(scores))
            selected = group.iloc[i]
            selections.append({'seed': int(seed), 'budget': int(budget), 'policy': name, 'selected_index': int(selected['index']),
                               'selected_final_f1': selected.target, 'regret': group.target.max() - selected.target,
                               'scope': 'held-out architectures only, offline policy replay; not a timed search'})
    pd.DataFrame(selections).to_csv(study / 'heldout_promotion_replay.csv', index=False)
    save_json(study / 'cohort_prediction_audit.json', {'verified_archives': len(verified), 'all_match': True, 'paths': verified})
    return len(verified)


def analyze_benchmark(study, root):
    protocol = json.loads((root / 'protocol.json').read_text())
    meta = json.loads((study / 'data/metadata.json').read_text())
    summary_rows, anytime, targets, metric_rows, promotion_rows, audit, survivor_rows, budget_checks = [], [], [], [], [], [], [], []
    for summary_path in sorted(root.glob('*/seed_*/search_summary.json')):
        folder = summary_path.parent
        s = json.loads(summary_path.read_text())
        events = json.loads((folder / 'evaluations.json').read_text())
        assert s['charged_epochs'] == sum(e['charged_epochs'] for e in events)
        assert s['charged_epochs'] <= s['compute_cap']['max_epochs']
        for event in events:
            if event['fitness'] is None:
                continue
            path = folder / 'candidates' / event['candidate']
            trajectory = json.loads((path / 'trajectory.json').read_text())
            prediction = path / f"validation_epoch_{event['completed_budget']}.npz"
            if prediction.exists():
                metrics = verify_prediction(prediction, trajectory, meta['classes'])
                if abs(metrics['macro_f1'] - event['fitness']) > 1e-12:
                    raise ValueError('Evaluation event fitness mismatch')
                audit.append(str(prediction))
        prior_charge = 0.
        for event in events:
            assert prior_charge < 30 + 1e-6, 'New training job started after GPU cap'
            assert event['completed_budget'] <= event['requested_budget'] <= 30
            assert event['search_wall_seconds'] - event['wall_seconds'] < 30 + .01, 'New job started after wall cap'
            assert event['gpu_seconds'] <= event['wall_seconds'] + .01
            prior_charge += event.get('gpu_budget_seconds', event['gpu_seconds'])
        budget_checks.append({'path': str(summary_path), 'all_job_starts_within_cap': True, 'charged_gpu_seconds': prior_charge, 'measured_wall_seconds': s['wall_seconds'],
                              'epoch_boundary_overrun_seconds': max(0., s['wall_seconds'] - 30)})
        variant = folder.parent.name
        terminal = sorted([e for e in events if e['failure'] is None and e['fitness'] is not None and e['completed_budget'] == 30],
                          key=lambda e: e['search_wall_seconds'])
        eligible = [e for e in terminal if e['search_wall_seconds'] <= 30]
        winner = max(eligible, key=lambda e: e['fitness']) if eligible else None
        points = []
        best = -1.
        for event in terminal:
            best = max(best, event['fitness'])
            points.append({'variant': variant, 'seed': s['seed'], 'seconds': event['search_wall_seconds'], 'best_f1': best})
        if points:
            save_json(folder / 'anytime_trace.json', points)
        for limit in protocol['anytime_limits']:
            within = [e for e in terminal if e['search_wall_seconds'] <= limit]
            anytime.append({'variant': variant, 'seed': s['seed'], 'limit_seconds': limit,
                            'best_f1': max((e['fitness'] for e in within), default=None),
                            'terminal_evaluations': len(within)})
        for target in protocol['f1_targets']:
            hit = next((e for e in eligible if e['fitness'] >= target), None)
            targets.append({'variant': variant, 'seed': s['seed'], 'target': target,
                            'seconds': hit['search_wall_seconds'] if hit else None, 'censored_at_seconds': 30, 'attained': hit is not None})
        adaptive = folder / 'adaptive_state.json'
        state = json.loads(adaptive.read_text()) if adaptive.exists() else {}
        promotion_path = folder / 'promotions.json'
        records = json.loads(promotion_path.read_text()) if promotion_path.exists() else []
        terminal_by_arch = {digest(e['architecture']): e for e in terminal}
        for record in records:
            for candidate in record['candidates']:
                terminal_event = terminal_by_arch.get(digest(candidate['architecture']))
                if terminal_event is not None:
                    trajectory = json.loads((folder / 'candidates' / terminal_event['candidate'] / 'trajectory.json').read_text())
                    actual_extra = sum(r['training_seconds'] + r['validation_seconds'] for r in trajectory if record['budget'] < r['epoch'] <= 30)
                    survivor_rows.append({'variant': variant, 'seed': s['seed'], 'budget': record['budget'],
                                          'observed_terminal_f1': terminal_event['fitness'], 'predicted_final_f1': candidate['predicted_final_f1'],
                                          'covered': candidate['lower'] <= terminal_event['fitness'] <= candidate['upper'],
                                          'predicted_remaining_seconds': candidate['predicted_remaining_execution_seconds'], 'actual_remaining_seconds': actual_extra,
                                          'scope': 'terminal survivors only; selection-biased coverage'})
            raw = np.array([c['observed_macro_f1'] for c in record['candidates']])
            ordinary = np.argsort(-raw, kind='stable')[:len(record['selected_indices'])]
            promotion_rows.append({'variant': variant, 'seed': s['seed'], 'generation': record['generation'], 'budget': record['budget'],
                                   'selected': len(record['selected_indices']),
                                   'differs_from_observed_ranking': set(ordinary.tolist()) != set(record['selected_indices']),
                                   'uncertainty_changes_ranking_same_pool': (set(np.argsort(-np.array([c['predicted_final_f1'] - (.05 if variant == 'ua_amf_bpso' else 0.) * c['predicted_remaining_execution_seconds'] / 60 for c in record['candidates']]), kind='stable')[:len(record['selected_indices'])].tolist()) != set(record['selected_indices'])) if variant != 'bpso_prediction' else False,
                                   'cost_changes_ranking_same_pool': (set(np.argsort(-np.array([c['predicted_final_f1'] + c['radius'] for c in record['candidates']]), kind='stable')[:len(record['selected_indices'])].tolist()) != set(record['selected_indices'])) if variant == 'ua_amf_bpso' else False})
        promoted = {e['candidate'] for e in events if e.get('requested_budget', 0) > 3 and e.get('completed_budget', 0) - e.get('charged_epochs', 0) > 0}
        if winner:
            save_json(folder / 'strict_budget_winner.json', winner)
            path = folder / 'candidates' / winner['candidate']
            trajectory = json.loads((path / 'trajectory.json').read_text())
            metrics = verify_prediction(path / 'validation_epoch_30.npz', trajectory, meta['classes'])
            metric_rows.append({'variant': variant, 'search_seed': s['seed'], 'scope': 'search validation winner, not final independent model', **metrics})
        summary_rows.append({'variant': variant, 'seed': s['seed'], 'wall_seconds': s['wall_seconds'], 'gpu_seconds': s['gpu_seconds'],
                             'charged_gpu_seconds': s['charged_gpu_seconds_upper_bound'], 'epochs': s['charged_epochs'],
                             'unique_architectures': s['unique_architectures'], 'promoted_architectures': len(promoted),
                             'peak_gpu_memory_bytes': s['peak_gpu_memory_bytes'], 'failures': s['failed_evaluations'],
                             'numerical_or_memory_failures': sum(bool(e['failure']) and 'time budget' not in e['failure'] for e in events),
                             'budget_stop_events': sum(bool(e['failure']) and 'time budget' in e['failure'] for e in events),
                             'retry_attempts': sum(max(0, count - 1) for count in __import__('collections').Counter((e['candidate'], e.get('requested_budget')) for e in events).values()),
                             'scheduler_seconds': s['scheduler_and_setup_seconds'], 'predictor_seconds': state.get('prediction_seconds', 0.),
                             'predictor_load_seconds': state.get('predictor_load_seconds', 0.),
                             'best_f1_at_30_seconds': winner['fitness'] if winner else None,
                             'winner_parameters': winner.get('parameters') if winner else None,
                             'winner_architecture': digest(winner['architecture']) if winner else None,
                             'last_epoch_boundary_overrun_seconds': max(0., s['wall_seconds'] - 30),
                             'terminal_evaluations_within_cap': len(eligible)})
    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return
    baseline = frame[frame.variant == 'bpso'].set_index('seed')
    for measure in ['wall_seconds', 'gpu_seconds']:
        frame[f'{measure}_saving_vs_bpso_percent'] = [100 * (baseline.loc[r.seed, measure] - getattr(r, measure)) / baseline.loc[r.seed, measure] for r in frame.itertuples()]
    reliability = json.loads((study / 'reliability_summary.json').read_text())
    predictor = json.loads((study / 'predictor_report.json').read_text())
    frame['unamortized_gpu_seconds'] = frame.gpu_seconds + frame.variant.str.startswith(('bpso_prediction', 'ua_amf')).astype(int) * reliability['gpu_seconds']
    frame['unamortized_wall_seconds'] = frame.wall_seconds + frame.variant.str.startswith(('bpso_prediction', 'ua_amf')).astype(int) * (reliability['wall_seconds'] + predictor['total_build_wall_seconds'])
    frame.to_csv(root / 'seed_results.csv', index=False)
    save_json(root / 'seed_results.json', json_rows(frame))
    pd.DataFrame(anytime).to_csv(root / 'anytime_results.csv', index=False)
    pd.DataFrame(targets).to_csv(root / 'time_to_target.csv', index=False)
    pd.DataFrame(promotion_rows).to_csv(root / 'promotion_analysis.csv', index=False)
    pd.DataFrame(survivor_rows).to_csv(root / 'promotion_survivor_analysis.csv', index=False)
    save_json(root / 'selected_validation_metrics.json', metric_rows)
    save_json(root / 'budget_audit.json', budget_checks)
    save_json(root / 'prediction_audit.json', {'verified_archives': len(audit), 'all_match': True})
    grouped = frame.groupby('variant').agg(n=('seed', 'count'), f1=('best_f1_at_30_seconds', 'mean'),
                   f1_sd=('best_f1_at_30_seconds', 'std'), wall=('wall_seconds', 'mean'), gpu=('gpu_seconds', 'mean'),
                   gpu_sd=('gpu_seconds', 'std'), scheduler=('scheduler_seconds', 'mean'), predictor=('predictor_seconds', 'mean'),
                   epochs=('epochs', 'mean'), unique=('unique_architectures', 'mean'), promotions=('promoted_architectures', 'mean'),
                   memory=('peak_gpu_memory_bytes', 'max'), failures=('failures', 'sum'),
                   total_gpu=('unamortized_gpu_seconds', 'mean'))
    grouped['gpu_saving_percent'] = 100 * (grouped.loc['bpso', 'gpu'] - grouped.gpu) / grouped.loc['bpso', 'gpu']
    grouped['wall_saving_percent'] = 100 * (grouped.loc['bpso', 'wall'] - grouped.wall) / grouped.loc['bpso', 'wall']
    grouped.to_csv(root / 'comparison.csv')
    latex = [r'\begin{tabular}{lrrr}\toprule', r'Method & Macro-F1 & GPU s & Saving (\%)\\\midrule']
    markdown = ['| Method | Seeds | Macro-F1 mean ± SD at 30 s | GPU s mean ± SD | Wall s | GPU saving vs BPSO |', '|---|---:|---:|---:|---:|---:|']
    for name, row in grouped.iterrows():
        label = name.replace('_', r'\_')
        latex.append(f'{label} & {row.f1:.4f} & {row.gpu:.2f} & {row.gpu_saving_percent:.1f}' + chr(92)*2)
        markdown.append(f'| {name} | {int(row.n)} | {row.f1:.4f} ± {row.f1_sd:.4f} | {row.gpu:.2f} ± {row.gpu_sd:.2f} | {row.wall:.2f} | {row.gpu_saving_percent:.1f}% |')
    latex.append(r'\bottomrule\end{tabular}')
    (root / 'comparison.tex').write_text('\n'.join(latex) + '\n')
    (root / 'comparison.md').write_text('\n'.join(markdown) + '\n')
    target_frame = pd.DataFrame(targets)
    target_frame.groupby(['variant', 'target']).agg(attained=('attained', 'sum'), total=('seed', 'count'),
                         conditional_mean_seconds=('seconds', 'mean')).to_csv(root / 'time_to_target_summary.csv')
    paired_targets = []
    for target, group in target_frame.groupby('target'):
        b = group[group.variant == 'bpso'].set_index('seed')
        p = group[group.variant == 'ua_amf_bpso'].set_index('seed')
        for seed in b.index.intersection(p.index):
            if b.loc[seed, 'attained'] and p.loc[seed, 'attained']:
                paired_targets.append({'target': target, 'seed': int(seed), 'baseline_seconds': b.loc[seed, 'seconds'],
                                       'proposed_seconds': p.loc[seed, 'seconds'],
                                       'saving_percent': 100 * (b.loc[seed, 'seconds'] - p.loc[seed, 'seconds']) / b.loc[seed, 'seconds']})
    pd.DataFrame(paired_targets).to_csv(root / 'paired_time_to_target.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    anytime_frame = pd.DataFrame(anytime)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, group in anytime_frame.groupby('variant'):
        stats = group.groupby('limit_seconds').best_f1.agg(['mean', 'std', 'count'])
        stats = stats[stats['count'] == len(protocol['seeds'])]
        ax.errorbar(stats.index, stats['mean'], yerr=stats['std'].fillna(0), marker='o', capsize=2, label=name)
    ax.set(xlabel='Measured cumulative search wall budget (s)', ylabel='Best observed 30-epoch validation macro-F1', xlim=(8, 32))
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(root / 'anytime_f1.png', dpi=170)
    fig.savefig(root / 'anytime_f1.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.get_cmap('tab10')
    for color_index, name in enumerate(sorted(frame.variant.unique())):
        for seed_index, seed in enumerate(protocol['seeds']):
            trace_path = root / name / f'seed_{seed}' / 'anytime_trace.json'
            if not trace_path.exists():
                continue
            trace = [r for r in json.loads(trace_path.read_text()) if r['seconds'] <= 30]
            if trace:
                xs = [r['seconds'] for r in trace] + [30]
                ys = [r['best_f1'] for r in trace] + [trace[-1]['best_f1']]
                ax.step(xs, ys, where='post', color=colors(color_index), alpha=.6, label=name if seed_index == 0 else None)
    ax.set(xlabel='Actual cumulative search wall time (s)', ylabel='Best actual 30-epoch validation macro-F1', xlim=(0, 30), ylim=(0, 1))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / 'anytime_seed_traces.png', dpi=170)
    fig.savefig(root / 'anytime_seed_traces.pdf')
    plt.close(fig)
    # Calibration plot: all held-out observations, not archive training predictions.
    heldout = pd.read_csv(study / 'predictor_heldout_predictions.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, (budget, group) in zip(axes, heldout.groupby('budget')):
        ax.errorbar(group.target, group.prediction,
                    yerr=np.vstack([group.prediction - group.lower, group.upper - group.prediction]), fmt='o', alpha=.6)
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set(title=f'Epoch {budget}', xlabel='Actual epoch-30 macro-F1', xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel('Predicted epoch-30 macro-F1 and interval')
    fig.tight_layout()
    fig.savefig(study / 'calibration.png', dpi=170)
    plt.close(fig)
    print('\n'.join(markdown), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', default='results/ua_amf_prediction_20261108')
    parser.add_argument('--benchmark', default='results/ua_amf_prediction_benchmark_20261108')
    args = parser.parse_args()
    study, root = Path(args.study), Path(args.benchmark)
    print('Cohort archives verified:', analyze_cohort(study))
    analyze_benchmark(study, root)


if __name__ == '__main__':
    main()
