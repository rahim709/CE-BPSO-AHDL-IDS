"""Controlled architecture cohort and censored long-budget extension."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from src.evaluation.ranking import ranking_metrics
from src.optimization.encoding import BinaryCodec
from src.optimization.evaluator import Evaluator
from src.utils.core import digest, save_json


def diverse_candidates(codec: BinaryCodec, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    cohort, seen = [], set()
    for _ in range(count):
        pool = [codec.sample(rng) for _ in range(128)]
        pool = [p for p in pool if digest(codec.decode(p)) not in seen]
        if not pool:
            raise ValueError('Cannot construct unique architecture cohort')
        # Greedy max-min Hamming spread on canonical active-architecture encodings.
        canonical = [codec.encode(codec.decode(p)) for p in pool]
        if cohort:
            distances = [min(np.count_nonzero(p != q) for q in cohort) for p in canonical]
            chosen = canonical[int(np.argmax(distances))]
        else:
            chosen = canonical[0]
        cohort.append(chosen)
        seen.add(digest(codec.decode(chosen)))
    return cohort


def run_diagnostics(cfg: dict, output: str | Path) -> dict:
    output = Path(output)
    codec = BinaryCodec(cfg.get('space', 'paper'))
    settings = cfg.get('diagnostics', {})
    epochs = sorted(set(settings.get('epochs', [3, 10, 30])))
    final_epoch = settings.get('long_epochs', 200)
    seeds = settings.get('seeds', [cfg.get('seed', 42)])
    eta = cfg.get('search', {}).get('eta', 3)
    if eta < 2:
        raise ValueError('Diagnostic promotion factor eta must be at least two')
    keep_fraction = 1 / eta
    if len(epochs) < 2 or epochs[0] < 1 or len(seeds) == 0 or len(set(seeds)) != len(seeds):
        raise ValueError('Diagnostics require at least two positive epochs and unique, nonempty seeds')
    if settings.get('candidates', 25) < 2 or final_epoch < epochs[-1]:
        raise ValueError('Need at least two candidates and a long budget at or above the reference')
    cohort = diverse_candidates(codec, settings.get('candidates', 25), settings.get('cohort_seed', 1729))
    evaluator = Evaluator(cfg, output, codec)
    save_json(output / 'cohort.json', [{'index': i, 'bits': p.tolist(), 'architecture': codec.decode(p)} for i, p in enumerate(cohort)])
    reports = []
    seed_reference_scores = []
    class_changes = []
    for seed in seeds:
        scores = np.zeros((len(epochs), len(cohort)))
        for i, bits in enumerate(cohort):
            # Every candidate reaches the common reference budget before selection.
            for j, budget in enumerate(epochs):
                scores[j, i] = evaluator.evaluate(bits, budget, seed)
        if not np.isfinite(scores).all():
            raise RuntimeError('Failed diagnostic candidate; resolve numerical failures before paired analysis')
        seed_reference_scores.append(scores[-1].tolist())
        for j, budget in enumerate(epochs[:-1]):
            reports.append({'seed': seed, 'early_epoch': budget, 'late_epoch': epochs[-1],
                            'scope': 'complete cohort', 'promotion_eta': eta,
                            **ranking_metrics(scores[j], scores[-1], settings.get('top_k', 5), keep_fraction)})
        early_ranks = np.argsort(np.argsort(-scores[0], kind='stable'), kind='stable')
        late_ranks = np.argsort(np.argsort(-scores[-1], kind='stable'), kind='stable')
        for i, bits in enumerate(cohort):
            trajectory = json.loads((evaluator.candidate_path(codec.decode(bits), seed) / 'trajectory.json').read_text())
            early = next(r for r in trajectory if r['epoch'] == epochs[0])
            late = next(r for r in trajectory if r['epoch'] == epochs[-1])
            for name in early['per_class_recall']:
                class_changes.append({'candidate': i, 'seed': seed, 'class': name,
                                      'early_rank': int(early_ranks[i]) + 1, 'late_rank': int(late_ranks[i]) + 1,
                                      'rank_improvement': int(early_ranks[i] - late_ranks[i]),
                                      'first_epoch_recall_at_least_0_8': next((r['epoch'] for r in trajectory if r['per_class_recall'][name] >= .8), None),
                                      'training_support': evaluator.metadata['splits']['train']['classes'].get(name, 0),
                                      'early_recall': early['per_class_recall'][name],
                                      'late_recall': late['per_class_recall'][name],
                                      'recall_gain': late['per_class_recall'][name] - early['per_class_recall'][name]})
        shortlist = set(np.argsort(-scores[-1], kind='stable')[:settings.get('shortlist', 5)].tolist())
        # Audit early-rejected candidates independently of their late performance.
        early_order = np.argsort(-scores[0], kind='stable')
        rejected = early_order[max(1, int(np.floor(len(cohort) * keep_fraction))):]
        audit_rng = np.random.default_rng(seed + 9182)
        count = min(settings.get('audit_rejected', 5), len(rejected))
        shortlist.update(audit_rng.choice(rejected, size=count, replace=False).tolist())
        # Add median and bottom examples selected at the complete reference epoch.
        late_order = np.argsort(-scores[-1], kind='stable')
        shortlist.update([int(late_order[len(cohort)//2]), int(late_order[-1])])
        long_scores = {}
        if final_epoch > epochs[-1]:
            for i in sorted(shortlist):
                long_scores[i] = evaluator.evaluate(cohort[i], final_epoch, seed)
        save_json(output / f'long_cohort_seed_{seed}.json', {
            'selected_indices': sorted(shortlist), 'long_scores': long_scores,
            'selection_rule': 'top at common reference + uniform early-rejected audit + median and worst at reference',
            'censoring': ('none; complete cohort evaluated at long budget' if len(long_scores) == len(cohort)
                          else 'Long-budget full-cohort ranks/regret/elimination are NOT identifiable from this selected subset')})
        if len(long_scores) > 1:
            indices = sorted(long_scores)
            reports.append({'seed': seed, 'early_epoch': epochs[-1], 'late_epoch': final_epoch,
                            'promotion_eta': eta,
                            'scope': ('complete cohort' if len(indices) == len(cohort)
                                      else 'selected subset ONLY; selection-biased'),
                            **ranking_metrics(scores[-1, indices], np.array([long_scores[i] for i in indices]),
                                              min(settings.get('top_k', 5), len(indices)), keep_fraction)})
    evaluator.persist()
    import pandas as pd
    pd.DataFrame(class_changes).to_csv(output / 'class_learning_changes.csv', index=False)
    matrix = np.asarray(seed_reference_scores)
    seed_comparisons = []
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            winners_a = set(np.flatnonzero(matrix[a] == matrix[a].max()).tolist())
            winners_b = set(np.flatnonzero(matrix[b] == matrix[b].max()).tolist())
            seed_comparisons.append({'seed_a': seeds[a], 'seed_b': seeds[b],
                                     'winner_agreement': (winners_a == winners_b if len(winners_a) == len(winners_b) == 1 else None),
                                     'winner_sets_overlap': bool(winners_a & winners_b),
                                     'winner_indices_a': sorted(winners_a), 'winner_indices_b': sorted(winners_b),
                                     **ranking_metrics(matrix[a], matrix[b], settings.get('top_k', 5), keep_fraction)})
    save_json(output / 'fitness_seed_reliability.json', {
        'seeds': seeds, 'reference_epoch': epochs[-1], 'candidate_scores': matrix.tolist(),
        'candidate_fitness_std': matrix.std(axis=0, ddof=1).tolist() if len(seeds) > 1 else None,
        'seed_pairs': seed_comparisons,
        'interpretation': 'Candidate score variability and winner disagreement diagnose noisy best selection; causal BPSO effects require controlled replay.'})
    result = {'rankings': reports, 'charged_epochs': evaluator.used_epochs,
              'synthetic_data': bool(evaluator.metadata.get('config', {}).get('synthetic', False)),
              'wall_seconds': evaluator.wall_seconds, 'gpu_seconds': sum(e['gpu_seconds'] for e in evaluator.events),
              'conclusion': 'Hypothesis requires real-data repeated-seed analysis; no automatic novelty claim.'}
    save_json(output / 'ranking_diagnostics.json', result)
    return result
