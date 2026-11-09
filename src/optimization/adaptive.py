"""Replicated-seed, cost-aware rung allocation; no future-fidelity confidence claim."""
from __future__ import annotations
import json
import time
import numpy as np
from src.optimization.bpso import BPSO, SwarmSettings
from src.optimization.evaluator import BudgetExhausted
from src.utils.core import digest, save_json


def promotion_scores(values, costs, beta=1., cost_weight=.05):
    """SEM describes initialization variation conditional on this validation split.

    The optimistic score is a heuristic, not a calibrated confidence bound.
    Independent training seeds are the sampling units, never successive epochs.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite candidate-by-seed observations')
    if beta and values.shape[1] < 3:
        raise ValueError('Uncertainty requires at least three independent seeds')
    means = values.mean(axis=1)
    sem = values.std(axis=1, ddof=1) / np.sqrt(values.shape[1]) if values.shape[1] > 1 else np.zeros(len(values))
    costs = np.asarray(costs, dtype=float)
    if costs.shape != means.shape or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError('Expected positive finite incremental costs')
    return means + beta * sem - cost_weight * costs / np.median(costs), means, sem


def adaptive_search(evaluator, codec, cfg):
    search = cfg['search']
    budgets = search.get('budgets', [3, 10, 30])
    if not budgets or budgets != sorted(set(budgets)) or budgets[0] < 1 or budgets[-1] != search.get('epochs', 30):
        raise ValueError('Budgets must be strictly increasing and end at search.epochs')
    offsets = search.get('replicate_offsets', [0, 1000, 2000])
    beta = float(search.get('uncertainty_beta', 1.))
    weight = float(search.get('cost_weight', .05))
    if len(set(offsets)) != len(offsets) or not offsets or beta < 0 or weight < 0:
        raise ValueError('Invalid replication or promotion settings')
    if beta and len(offsets) < 3:
        raise ValueError('Uncertainty requires at least three independent seeds')
    settings = SwarmSettings(**{k: v for k, v in search.items() if k in SwarmSettings.__dataclass_fields__})
    swarm = BPSO(codec.length, settings, cfg.get('seed', 42))
    records, finalists = [], []
    restarted = 0
    began = time.perf_counter()
    try:
        for generation in range(settings.generations):
            # Canonicalize inactive slots: equivalent decoded architectures share one job.
            unique = {}
            for i, bits in enumerate(swarm.position):
                key = digest(codec.decode(bits))
                unique.setdefault(key, {'bits': bits.copy(), 'particles': []})['particles'].append(i)
            active = list(unique.values())
            fitness = np.full(settings.population, -np.inf)
            for level, budget in enumerate(budgets):
                valid, observations, costs = [], [], []
                for candidate in active:
                    bits = candidate['bits']
                    seeds = [cfg.get('seed', 42) + offset for offset in offsets]
                    # Reserve the full replicated job before starting; partial jobs never promote.
                    needed = 0
                    for seed in seeds:
                        path = evaluator.candidate_path(codec.decode(bits), seed) / 'trajectory.json'
                        rows = json.loads(path.read_text()) if path.exists() else []
                        needed += max(0, budget - max((r['epoch'] for r in rows), default=0))
                    if not evaluator.available(needed):
                        raise BudgetExhausted('Replicated job exceeds shared compute cap')
                    values = [evaluator.evaluate(bits, budget, seed) for seed in seeds]
                    if not np.isfinite(values).all():
                        continue
                    valid.append(candidate)
                    observations.append(values)
                    elapsed = []
                    for seed in seeds:
                        path = evaluator.candidate_path(codec.decode(bits), seed) / 'trajectory.json'
                        rows = json.loads(path.read_text())
                        elapsed.extend(r['training_seconds'] + r['validation_seconds'] for r in rows if r['epoch'] <= budget)
                    next_budget = budgets[min(level + 1, len(budgets) - 1)]
                    costs.append(max(1e-9, np.mean(elapsed) * max(1, next_budget - budget) * len(seeds)))
                if not valid:
                    break
                scores, means, sem = promotion_scores(observations, costs, beta, weight)
                if level == len(budgets) - 1:
                    for index, (candidate, mean, error) in enumerate(zip(valid, means, sem)):
                        architecture = codec.decode(candidate['bits'])
                        entry = {'architecture': architecture, 'fitness': float(mean), 'standard_error': float(error),
                                 'completed_budget': budget, 'replicate_seeds': seeds, 'observed_seed_fitness': observations[index]}
                        finalists.append(entry)
                        fitness[candidate['particles']] = mean
                    break
                count = max(1, len(valid) // search.get('eta', 3))
                order = np.argsort(-scores, kind='stable')[:count]
                records.append({'generation': generation, 'budget': budget, 'next_budget': budgets[level + 1],
                                'architectures': [codec.decode(c['bits']) for c in valid],
                                'observations': observations, 'means': means.tolist(), 'sem': sem.tolist(),
                                'predicted_incremental_seconds': costs, 'promotion_scores': scores.tolist(),
                                'selected_indices': order.tolist()})
                active = [valid[i] for i in order]
            swarm.tell(fitness)  # Only common maximum-fidelity means update swarm attractors.
            swarm.step()
            fraction = float(search.get('restart_fraction', .2))
            if not 0 <= fraction <= 1:
                raise ValueError('restart_fraction must lie in [0,1]')
            n = min(settings.population, int(np.ceil(settings.population * fraction)))
            if n:
                indices = np.argsort(swarm.personal_fitness, kind='stable')[:n]
                swarm.position[indices] = swarm.rng.integers(0, 2, (n, codec.length), dtype=np.int8)
                swarm.personal_position[indices] = swarm.position[indices]
                swarm.personal_fitness[indices] = -np.inf
                restarted += n
    finally:
        save_json(evaluator.output / 'promotions.json', records)
        save_json(evaluator.output / 'adaptive_state.json', {'replay': 'deterministic decisions with persisted evaluations',
                  'restart_count': restarted, 'finalists': finalists,
                  'scheduler_inclusive_seconds_this_invocation': time.perf_counter() - began})
    return max(finalists, key=lambda row: row['fitness']) if finalists else None
