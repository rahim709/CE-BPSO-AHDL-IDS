"""Learning-curve/conformal variant of UA-AMF-BPSO; legacy SEM path retained."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import time
import numpy as np
from src.optimization.bpso import BPSO, SwarmSettings
from src.optimization.curve_predictor import CurvePredictor
from src.optimization.evaluator import BudgetExhausted
from src.utils.core import digest, save_json


def acquisition(mean, radius, additional_seconds, beta=1., cost_weight=.05, cost_unit_seconds=60.):
    if not np.isfinite([mean, radius, additional_seconds, beta, cost_weight, cost_unit_seconds]).all():
        raise ValueError('Nonfinite acquisition component')
    if min(radius, additional_seconds, beta, cost_weight) < 0 or cost_unit_seconds <= 0:
        raise ValueError('Invalid acquisition component')
    return float(mean + beta * radius - cost_weight * additional_seconds / cost_unit_seconds)


def predictive_search(evaluator, codec, cfg):
    search = cfg['search']
    budgets = search.get('budgets', [3, 10, 30])
    if budgets != sorted(set(budgets)) or not budgets or budgets[0] < 1 or budgets[-1] != search['epochs']:
        raise ValueError('Invalid cumulative rung ladder')
    began = time.perf_counter()
    bundle = Path(search['predictor_bundle'])
    if search.get('predictor_sha256') and hashlib.sha256(bundle.read_bytes()).hexdigest() != search['predictor_sha256']:
        raise ValueError('Predictor bundle changed after protocol freeze')
    predictor = CurvePredictor.load(bundle)
    load_seconds = time.perf_counter() - began
    context = digest({'data_fingerprint': evaluator.metadata['fingerprint'], 'training': cfg['training'], 'space': cfg.get('space', 'paper')})
    if predictor.context != context or any(b not in predictor.models for b in budgets[:-1]):
        raise ValueError('Predictor context or available rungs differ from search')
    settings = SwarmSettings(**{k: v for k, v in search.items() if k in SwarmSettings.__dataclass_fields__})
    swarm = BPSO(codec.length, settings, cfg.get('seed', 42))
    records, finalists, trace = [], [], []
    prediction_seconds = 0.
    beta, cost_weight = search.get('uncertainty_beta', 1.), search.get('cost_weight', .05)
    fraction = search.get('restart_fraction', .2)
    if not 0 <= fraction <= 1:
        raise ValueError('restart_fraction must be in [0,1]')
    try:
        for generation in range(settings.generations):
            if not evaluator.available():
                raise BudgetExhausted('Search resource cap reached')
            unique = {}
            for i, bits in enumerate(swarm.position):
                # Never use a development-archive terminal target for a search proposal.
                while digest(codec.decode(bits)) in predictor.archive_architectures:
                    bits = codec.sample(swarm.rng)
                    swarm.position[i] = bits
                key = digest(codec.decode(bits))
                unique.setdefault(key, {'bits': bits.copy(), 'particles': []})['particles'].append(i)
            active = list(unique.values())
            fitness = np.full(settings.population, -np.inf)
            for level, budget in enumerate(budgets):
                valid, scores, rung_records = [], [], []
                for candidate in active:
                    bits = candidate['bits']
                    score = evaluator.evaluate(bits, budget)
                    if not np.isfinite(score):
                        continue
                    architecture = codec.decode(bits)
                    if budget == budgets[-1]:
                        entry = {'architecture': architecture, 'fitness': score, 'completed_budget': budget,
                                 'replicate_seeds': [evaluator.seed], 'observed_seed_fitness': [score],
                                 'search_wall_seconds': evaluator.wall_seconds}
                        finalists.append(entry)
                        fitness[candidate['particles']] = score
                        continue
                    path = evaluator.candidate_path(architecture, evaluator.seed) / 'trajectory.json'
                    trajectory = json.loads(path.read_text())
                    start = time.perf_counter()
                    mean, radius, low, high = predictor.predict(architecture, trajectory, budget, context, codec=codec)
                    prediction_seconds += time.perf_counter() - start
                    observed = [r for r in trajectory if r['epoch'] <= budget]
                    epoch_cost = np.mean([r['training_seconds'] + r['validation_seconds'] for r in observed])
                    cost = float(epoch_cost * (budgets[-1] - budget))
                    value = acquisition(mean[0], radius[0], cost, beta, cost_weight, search.get('cost_unit_seconds', 60.))
                    valid.append(candidate)
                    scores.append(value)
                    rung_records.append({'architecture': architecture, 'observed_macro_f1': score,
                                         'predicted_final_f1': float(mean[0]), 'radius': float(radius[0]),
                                         'lower': float(low[0]), 'upper': float(high[0]),
                                         'predicted_remaining_execution_seconds': cost, 'acquisition': value})
                if budget == budgets[-1] or not valid:
                    break
                count = max(1, len(valid) // search.get('eta', 3))
                selected = np.argsort(-np.asarray(scores), kind='stable')[:count]
                records.append({'generation': generation, 'budget': budget, 'next_budget': budgets[level + 1],
                                'candidates': rung_records, 'selected_indices': selected.tolist()})
                active = [valid[i] for i in selected]
            swarm.tell(fitness)  # Same actually observed terminal fidelity for every attractor.
            trace.append({'generation': generation, 'terminal_evaluations': int(np.isfinite(fitness).sum()),
                          'best_fitness': swarm.best_fitness if np.isfinite(swarm.best_fitness) else None})
            swarm.step()
            n = int(np.ceil(settings.population * fraction))
            if n:
                indices = np.argsort(swarm.personal_fitness, kind='stable')[:n]
                swarm.position[indices] = swarm.rng.integers(0, 2, (n, codec.length), dtype=np.int8)
                swarm.personal_position[indices] = swarm.position[indices]
                swarm.personal_fitness[indices] = -np.inf
    finally:
        save_json(evaluator.output / 'promotions.json', records)
        save_json(evaluator.output / 'swarm_trace.json', trace)
        save_json(evaluator.output / 'adaptive_state.json', {'finalists': finalists, 'predictor_load_seconds': load_seconds,
                  'prediction_seconds': prediction_seconds, 'predictor_fit_charged_separately': True,
                  'scheduler_inclusive_seconds_this_invocation': time.perf_counter() - began})
    return max(finalists, key=lambda r: r['fitness']) if finalists else None
