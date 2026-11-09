"""Established search mechanics sharing one space and evaluator.

All schedulers are serial. Promotions resume weights/optimizer/RNG. Deterministic
replay plus persistent evaluation caching resumes scheduler decisions after a stop.
"""
from __future__ import annotations
import math
from pathlib import Path
import time
import numpy as np
from src.optimization.bpso import BPSO, SwarmSettings
from src.optimization.encoding import BinaryCodec
from src.optimization.evaluator import BudgetExhausted, Evaluator, TargetReached
from src.utils.core import digest, save_json


def halving(evaluator: Evaluator, candidates: list[np.ndarray], budgets: list[int], eta: int) -> dict[int, float]:
    active = list(range(len(candidates)))
    final = {}
    for level, budget in enumerate(budgets):
        scores = {i: evaluator.evaluate(candidates[i], budget) for i in active}
        if level == len(budgets) - 1:
            final = scores
            break
        active = sorted(active, key=lambda i: (-scores[i], i))[:max(1, len(active) // eta)]
    return final


def run_search(cfg: dict, output: str | Path) -> dict:
    codec = BinaryCodec(cfg.get('space', 'paper'))
    evaluator = Evaluator(cfg, output, codec)
    search = cfg['search']
    method = search.get('method', 'bpso')
    seed = cfg.get('seed', 42)
    rng = np.random.default_rng(seed)
    maximum = search.get('epochs', 30)
    minimum = search.get('min_epochs', 3)
    eta = search.get('eta', 3)
    if eta < 2 or minimum < 1 or maximum < minimum:
        raise ValueError('Invalid fidelity bounds or eta')
    status = 'completed'
    adaptive_best = None
    try:
        if method == 'ua_amf_bpso':
            if search.get('promotion_policy') == 'online':
                from src.optimization.online import archive_free_search
                adaptive_best = archive_free_search(evaluator, codec, cfg)
            elif search.get('promotion_policy') == 'dynamic':
                from src.optimization.dynamic import dynamic_search
                adaptive_best = dynamic_search(evaluator, codec, cfg)
            elif search.get('promotion_policy') == 'learning_curve':
                from src.optimization.predictive import predictive_search
                adaptive_best = predictive_search(evaluator, codec, cfg)
            else:
                from src.optimization.adaptive import adaptive_search
                adaptive_best = adaptive_search(evaluator, codec, cfg)
        elif method in {'bpso', 'bpso_reduced', 'bpso_sh'}:
            settings = SwarmSettings(**{k: v for k, v in search.items() if k in SwarmSettings.__dataclass_fields__})
            swarm = BPSO(codec.length, settings, seed)
            budgets = search.get('budgets', sorted(set([minimum, min(maximum, minimum * eta), maximum])))
            if budgets != sorted(set(budgets)) or budgets[0] < 1 or budgets[-1] != maximum:
                raise ValueError('Invalid configured rung ladder')
            for _ in range(settings.generations):
                if method == 'bpso_sh':
                    scores = halving(evaluator, list(swarm.position), budgets, eta)
                    fitness = np.asarray([scores.get(i, -np.inf) for i in range(settings.population)])
                else:
                    budget = minimum if method == 'bpso_reduced' else maximum
                    fitness = np.asarray([evaluator.evaluate(p, budget) for p in swarm.position])
                swarm.tell(fitness)
                # Nonfinite eliminated particles are represented by null for strict JSON.
                history = [{k: [None if not np.isfinite(x) else x for x in v] if isinstance(v, list)
                            else (None if isinstance(v, float) and not np.isfinite(v) else v)
                            for k, v in row.items()} for row in swarm.history]
                save_json(Path(output) / 'convergence.json', history)
                swarm.step()
        elif method in {'random', 'surrogate'}:
            observed: list[tuple[np.ndarray, float]] = []
            for i in range(search.get('candidates', 600)):
                candidate = codec.sample(rng)
                if method == 'surrogate' and len(observed) >= search.get('warmup', 10):
                    # Standard random-forest acquisition; no novelty claim. Epsilon exploration.
                    from sklearn.ensemble import RandomForestRegressor
                    pool = [codec.sample(rng) for _ in range(64)]
                    if rng.random() > .2:
                        forest = RandomForestRegressor(n_estimators=64, random_state=seed + i, n_jobs=1)
                        forest.fit(np.stack([p for p, _ in observed]), [f for _, f in observed])
                        candidate = pool[int(np.argmax(forest.predict(pool)))]
                score = evaluator.evaluate(candidate, maximum)
                if np.isfinite(score):
                    observed.append((candidate.copy(), score))
        elif method == 'hyperband':
            # Li et al. standard brackets; integer cumulative epoch budgets.
            smax = int(math.floor(math.log(maximum / minimum, eta)))
            for _ in range(search.get('brackets', 1)):
                for s in range(smax, -1, -1):
                    n = int(math.ceil((smax + 1) / (s + 1) * eta ** s))
                    r = maximum / eta ** s
                    budgets = [max(1, int(round(r * eta ** i))) for i in range(s + 1)]
                    halving(evaluator, [codec.sample(rng) for _ in range(n)], budgets, eta)
        elif method == 'dehb':
            from dehb import DEHB
            # Upstream DEHB 0.1.1 uses fidelity, not the historical budget API.
            dehb_exhausted = None
            def objective(config: np.ndarray, fidelity: float, **kwargs: object) -> dict:
                nonlocal dehb_exhausted
                bits = (np.asarray(config) >= .5).astype(np.int8)
                began = time.perf_counter()
                try:
                    score = evaluator.evaluate(bits, max(1, int(round(fidelity))))
                except BudgetExhausted as exc:
                    dehb_exhausted = exc
                    raise
                return {'fitness': 1 - score if np.isfinite(score) else 1., 'cost': time.perf_counter() - began}
            optimizer = DEHB(f=objective, dimensions=codec.length, min_fidelity=minimum, max_fidelity=maximum,
                             eta=eta, n_workers=1, seed=seed, output_path=str(Path(output) / 'dehb'))
            optimizer.run(brackets=search.get('brackets', 1))
            # Upstream loguru catch may suppress objective budget exceptions.
            if dehb_exhausted is not None:
                raise dehb_exhausted
        else:
            raise ValueError(f'Unknown method {method}')
    except TargetReached:
        status = 'target_reached'
    except BudgetExhausted:
        status = 'budget_exhausted'
    finally:
        evaluator.persist()
    if method == 'ua_amf_bpso':
        state_path = Path(output) / 'adaptive_state.json'
        if state_path.exists():
            import json
            finalists = json.loads(state_path.read_text())['finalists']
            adaptive_best = max(finalists, key=lambda row: row['fitness']) if finalists else None
        best = adaptive_best
    else:
        best = evaluator.best()
    if status == 'target_reached':
        best = evaluator.best()
    required_fidelity = minimum if method == 'bpso_reduced' else maximum
    selection_complete = best is not None and best['completed_budget'] >= required_fidelity
    result = {'status': status, 'target_reached': status == 'target_reached', 'stop_f1': search.get('stop_f1'), 'method': method, 'seed': seed, 'data_fingerprint': evaluator.metadata['fingerprint'],
              'required_selection_fidelity': required_fidelity,
              'selection_complete': selection_complete,
              'selection_status': ('complete' if selection_complete else
                                   'incomplete_fidelity' if best else 'no_valid_candidate'),
              'best': best, 'charged_epochs': evaluator.used_epochs, 'wall_seconds': evaluator.wall_seconds,
              'gpu_seconds': sum(e['gpu_seconds'] for e in evaluator.events),
              'charged_gpu_seconds_upper_bound': evaluator.charged_gpu_seconds, 'cache_hits_this_invocation': evaluator.cache_hits,
              'compute_cap': evaluator.budget,
              'unique_architectures': len({digest(e['architecture']) for e in evaluator.events}),
              'peak_gpu_memory_bytes': max((e.get('peak_gpu_memory_bytes', 0) for e in evaluator.events), default=0),
              'failed_evaluations': sum(e['failure'] is not None for e in evaluator.events),
              'evaluation_wall_seconds': sum(e['wall_seconds'] for e in evaluator.events),
              'scheduler_and_setup_seconds': max(0., evaluator.wall_seconds - sum(e['wall_seconds'] for e in evaluator.events)),
              'synthetic_data': bool(evaluator.metadata.get('config', {}).get('synthetic', False)),
              'protocol_fingerprint': digest({k: v for k, v in cfg.items() if k not in {'seed', 'final_seed'}})}
    save_json(Path(output) / 'search_summary.json', result)
    if best:
        save_json(Path(output) / 'best_architecture.json', best['architecture'])
    return result
