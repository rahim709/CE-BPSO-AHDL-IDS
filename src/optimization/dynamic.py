"""Eager, dynamically allocated UA-AMF; calibrated bounds screen continuations."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from src.optimization.bpso import BPSO, SwarmSettings
from src.optimization.curve_predictor import CurvePredictor
from src.optimization.evaluator import BudgetExhausted
from src.utils.core import digest, save_json


def dynamic_search(evaluator, codec, cfg):
    search = cfg['search']
    budgets = search.get('budgets', [3,10,30])
    if budgets != sorted(set(budgets)) or budgets[-1] != search['epochs']:
        raise ValueError('Invalid dynamic rung ladder')
    started = time.perf_counter()
    path = Path(search['predictor_bundle'])
    if search.get('predictor_sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != search['predictor_sha256']:
        raise ValueError('Predictor bundle changed after protocol freeze')
    predictor = CurvePredictor.load(path)
    context = digest({'data_fingerprint':evaluator.metadata['fingerprint'],'training':cfg['training'],'space':cfg.get('space','paper')})
    if context != predictor.context:
        raise ValueError('Dynamic predictor context differs')
    settings = SwarmSettings(**{k:v for k,v in search.items() if k in SwarmSettings.__dataclass_fields__})
    swarm = BPSO(codec.length,settings,cfg.get('seed',42))
    beta, weight = search.get('uncertainty_beta',.1), search.get('cost_weight',.05)
    records, finalists = [], []
    prediction_time = 0.
    incumbent = -np.inf
    try:
        for generation in range(settings.generations):
            if not evaluator.available():
                raise BudgetExhausted('Dynamic resource cap reached')
            unique = {}
            for i,bits in enumerate(swarm.position):
                while digest(codec.decode(bits)) in predictor.archive_architectures:
                    bits = codec.sample(swarm.rng)
                    swarm.position[i] = bits
                unique.setdefault(digest(codec.decode(bits)),{'bits':bits.copy(),'particles':[]})['particles'].append(i)
            untested, active = list(unique.values()), []
            fitness = np.full(settings.population,-np.inf)
            probes, promotions_since_probe = 0, 0
            while untested or active:
                if not evaluator.available():
                    raise BudgetExhausted('Dynamic resource cap reached')
                if untested and (probes < search.get('initial_probes',2) or not active or promotions_since_probe >= search.get('probe_interval',2)):
                    candidate = untested.pop(0)
                    budget = budgets[0]
                    probes += 1
                    promotions_since_probe = 0
                else:
                    kept = []
                    for candidate in active:
                        if np.isfinite(incumbent) and candidate['upper'] < incumbent + search.get('min_improvement',.01):
                            records.append({'generation':generation,'architecture':codec.decode(candidate['bits']), 'budget':candidate['budget'],
                                            'action':'prune_upper_bound','upper':candidate['upper'],'incumbent':incumbent})
                        else:
                            kept.append(candidate)
                    active = kept
                    if not active:
                        continue
                    median_cost = max(1e-8,float(np.median([c['cost'] for c in active])))
                    for c in active:
                        c['priority'] = c['mean'] + beta*c['radius'] - weight*c['cost']/median_cost
                    chosen = max(range(len(active)),key=lambda i:(active[i]['priority'],-i))
                    candidate = active.pop(chosen)
                    budget = budgets[budgets.index(candidate['budget'])+1]
                    promotions_since_probe += 1
                    records.append({'generation':generation,'architecture':codec.decode(candidate['bits']),'budget':candidate['budget'],
                                    'next_budget':budget,'action':'promote','prediction':candidate['mean'],'radius':candidate['radius'],
                                    'upper':candidate['upper'],'expected_remaining_seconds':candidate['cost'],'priority':candidate['priority']})
                score = evaluator.evaluate(candidate['bits'],budget)
                if not np.isfinite(score):
                    continue
                architecture = codec.decode(candidate['bits'])
                if budget == budgets[-1]:
                    incumbent = max(incumbent,score)
                    fitness[candidate['particles']] = score
                    finalists.append({'architecture':architecture,'fitness':score,'completed_budget':budget,
                                      'replicate_seeds':[evaluator.seed],'observed_seed_fitness':[score]})
                else:
                    trajectory = json.loads((evaluator.candidate_path(architecture,evaluator.seed)/'trajectory.json').read_text())
                    begin = time.perf_counter()
                    mean,radius,lower,upper = predictor.predict(architecture,trajectory,budget,context,codec)
                    prediction_time += time.perf_counter()-begin
                    observed = [r for r in trajectory if r['epoch']<=budget]
                    train_cost = np.mean([r['training_seconds'] for r in observed])*(budgets[-1]-budget)
                    validations = [r['validation_seconds'] for r in observed if r.get('macro_f1') is not None]
                    checks = cfg.get('runtime',{}).get('validation_epochs')
                    future_checks = sum(budget < e <= budgets[-1] for e in set(checks + [budgets[-1]])) if checks else budgets[-1]-budget
                    cost = train_cost + (np.mean(validations) if validations else 0.)*future_checks
                    candidate.update(budget=budget,mean=float(mean[0]),radius=float(radius[0]),upper=float(upper[0]),cost=float(cost))
                    active.append(candidate)
            swarm.tell(fitness)
            swarm.step()
            n = int(np.ceil(settings.population*search.get('restart_fraction',.2)))
            if n:
                idx = np.argsort(swarm.personal_fitness,kind='stable')[:n]
                swarm.position[idx] = swarm.rng.integers(0,2,(n,codec.length),dtype=np.int8)
                swarm.personal_position[idx] = swarm.position[idx]
                swarm.personal_fitness[idx] = -np.inf
    finally:
        save_json(evaluator.output/'dynamic_decisions.json',records)
        save_json(evaluator.output/'adaptive_state.json',{'finalists':finalists,'prediction_seconds':prediction_time,
                  'scheduler_inclusive_seconds_this_invocation':time.perf_counter()-started})
    return max(finalists,key=lambda r:r['fitness']) if finalists else None
