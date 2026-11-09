"""Shared validation-only evaluator and persistent compute accounting."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.optimization.encoding import BinaryCodec
from src.training.engine import train_to
from src.utils.core import digest, save_json


class BudgetExhausted(RuntimeError):
    pass


class TargetReached(BudgetExhausted):
    pass


class Evaluator:
    def __init__(self, cfg: dict, output: str | Path, codec: BinaryCodec) -> None:
        self.cfg, self.codec = cfg, codec
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.data = cfg['data']['processed']
        self.metadata = json.loads((Path(self.data) / 'metadata.json').read_text())
        self.seed = cfg.get('seed', 42)
        self.training = cfg.get('training', {})
        self.identity = digest({'config': cfg, 'data': self.metadata['fingerprint']})
        identity_path = self.output / 'search_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())['identity'] != self.identity:
            raise ValueError('Search output already belongs to a different config or dataset')
        save_json(identity_path, {'identity': self.identity})
        self.path = self.output / 'evaluations.json'
        self.events = json.loads(self.path.read_text()) if self.path.exists() else []
        if self.events and self.events[0]['identity'] != self.identity:
            raise ValueError('Search output already belongs to a different config or dataset')
        self.budget = cfg.get('compute', {})
        self.began = time.perf_counter()
        self.previous_wall = 0.
        wall_path = self.output / 'wall_state.json'
        if wall_path.exists():
            self.previous_wall = json.loads(wall_path.read_text())['seconds']
        self.cache_hits = 0
        device = self.training.get('device', 'auto')
        self.on_cuda = device.startswith('cuda') or (device == 'auto' and torch.cuda.is_available())
        self._recover_completed_epochs()

    def _recover_completed_epochs(self) -> None:
        """Recover a checkpoint committed before an event/trajectory write."""
        recovered = False
        for path in sorted((self.output / 'candidates').glob('*')):
            trajectory, checkpoint = path / 'trajectory.json', path / 'last.pt'
            if checkpoint.exists() and (not trajectory.exists() or checkpoint.stat().st_mtime_ns > trajectory.stat().st_mtime_ns):
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                save_json(trajectory, saved['rows'])
            if not trajectory.exists():
                continue
            rows = json.loads(trajectory.read_text())
            logged = max((e['completed_budget'] for e in self.events if e['candidate'] == path.name), default=0)
            new = [r for r in rows if r['epoch'] > logged]
            if not new:
                continue
            architecture = json.loads((path / 'architecture.json').read_text())
            seconds = sum(r['training_seconds'] + r['validation_seconds'] for r in new)
            gpu = sum(r['training_seconds'] + r['validation_seconds'] for r in new if r['device'].startswith('cuda'))
            # The committed epoch is an actual validation observation, eligible for legacy selection.
            self.events.append({'identity': self.identity, 'candidate': path.name, 'architecture': architecture,
                                'completed_budget': new[-1]['epoch'], 'requested_budget': new[-1]['epoch'],
                                'charged_epochs': len(new), 'wall_seconds': seconds, 'gpu_seconds': gpu,
                                'gpu_budget_seconds': gpu, 'fitness': new[-1]['macro_f1'], 'failure': None,
                                'parameters': new[-1].get('parameters'), 'recovered_unlogged_compute': True,
                                'recovery_wall_is_lower_bound': True})
            self.previous_wall += seconds
            recovered = True
        if recovered:
            self.persist()

    @property
    def charged_gpu_seconds(self) -> float:
        # Conservatively reserve complete CUDA evaluation wall, including failed work.
        return sum(e.get('gpu_budget_seconds', e['gpu_seconds']) for e in self.events)

    @property
    def used_epochs(self) -> int:
        return sum(e['charged_epochs'] for e in self.events)

    @property
    def wall_seconds(self) -> float:
        return self.previous_wall + time.perf_counter() - self.began

    def persist(self) -> None:
        save_json(self.path, self.events)
        save_json(self.output / 'wall_state.json', {'seconds': self.wall_seconds})

    @property
    def measured_gpu_seconds(self) -> float:
        return sum(e['gpu_seconds'] for e in self.events)

    def _check_target(self, score, budget):
        target = self.cfg.get('search', {}).get('stop_f1')
        if target is not None and budget >= self.cfg['search']['epochs'] and score >= target:
            raise TargetReached('Common-fidelity validation target reached')

    def available(self, epochs: int = 0) -> bool:
        return (self.used_epochs + epochs <= self.budget.get('max_epochs', float('inf')) and
                self.wall_seconds < self.budget.get('max_seconds', float('inf')) and
                (not self.on_cuda or (self.measured_gpu_seconds if self.budget.get('gpu_budget_mode') == 'execution' else self.charged_gpu_seconds) < self.budget.get('max_gpu_seconds', float('inf'))))

    def candidate_path(self, architecture: dict, seed: int) -> Path:
        key = digest({'architecture': architecture, 'seed': seed, 'training': self.training,
                      'data': self.metadata['fingerprint']})
        return self.output / 'candidates' / key

    def evaluate(self, bits: np.ndarray, budget: int, seed: int | None = None) -> float:
        seed = self.seed if seed is None else seed
        architecture = self.codec.decode(bits)
        path = self.candidate_path(architecture, seed)
        trajectory = path / 'trajectory.json'
        rows = json.loads(trajectory.read_text()) if trajectory.exists() else []
        cached = next((r for r in rows if r['epoch'] == budget), None)
        if cached is not None:
            if cached.get('macro_f1') is None:
                raise ValueError('Requested epoch was not validated; configure validation_epochs to cover search fidelities')
            self.cache_hits += 1
            self._check_target(cached['macro_f1'], budget)
            return cached['macro_f1']
        previous = max([r['epoch'] for r in rows], default=0)
        delta = budget - previous
        if delta <= 0:
            raise RuntimeError('Missing trajectory epoch in resumed candidate')
        if not self.available(delta):
            raise BudgetExhausted('Shared compute cap reached')
        began = time.perf_counter()
        remaining = self.budget.get('max_seconds', float('inf')) - self.wall_seconds
        device = self.training.get('device', 'auto')
        execution_budget = None
        if self.budget.get('gpu_budget_mode') == 'execution' and self.on_cuda:
            execution_budget = self.budget.get('max_gpu_seconds', float('inf')) - self.measured_gpu_seconds
        elif device.startswith('cuda') or (device == 'auto' and torch.cuda.is_available()):
            remaining = min(remaining, self.budget.get('max_gpu_seconds', float('inf')) -
                            self.charged_gpu_seconds)
        deadline = time.perf_counter() + remaining
        failure = None
        result = None
        try:
            kwargs = {'deadline': deadline}
            if self.cfg.get('runtime'):
                kwargs['runtime'] = self.cfg['runtime']
            if execution_budget is not None:
                kwargs['execution_budget'] = execution_budget
            result = train_to(architecture, self.data, path, budget, seed, self.training, **kwargs)
        except (TimeoutError, FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
            failure = str(exc)
        except BaseException as exc:
            failure = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            actual = json.loads(trajectory.read_text()) if trajectory.exists() else []
            completed = max([r['epoch'] for r in actual], default=previous)
            new_rows = [r for r in actual if r['epoch'] > previous]
            event = {'identity': self.identity, 'candidate': path.name, 'architecture': architecture, 'bits': bits.tolist(),
                     'seed': seed, 'requested_budget': budget, 'completed_budget': completed,
                     'charged_epochs': completed - previous, 'wall_seconds': time.perf_counter() - began,
                     'gpu_budget_seconds': time.perf_counter() - began if self.on_cuda else 0.,
                     'training_seconds': sum(r['training_seconds'] for r in new_rows),
                     'validation_seconds': sum(r['validation_seconds'] for r in new_rows),
                     'gpu_seconds': sum(r['training_seconds'] + r['validation_seconds'] for r in new_rows
                                        if r['device'].startswith('cuda')),
                     'peak_gpu_memory_bytes': max((r.get('peak_gpu_memory_bytes', 0) for r in new_rows), default=0),
                     'parameters': result.get('parameters') if result else None,
                     'fitness': result['macro_f1'] if result else None,
                     'validation_accuracy': result['metrics']['accuracy'] if result else None,
                     'predicted_class_count': result['metrics'].get('predicted_class_count') if result else None,
                     'collapsed_to_single_class': result['metrics'].get('collapsed_to_single_class') if result else None,
                     'majority_baseline_macro_f1': result.get('majority_baseline', {}).get('macro_f1') if result else None,
                     'search_wall_seconds': self.wall_seconds, 'failure': failure}
            self.events.append(event)
            self.persist()
        if failure:
            if 'time budget' in failure:
                raise BudgetExhausted(failure)
            return float('-inf')
        self._check_target(result['macro_f1'], budget)
        return result['macro_f1']

    def best(self) -> dict | None:
        valid = [e for e in self.events if e['fitness'] is not None and e['failure'] is None]
        if not valid:
            return None
        # Do not compare a 3-epoch number to a 30-epoch number.
        highest = max(e['completed_budget'] for e in valid)
        return max((e for e in valid if e['completed_budget'] == highest), key=lambda e: e['fitness'])
