"""Maximizing binary PSO, with Eq.19 random velocity replacement."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SwarmSettings:
    population: int = 20
    generations: int = 30
    w_max: float = .8
    w_min: float = .4
    v_max: float = 10.
    v_min: float = -10.
    c1: float = 1.5
    c2: float = 1.5


class BPSO:
    def __init__(self, dimensions: int, settings: SwarmSettings, seed: int) -> None:
        if dimensions < 1 or settings.population < 1 or settings.generations < 1 or settings.v_min >= settings.v_max:
            raise ValueError('Invalid swarm settings')
        self.settings = settings
        self.rng = np.random.default_rng(seed)
        self.position = self.rng.integers(0, 2, (settings.population, dimensions), dtype=np.int8)
        self.velocity = self.rng.uniform(settings.v_min, settings.v_max, self.position.shape)
        self.personal_position = self.position.copy()
        self.personal_fitness = np.full(settings.population, -np.inf)
        self.best_position = self.position[0].copy()
        self.best_fitness = -np.inf
        self.generation = 0
        self.history: list[dict] = []

    def tell(self, fitness: np.ndarray) -> None:
        fitness = np.asarray(fitness, dtype=float)
        if fitness.shape != self.personal_fitness.shape or np.isnan(fitness).any():
            raise ValueError('Expected one non-NaN fitness per particle')
        improve = fitness > self.personal_fitness
        self.personal_position[improve] = self.position[improve]
        self.personal_fitness[improve] = fitness[improve]
        best = int(np.argmax(self.personal_fitness))
        if self.personal_fitness[best] > self.best_fitness:
            self.best_fitness = float(self.personal_fitness[best])
            self.best_position = self.personal_position[best].copy()
        self.history.append({'generation': self.generation, 'best_fitness': self.best_fitness,
                             'fitness': fitness.tolist(), 'personal_fitness': self.personal_fitness.tolist()})

    def step(self) -> None:
        s = self.settings
        w = s.w_max - (s.w_max - s.w_min) * self.generation / s.generations
        r1, r2, r3, r = self.rng.random((4, *self.position.shape))
        v = w * self.velocity + s.c1 * r1 * (self.personal_position - self.position) + s.c2 * r2 * (self.best_position - self.position)
        self.velocity = np.where((v > s.v_max) | (v < s.v_min), r3, v)
        probability = 1 / (1 + np.exp(-np.clip(self.velocity, -700, 700)))
        self.position = (r <= probability).astype(np.int8)
        self.generation += 1

    def state_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != 'rng'} | {'rng_state': self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        state = dict(state)
        rng = state.pop('rng_state')
        self.__dict__.update(state)
        self.rng.bit_generator.state = rng
