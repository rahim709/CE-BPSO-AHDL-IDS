"""Ablation interfaces; future policies are not claimed as novel methods."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class FitnessEstimate:
    mean: float
    budget: int
    seed: int
    standard_error: float | None = None


class AllocationPolicy(Protocol):
    def next_budget(self, trajectory: list[dict], current_budget: int) -> int | None: ...


class PromotionPolicy(Protocol):
    def select(self, estimates: list[FitnessEstimate], count: int) -> list[int]: ...


class BestUpdatePolicy(Protocol):
    def accept(self, previous: FitnessEstimate | None, proposed: FitnessEstimate) -> bool: ...


class TrajectoryPredictor(Protocol):
    def predict(self, trajectory: list[dict], target_budget: int) -> tuple[float, float]: ...


class StrictImprovement:
    def accept(self, previous: FitnessEstimate | None, proposed: FitnessEstimate) -> bool:
        return previous is None or (previous.budget == proposed.budget and proposed.mean > previous.mean)


class RankPromotion:
    def select(self, estimates: list[FitnessEstimate], count: int) -> list[int]:
        return np.argsort([-e.mean for e in estimates], kind='stable')[:count].tolist()
