"""Optional, standard class weighting fitted only to training targets."""
from __future__ import annotations
import numpy as np
import torch


def class_weights(training_targets: np.ndarray, classes: int, strategy: str = 'none') -> torch.Tensor | None:
    if strategy not in {'none', 'balanced', 'sqrt_balanced'}:
        raise ValueError('class_weighting must be none, balanced or sqrt_balanced')
    if strategy == 'none':
        return None
    y = np.asarray(training_targets)
    if y.ndim != 1 or len(y) == 0 or classes < 2 or not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= classes).any():
        raise ValueError('Training targets must be nonempty integer class indices')
    counts = np.bincount(y, minlength=classes)
    if (counts == 0).any():
        raise ValueError('Cannot estimate class weights for classes absent from training')
    weights = len(y) / (classes * counts.astype(float))
    if strategy == 'sqrt_balanced':
        weights = np.sqrt(weights)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
