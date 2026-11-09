"""Paired validation ranking diagnostics, including non-identifiable tie boundaries."""
from __future__ import annotations
import numpy as np
from scipy.stats import kendalltau, spearmanr


def ranking_metrics(early: np.ndarray, late: np.ndarray, k: int = 5, keep_fraction: float = 1/3) -> dict:
    early, late = np.asarray(early, dtype=float), np.asarray(late, dtype=float)
    if early.ndim != 1 or early.shape != late.shape or len(early) < 2 or not np.isfinite([early, late]).all():
        raise ValueError('Need at least two matched finite ranking observations')
    if not 0 < keep_fraction <= 1 or k < 1:
        raise ValueError('Need positive k and keep_fraction in (0,1]')
    n = len(early)
    k = min(k, n)
    keep = min(n, max(1, int(np.floor(n * keep_fraction))))
    a = np.argsort(-early, kind='stable')
    b = np.argsort(-late, kind='stable')
    def boundary_tied(scores: np.ndarray, order: np.ndarray, count: int) -> bool:
        return count < n and bool(scores[order[count-1]] == scores[order[count]])
    early_k_tied = boundary_tied(early, a, k)
    late_k_tied = boundary_tied(late, b, k)
    promotion_tied = boundary_tied(early, a, keep)
    top_late = set(b[:k])
    eliminated = top_late - set(a[:keep])
    constant = len(np.unique(early)) == 1 or len(np.unique(late)) == 1
    spearman = float('nan') if constant else float(spearmanr(early, late).statistic)
    kendall = float('nan') if constant else float(kendalltau(early, late).statistic)
    possible_winners = np.flatnonzero(early == early.max())
    regrets = late.max() - late[possible_winners]
    regret_min, regret_max = float(regrets.min()), float(regrets.max())
    overlap = len(set(a[:k]) & top_late) / k
    elimination = len(eliminated) / k
    return {'n': n, 'k': k, 'kept': keep,
            'ranking_informative': not constant,
            'spearman': spearman if np.isfinite(spearman) else None,
            'kendall': kendall if np.isfinite(kendall) else None,
            'top_k_overlap': None if early_k_tied or late_k_tied else overlap,
            'premature_elimination_rate': None if promotion_tied or late_k_tied else elimination,
            'selection_regret': regret_min if regret_min == regret_max else None,
            'selection_regret_range': [regret_min, regret_max],
            'incorrectly_rejected_indices': None if promotion_tied or late_k_tied else sorted(int(i) for i in eliminated),
            'early_top_k_boundary_tied': early_k_tied, 'late_top_k_boundary_tied': late_k_tied,
            'promotion_boundary_tied': promotion_tied,
            'deterministic_tie_break': {'policy': 'stable candidate index; descriptive only',
                'top_k_overlap': overlap, 'premature_elimination_rate': elimination,
                'selection_regret': float(late.max() - late[a[0]])},
            'early_ties': n - len(np.unique(early)), 'late_ties': n - len(np.unique(late))}
