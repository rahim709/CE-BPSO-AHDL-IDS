"""Architecture-disjoint, group split-conformal learning-curve prediction.

Nominal coverage is a design target; held-out coverage must be measured.
Training seeds are grouped under architecture, never treated as independent
calibration architectures. Features only consume rows at/before the query rung.
"""
from __future__ import annotations
import math
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.optimization.encoding import BinaryCodec
from src.utils.core import digest


def curve_features(architecture, trajectory, budget, codec=None):
    codec = codec or BinaryCodec('paper')
    observed = sorted((r for r in trajectory if r['epoch'] <= budget), key=lambda r: r['epoch'])
    rows = [r for r in observed if r.get('macro_f1') is not None]
    if not rows or rows[-1]['epoch'] != budget:
        raise ValueError('Query needs a completed current rung')
    y = np.asarray([r['macro_f1'] for r in rows])
    epochs = np.asarray([r['epoch'] for r in rows])
    slope = float(np.polyfit(epochs, y, 1)[0]) if len(rows) > 1 else 0.
    seconds = np.mean([r['training_seconds'] + r['validation_seconds'] for r in observed])
    return np.r_[codec.encode(architecture), np.log1p(rows[-1].get('parameters', 0)),
                 budget, y[-1], y.mean(), y.max(), y.min(), y.std(), slope,
                 rows[-1]['loss'], np.log1p(seconds)].astype(float)


def raw_prediction(forest, features):
    predictions = np.stack([tree.predict(features) for tree in forest.estimators_])
    return predictions.mean(axis=0), np.maximum(.02, predictions.std(axis=0, ddof=1))


class CurvePredictor:
    def __init__(self, alpha=.2, random_state=718):
        if not 0 < alpha < 1:
            raise ValueError('alpha must be in (0,1)')
        self.alpha, self.random_state = alpha, random_state
        self.models = {}
        self.archive_architectures = set()
        self.context = None

    def fit(self, records, context):
        """records contain features/target, budget, architecture_id and partition.

        Partitions must be assigned to architecture IDs before examining targets.
        """
        began = time.perf_counter()
        memberships = {}
        for row in records:
            memberships.setdefault(row['architecture_id'], set()).add(row['partition'])
        if any(len(parts) != 1 for parts in memberships.values()):
            raise ValueError('Architecture crosses fit/calibration/test partitions')
        self.archive_architectures = set(memberships)
        self.context = context
        for budget in sorted({r['budget'] for r in records}):
            fit = [r for r in records if r['budget'] == budget and r['partition'] == 'fit']
            cal = [r for r in records if r['budget'] == budget and r['partition'] == 'calibration']
            if len({r['architecture_id'] for r in fit}) < 4 or len({r['architecture_id'] for r in cal}) < 4:
                raise ValueError('Need at least four fit and four calibration architectures per rung')
            forest = RandomForestRegressor(n_estimators=64, min_samples_leaf=2, random_state=self.random_state + budget, n_jobs=1)
            forest.fit(np.stack([r['features'] for r in fit]), [r['target'] for r in fit])
            mean, scale = raw_prediction(forest, np.stack([r['features'] for r in cal]))
            grouped = {}
            for row, residual in zip(cal, np.abs(np.asarray([r['target'] for r in cal]) - mean) / scale):
                grouped[row['architecture_id']] = max(grouped.get(row['architecture_id'], 0.), float(residual))
            residuals = np.sort(list(grouped.values()))
            k = math.ceil((len(residuals) + 1) * (1 - self.alpha))
            q = float(residuals[k - 1]) if k <= len(residuals) else float('inf')
            self.models[budget] = {'forest': forest, 'q': q, 'calibration_architectures': len(residuals),
                                   'fit_architectures': len({r['architecture_id'] for r in fit})}
        self.fit_wall_seconds = time.perf_counter() - began
        return self

    def predict_features(self, features, budget):
        model = self.models[budget]
        mean, scale = raw_prediction(model['forest'], np.atleast_2d(features))
        radius = np.minimum(1., model['q'] * scale)
        low, high = np.maximum(0., mean - radius), np.minimum(1., mean + radius)
        return mean, radius, low, high

    def predict(self, architecture, trajectory, budget, context, codec=None):
        if context != self.context:
            raise ValueError('Predictor dataset/training/space context differs')
        if digest(architecture) in self.archive_architectures:
            raise ValueError('Search architecture overlaps predictor development archive')
        features = curve_features(architecture, trajectory, budget, codec)
        if getattr(self, 'time_invariant', False):
            features[-1] = 0.
        return self.predict_features(features, budget)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(Path(path))
