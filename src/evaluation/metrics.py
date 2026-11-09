"""Fixed-label metrics including multiclass and normal-versus-attack errors."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y: np.ndarray, prediction: np.ndarray, classes: list[str], normal_label: str | None = None) -> dict:
    labels = np.arange(len(classes))
    p, r, f, support = precision_recall_fscore_support(y, prediction, labels=labels, zero_division=0)
    cm = confusion_matrix(y, prediction, labels=labels)
    tp = cm.diagonal()
    fp, fn = cm.sum(axis=0) - tp, cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    result = {'accuracy': float(accuracy_score(y, prediction)), 'macro_precision': float(p.mean()),
              'macro_recall': float(r.mean()), 'macro_f1': float(f.mean()),
              'weighted_precision': float(np.average(p, weights=support)),
              'weighted_recall': float(np.average(r, weights=support)),
              'weighted_f1': float(np.average(f, weights=support)), 'confusion_matrix': cm.tolist(),
              'predicted_class_count': int(np.count_nonzero(cm.sum(axis=0))),
              'collapsed_to_single_class': bool(np.count_nonzero(cm.sum(axis=0)) == 1),
              'prediction_counts': {c: int(cm[:, i].sum()) for i, c in enumerate(classes)},
              'zero_recall_classes': [c for i, c in enumerate(classes) if support[i] > 0 and r[i] == 0],
              'per_class': {c: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]),
                               'support': int(support[i]), 'tp': int(tp[i]), 'fp': int(fp[i]), 'fn': int(fn[i]),
                               'tn': int(tn[i]), 'fpr': float(fp[i] / max(1, fp[i] + tn[i])),
                               'fnr': float(fn[i] / max(1, fn[i] + tp[i]))} for i, c in enumerate(classes)}}
    if normal_label is not None:
        if normal_label not in classes:
            raise ValueError(f'Normal label {normal_label!r} not in {classes}')
        index = classes.index(normal_label)
        normal = y == index
        predicted_normal = prediction == index
        result['attack_detection'] = {'false_positive': int((normal & ~predicted_normal).sum()),
                                      'false_negative': int((~normal & predicted_normal).sum()),
                                      'fpr': float((normal & ~predicted_normal).sum() / max(1, normal.sum())),
                                      'fnr': float((~normal & predicted_normal).sum() / max(1, (~normal).sum()))}
    return result


def majority_baseline(training_targets: np.ndarray, targets: np.ndarray, classes: list[str]) -> dict:
    """Choose the majority label using training only, then measure it on a partition."""
    index = int(np.argmax(np.bincount(training_targets, minlength=len(classes))))
    metrics = classification_metrics(targets, np.full(len(targets), index), classes)
    return {'training_majority_class': classes[index], 'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1']}
