"""Explicit normal/NMRI specialist routing and validation FPR calibration."""
import numpy as np


def route_specialist(base_prediction, scores, threshold, normal=0, target=1):
    base=np.asarray(base_prediction)
    scores=np.asarray(scores,dtype=float)
    if base.ndim!=1 or scores.shape!=base.shape or not np.isfinite(scores).all():
        raise ValueError('Need matched finite one-dimensional predictions and scores')
    if normal==target or not np.isfinite(threshold):
        raise ValueError('Distinct normal/target classes and finite threshold required')
    out=base.copy()
    eligible=(base==normal)|(base==target)
    out[eligible]=np.where(scores[eligible]>threshold,target,normal)
    return out


def calibrate_specialist(base_prediction, scores, validation_labels, max_fpr=.03, normal=0, target=1):
    base=np.asarray(base_prediction);scores=np.asarray(scores,dtype=float);y=np.asarray(validation_labels)
    route_specialist(base,scores,0.,normal,target)
    if y.shape!=base.shape or not 0<=max_fpr<1 or not np.any(y==normal):
        raise ValueError('Need matched labels, normal support and a valid FPR cap')
    eligible=(base==normal)|(base==target);normal_rows=y==normal
    fixed=int(np.sum(normal_rows & ~eligible))
    allowance=int(np.floor(max_fpr*np.sum(normal_rows)))-fixed
    if allowance<0:
        raise ValueError('Other-class predictions already exceed the FPR cap')
    values=np.sort(scores[normal_rows & eligible])[::-1]
    if not len(values):
        raise ValueError('No eligible normal calibration records')
    threshold=float(values[min(allowance,len(values)-1)])
    return {'threshold':threshold,'max_validation_fpr':max_fpr,
            'fixed_normal_false_positives':fixed,'normal_validation_support':int(normal_rows.sum()),
            'rule':'target iff specialist score is strictly greater than threshold; ties remain normal'}
