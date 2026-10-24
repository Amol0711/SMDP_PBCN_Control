from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from .actor_geometry import finite_array, positive_scalar

def softmax_rows(logits: ArrayLike, temperature: float=1.0):
    w = finite_array(logits, 'logits')
    temp = positive_scalar(temperature, 'temperature')
    z = (w - w.max(axis=-1, keepdims=True)) / temp
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
    if not np.all(np.isfinite(p)):
        raise ArithmeticError('nonfinite softmax')
    return p

def tabular_score(logits: ArrayLike, action: int, temperature: float=1.0):
    row = finite_array(logits, 'logits')
    if row.ndim != 1:
        raise ValueError('tabular_score expects one row')
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or (not 0 <= action < row.size):
        raise ValueError('illegal action index')
    p = softmax_rows(row, temperature)
    score = -p
    score[action] += 1.0
    return score / positive_scalar(temperature, 'temperature')

def feature_softmax_score(weights: ArrayLike, features: ArrayLike, action: int):
    w = finite_array(weights, 'weights', min_ndim=2)
    x = finite_array(features, 'features')
    if w.ndim != 2 or x.ndim != 1 or w.shape[1] != len(x):
        raise ValueError('feature dimensions disagree')
    return tabular_score(w @ x, action)[:, None] * x[None, :]

def clipped_logistic_audit(weights: ArrayLike, features: ArrayLike, action: int, clip: float=30.0) -> dict:
    w = finite_array(weights, 'weights')
    x = finite_array(features, 'features')
    if w.ndim != 1 or x.shape != w.shape or action not in (0, 1):
        raise ValueError('invalid logistic inputs')
    c = positive_scalar(clip, 'clip')
    z = float(w @ x)
    p = float(1.0 / (1.0 + np.exp(-np.clip(z, -c, c))))
    surrogate = (action - p) * x
    kink = abs(z) == c
    saturated = abs(z) > c
    return {'preclip_logit': z, 'probability': p, 'saturated': saturated, 'at_clip_kink': kink, 'inherited_surrogate_score': surrogate, 'true_composite_score': None if kink else np.zeros_like(x) if saturated else surrogate.copy(), 'status': 'nondifferentiable_at_kink' if kink else 'surrogate_mismatch_possible' if saturated else 'score_matches_inside_clip', 'protocol_changed': False}
