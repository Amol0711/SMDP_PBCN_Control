from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from .actor_geometry import require_domain, project_zero_sum_box, finite_array, positive_scalar
from .actor_scores import tabular_score

def sampled_row_step(row: ArrayLike, action: int, td_error: float, gain: float, temperature: float, bound: float=12.0):
    w = require_domain(row, bound)
    if w.ndim != 1 or not np.isfinite(td_error):
        raise ValueError('invalid sampled row or TD error')
    gain = positive_scalar(gain, 'gain')
    return project_zero_sum_box(w + gain * float(td_error) * tabular_score(w, action, temperature), bound)

def meanfield_step(theta: ArrayLike, rate_lagrangian_gradient: ArrayLike, gain: float=10.0, bound: float=12.0):
    w = require_domain(theta, bound)
    g = finite_array(rate_lagrangian_gradient, 'gradient')
    if w.shape != g.shape:
        raise ValueError('gradient shape mismatch')
    return project_zero_sum_box(w + positive_scalar(gain, 'gain') * g, bound)

def episodic_actor_step(theta: ArrayLike, accumulated_td_scores: ArrayLike, gain: float, bound: float=12.0):
    w = require_domain(theta, bound)
    g = finite_array(accumulated_td_scores, 'accumulated scores')
    gain = positive_scalar(gain, 'gain')
    if w.shape != g.shape:
        raise ValueError('episodic score shape mismatch')
    if np.all(g == 0):
        return w.copy()
    return meanfield_step(w, g, gain, bound)

def multiplier_step(multiplier: float, residual: float, gain: float, upper: float | None):
    if not np.isfinite(multiplier) or not np.isfinite(residual) or multiplier < 0:
        raise ValueError('invalid multiplier or residual')
    gain = positive_scalar(gain, 'gain')
    if upper is not None:
        upper = positive_scalar(upper, 'upper')
        if multiplier > upper:
            raise ValueError('multiplier above its declared cap')
    return float(np.clip(multiplier + gain * residual, 0.0, np.inf if upper is None else upper))
