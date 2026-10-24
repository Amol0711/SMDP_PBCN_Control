from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
PROJECTION_ID = 'zero_sum_box_euclidean_v1'
FEASIBILITY_ATOL = 1e-10
IDEMPOTENCE_ATOL = 1e-10
OPTIMALITY_ATOL = 1e-09

def finite_array(value: ArrayLike, name: str, *, min_ndim: int=1) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.dtype.kind not in 'biuf':
        raise ValueError(f'{name} must be real numeric data')
    a = np.array(raw, dtype=np.float64, copy=True)
    if a.ndim < min_ndim or not a.size or (not np.all(np.isfinite(a))):
        raise ValueError(f'{name} must be nonempty and finite with ndim>={min_ndim}')
    return a

def positive_scalar(value: float, name: str) -> float:
    if not np.isscalar(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a positive finite scalar')
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f'{name} must be a positive finite scalar')
    return v

@dataclass(frozen=True)
class ProjectionCertificate:
    equality_multiplier: NDArray[np.float64]
    row_sum_error: float
    bound_violation: float
    normal_cone_error: float
    lower_multiplier: NDArray[np.float64]
    upper_multiplier: NDArray[np.float64]
    stationarity_error: float
    complementarity_error: float

def domain_errors(theta: ArrayLike, bound: float=12.0) -> dict[str, float]:
    a = finite_array(theta, 'theta')
    b = positive_scalar(bound, 'bound')
    return {'row_sum_error': float(np.max(np.abs(a.sum(axis=-1)))), 'bound_violation': float(max(0.0, np.max(np.abs(a)) - b))}

def require_domain(theta: ArrayLike, bound: float=12.0) -> NDArray[np.float64]:
    a = finite_array(theta, 'theta')
    e = domain_errors(a, bound)
    if max(e.values()) > FEASIBILITY_ATOL:
        raise ValueError(f'theta is outside declared zero-sum box: {e}')
    return a

def projection_certificate(v: ArrayLike, x: ArrayLike, equality_multiplier: ArrayLike, bound: float=12.0) -> ProjectionCertificate:
    v = finite_array(v, 'v')
    x = finite_array(x, 'x')
    b = positive_scalar(bound, 'bound')
    lam = np.asarray(equality_multiplier, dtype=np.float64)
    if v.shape != x.shape or lam.shape != v.shape[:-1] or (not np.all(np.isfinite(lam))):
        raise ValueError('projection certificate shape/multiplier mismatch')
    q = x - v + lam[..., None]
    lower = np.where(x <= -b + 1e-12, np.maximum(q, 0.0), 0.0)
    upper = np.where(x >= b - 1e-12, np.maximum(-q, 0.0), 0.0)
    err = domain_errors(x, b)
    stationarity = float(np.max(np.abs(q + upper - lower)))
    complementarity = float(max(np.max(np.abs(lower * (x + b))), np.max(np.abs(upper * (b - x)))))
    normal = float(np.max(np.abs(x - np.clip(v - lam[..., None], -b, b))))
    return ProjectionCertificate(lam.copy(), err['row_sum_error'], err['bound_violation'], normal, lower, upper, stationarity, complementarity)

def project_zero_sum_box(v: ArrayLike, bound: float=12.0, *, return_certificate: bool=False):
    a = finite_array(v, 'v')
    b = positive_scalar(bound, 'bound')
    with np.errstate(over='raise', invalid='raise'):
        try:
            center = a.mean(axis=-1, keepdims=True)
            z = a - center
            lo = np.min(z, axis=-1, keepdims=True) - b
            hi = np.max(z, axis=-1, keepdims=True) + b
        except FloatingPointError as e:
            raise ArithmeticError('projection input exceeds safe floating-point scale') from e
    for _ in range(100):
        mid = lo / 2.0 + hi / 2.0
        f = np.clip(z - mid, -b, b).sum(axis=-1, keepdims=True)
        lo = np.where(f > 0, mid, lo)
        hi = np.where(f <= 0, mid, hi)
    shift = lo / 2.0 + hi / 2.0
    for _ in range(2):
        y = z - shift
        free = (y > -b) & (y < b)
        count = free.sum(axis=-1, keepdims=True)
        numerator = np.where(free, z, 0.0).sum(axis=-1, keepdims=True) + b * ((y >= b).sum(axis=-1, keepdims=True) - (y <= -b).sum(axis=-1, keepdims=True))
        refined = np.divide(numerator, count, out=shift.copy(), where=count > 0)
        shift = np.where(count > 0, refined, shift)
    x = np.clip(z - shift, -b, b)
    lam = (shift + center)[..., 0]
    cert = projection_certificate(a, x, lam, b)
    if cert.row_sum_error > FEASIBILITY_ATOL or cert.bound_violation > FEASIBILITY_ATOL or max(cert.normal_cone_error, cert.stationarity_error, cert.complementarity_error) > OPTIMALITY_ATOL:
        raise ArithmeticError('projection certificate failed; do not accept this step')
    return (x, cert) if return_certificate else x

def zero_sum_basis(n_actions: int) -> NDArray[np.float64]:
    if not isinstance(n_actions, int) or isinstance(n_actions, bool) or n_actions < 1:
        raise ValueError('n_actions must be a positive integer')
    q = np.zeros((n_actions, n_actions - 1), dtype=np.float64)
    for j in range(n_actions - 1):
        den = np.sqrt((j + 1) * (j + 2))
        q[:j + 1, j] = 1 / den
        q[j + 1, j] = -(j + 1) / den
    return q

def actor_mapping(theta: ArrayLike, gradient: ArrayLike, bound: float=12.0, step: float=1.0) -> NDArray[np.float64]:
    w = require_domain(theta, bound)
    g = finite_array(gradient, 'gradient')
    h = positive_scalar(step, 'step')
    if w.shape != g.shape:
        raise ValueError('actor/gradient shape mismatch')
    return (project_zero_sum_box(w + h * g, bound) - w) / h
