from __future__ import annotations
from dataclasses import dataclass
import itertools, json
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Model:
    plant: np.ndarray
    actions: np.ndarray
    transitions: np.ndarray
    occupancy: np.ndarray
    rewards: np.ndarray
    duration: np.ndarray
    exposure: np.ndarray
    target: np.ndarray
    budget: float
    cost: float

def build(config_path: Path | None=None) -> Model:
    c = json.loads((config_path or Path(__file__).with_name('config.json')).read_text())
    p = np.array([c['model']['P0'], c['model']['P1']], float)
    if np.min(p) < 0 or np.max(abs(p.sum(axis=2) - 1)) > 1e-12:
        raise ValueError('invalid plant')
    target = np.array(c['model']['target_indicator'], float)
    actions = np.array(list(itertools.product(c['model']['inputs'], c['model']['durations'])), int)
    n = 4
    m = len(actions)
    Q = np.empty((n, m, n))
    O = np.empty((n, m))
    for a, (u, tau) in enumerate(actions):
        power = np.eye(n)
        o = np.zeros(n)
        for _ in range(tau):
            power = power @ p[u]
            o += power @ target
        Q[:, a, :] = power
        O[:, a] = o
    T = np.broadcast_to(actions[:, 1], (n, m)).astype(float).copy()
    E = T * actions[:, 0]
    cost = c['objectives']['control_cost']
    return Model(p, actions, Q, O, O - cost * E, T, E, target, c['objectives']['budget'], cost)

def classes(P: np.ndarray) -> dict:
    reach = (P > 0) | np.eye(len(P), dtype=bool)
    for k in range(len(P)):
        reach |= reach[:, k, None] & reach[None, k, :]
    groups = []
    remaining = set(range(len(P)))
    while remaining:
        i = min(remaining)
        group = [j for j in remaining if reach[i, j] and reach[j, i]]
        groups.append(group)
        remaining.difference_update(group)
    recurrent = [g for g in groups if not any((P[i, j] > 0 for i in g for j in range(len(P)) if j not in g))]
    return {'communicating_classes': groups, 'recurrent_classes': recurrent, 'irreducible': len(groups) == 1, 'unichain': len(recurrent) == 1, 'transient_states': [i for g in groups if g not in recurrent for i in g]}

def evaluate(model: Model, pi: np.ndarray) -> dict:
    pi = np.asarray(pi, float)
    if pi.shape != (4, 10) or np.min(pi) < 0 or np.max(abs(pi.sum(1) - 1)) > 1e-12:
        raise ValueError('invalid policy')
    P = np.einsum('sa,san->sn', pi, model.transitions)
    cl = classes(P)
    if not cl['unichain']:
        raise ValueError('multichain policy requires initial-law treatment')
    A = P.T - np.eye(4)
    A[-1] = 1
    rhs = np.array([0.0, 0, 0, 1])
    d = np.linalg.solve(A, rhs)
    ds = np.linalg.lstsq(np.vstack((P.T - np.eye(4), np.ones((1, 4)))), np.r_[np.zeros(4), 1.0], rcond=None)[0]
    err = max(float(np.max(abs(d @ P - d))), float(abs(d.sum() - 1)), float(np.max(abs(d - ds))))
    if min(d) < -1e-14 or err > 1e-10:
        raise ArithmeticError('invariant law failed')
    weight = d[:, None] * pi
    tb = float(np.sum(weight * model.duration))
    y = weight / tb
    return {'occupancy': float(np.sum(y * model.occupancy)), 'reward': float(np.sum(y * model.rewards)), 'rate': float(y.sum()), 'mean_duration': tb, 'exposure_rate': float(np.sum(y * model.exposure)), 'rate_residual': float(y.sum() - model.budget), 'feasible': bool(y.sum() <= model.budget + 1e-10), 'stationary': d.tolist(), 'stationary_check_abs': err, 'occupation_measure': y.tolist(), 'policy': pi.tolist(), 'chain': cl}

def deterministic(indices) -> np.ndarray:
    idx = np.asarray(indices, int)
    if idx.shape != (4,) or min(idx) < 0 or max(idx) >= 10:
        raise ValueError('illegal selector')
    return np.eye(10)[idx]
