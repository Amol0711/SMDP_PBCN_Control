from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .contracts import Protocol, integer, scalar

@dataclass(frozen=True)
class Plant:
    P: np.ndarray
    target: np.ndarray
    actions: tuple[tuple[int, int], ...]
    penalty: float
    budget: float

    @classmethod
    def from_protocol(cls, p: Protocol):
        c = p.config
        m = c['model']
        arr = np.array([m['P0'], m['P1']], dtype=np.float64)
        target = np.array(m['target_indicator'], dtype=np.float64)
        if arr.shape != (2, 4, 4) or np.any(arr < 0) or np.max(abs(arr.sum(2) - 1)) > 1e-12:
            raise ValueError('invalid plant')
        if target.shape != (4,) or not np.all(np.isin(target, [0, 1])):
            raise ValueError('invalid target')
        arr.setflags(write=False)
        target.setflags(write=False)
        return cls(arr, target, tuple((tuple(x) for x in c['actor']['action_order'])), float(c['objective']['penalty']), float(c['objective']['budget']))

def categorical(probabilities: np.ndarray, u: float) -> tuple[int, bool]:
    p = np.asarray(probabilities, dtype=np.float64)
    u = scalar(u, 'uniform')
    if p.ndim != 1 or len(p) == 0 or (not np.all(np.isfinite(p))) or np.any(p < 0) or (abs(float(p.sum()) - 1) > 1e-12):
        raise ValueError('invalid probability row')
    if not 0 <= u < 1:
        raise ValueError('uniform outside [0,1)')
    j = int(np.searchsorted(np.cumsum(p), u, side='right'))
    tail = j == len(p)
    if tail:
        if p[-1] <= 0 or 1 - float(p.sum()) > 1e-12:
            raise ArithmeticError('unresolved categorical tail')
        j = len(p) - 1
    return (j, tail)

@dataclass(frozen=True)
class Hold:
    state: int
    action: int
    input_value: int
    duration: int
    next_state: int
    reward: float
    target_count: int
    plant_uniforms: np.ndarray
    post_states: np.ndarray
    tail_count: int

def held_sample(model: Plant, state: int, action: int, uniforms: np.ndarray) -> Hold:
    state = integer(state, 'state', 0, 3)
    action = integer(action, 'action', 0, 9)
    u, t = model.actions[action]
    us = np.array(uniforms, dtype=np.float64, copy=True)
    if us.shape != (t,) or not np.all(np.isfinite(us)):
        raise ValueError('one draw for each complete plant step required')
    current = state
    post = []
    count = 0
    reward = 0.0
    tail = 0
    for draw in us:
        current, tf = categorical(model.P[u, current], float(draw))
        post.append(current)
        tail += int(tf)
        count += int(model.target[current])
        reward += float(model.target[current] - model.penalty * u)
    post = np.asarray(post, dtype=np.int8)
    post.setflags(write=False)
    us.setflags(write=False)
    return Hold(state, action, u, t, current, float(reward), count, us, post, tail)

def macro_arrays(model: Plant) -> dict[str, np.ndarray]:
    Q = np.zeros((4, 10, 4))
    O = np.zeros((4, 10))
    G = O.copy()
    T = O.copy()
    C = np.ones_like(O)
    for a, (u, t) in enumerate(model.actions):
        power = np.eye(4)
        occ = np.zeros(4)
        for _ in range(t):
            power = power @ model.P[u]
            occ += power @ model.target
        Q[:, a] = power
        O[:, a] = occ
        G[:, a] = occ - model.penalty * u * t
        T[:, a] = t
    return {'Q': Q, 'occupancy': O, 'reward': G, 'duration': T, 'expenditure': C}
