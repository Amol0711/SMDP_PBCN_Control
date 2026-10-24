from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
INPUTS = tuple(product((0, 1), repeat=2))
DURATIONS = (1, 2, 3, 4, 5)
TARGET = (1, 1, 1, 1, 1, 1, 0, 1, 1)

def state_bits(index: int) -> tuple[int, ...]:
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool) or (not 0 <= index < 512):
        raise ValueError('state index must be an integer from 0 to 511')
    return tuple((int(index) >> j & 1 for j in range(8, -1, -1)))

def state_index(bits) -> int:
    x = tuple(bits)
    if len(x) != 9 or any((v not in (0, 1) for v in x)):
        raise ValueError('state must contain nine binary entries')
    return sum((int(v) << 8 - j for j, v in enumerate(x)))

def transition_distribution(index: int, u: tuple[int, int]) -> dict[int, float]:
    x = state_bits(index)
    if tuple(u) not in INPUTS:
        raise ValueError('input must contain two binary entries')
    u1, u2 = map(int, u)
    base = [(1 - x[6]) * x[2], x[0], 1 - u1, x[4] * x[5], 0, x[0], (1 - x[3]) * (1 - x[7]), int(bool(x[3] or x[4] or x[8])), 0]
    out = {}
    for b5, p5 in (((1 - u1) * x[1] * u2, 0.7), (x[4], 0.3)):
        for b9, p9 in (((1 - u1) * int(bool(x[4] or u2)), 0.6), (x[8], 0.4)):
            nxt = base.copy()
            nxt[4] = b5
            nxt[8] = b9
            j = state_index(nxt)
            out[j] = out.get(j, 0.0) + p5 * p9
    return out

@dataclass(frozen=True)
class LactoseModel:
    states: np.ndarray
    actions: np.ndarray
    one_step: np.ndarray
    transitions: np.ndarray
    reward: np.ndarray
    occupancy: np.ndarray
    exposure: np.ndarray
    duration: np.ndarray
    expenditure: np.ndarray
    target_index: int

    def arrays(self) -> dict:
        return {k: getattr(self, k) for k in ('states', 'actions', 'one_step', 'transitions', 'reward', 'occupancy', 'exposure', 'duration', 'expenditure')}

def build_model() -> LactoseModel:
    all_dist = {(s, u): transition_distribution(s, u) for s in range(512) for u in INPUTS}
    states = np.array(sorted({j for d in all_dist.values() for j in d}), dtype=np.int64)
    rmap = {int(s): i for i, s in enumerate(states)}
    n = len(states)
    one = np.zeros((4, n, n), dtype=np.float64)
    for ui, u in enumerate(INPUTS):
        for i, s in enumerate(states):
            for j, p in all_dist[int(s), u].items():
                one[ui, i, rmap[j]] += p
    ti = rmap[state_index(TARGET)]
    target = np.zeros(n)
    target[ti] = 1.0
    actions = []
    kernels = []
    occ = []
    exp = []
    for ui, u in enumerate(INPUTS):
        power = np.eye(n)
        acc = np.zeros(n)
        for tau in DURATIONS:
            power = power @ one[ui]
            acc = acc + power @ target
            actions.append((*u, tau))
            kernels.append(power.copy())
            occ.append(acc.copy())
            exp.append(np.full(n, tau * sum(u), dtype=np.float64))
    action = np.array(actions, dtype=np.int64)
    O = np.stack(occ, axis=1)
    E = np.stack(exp, axis=1)
    m = LactoseModel(states, action, one, np.stack(kernels, axis=1), O - 0.01 * E, O, E, action[:, 2].astype(np.float64), np.ones(20), ti)
    for arr in m.arrays().values():
        arr.setflags(write=False)
    return m

def strict_feasibility_witness(model: LactoseModel) -> tuple[np.ndarray, dict]:
    row = np.where(model.duration == 5, 4.0, -1.0)
    w = np.broadcast_to(row, model.reward.shape).copy()
    mean_duration = float((5 * np.exp(5) + 10) / (np.exp(5) + 4))
    return (w, {'construction': 'w_(u,5)=4; w_(u,tau<5)=-1, every state/input', 'mean_duration_closed_form': '(5*exp(5)+10)/(exp(5)+4)', 'mean_duration': mean_duration, 'update_rate': 1 / mean_duration, 'budget': 0.25, 'strict_margin': 0.25 - 1 / mean_duration, 'max_abs_coordinate': 4.0, 'row_sum': 0.0, 'strict_feasibility_reason': 'mean_duration>4 iff exp(5)>6; the witness lies strictly inside the B=12 box'})
