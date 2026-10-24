from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix, eye
from ..lactose_meanfield.model import transition_distribution, state_index, TARGET, INPUTS
TARGET_ID = state_index(TARGET)
ACTIONS = np.array([(u[0], u[1], t) for u in INPUTS for t in range(1, 6)], dtype=np.int16)

def branch_table():
    out = np.zeros((512, 4, 2, 2), dtype=np.int16)
    for s in range(512):
        x = [s >> 8 - i & 1 for i in range(9)]
        for ui, (u1, u2) in enumerate(INPUTS):
            for b5 in range(2):
                for b9 in range(2):
                    y = [int(not x[6] and x[2]), x[0], 1 - u1, int(x[4] and x[5]), int(not u1 and x[1] and u2) if b5 == 0 else x[4], x[0], int(not x[3] and (not x[7])), int(x[3] or x[4] or x[8]), int(not u1 and (x[4] or u2)) if b9 == 0 else x[8]]
                    out[s, ui, b5, b9] = sum((v << 8 - i for i, v in enumerate(y)))
    return out

@dataclass
class StoppingModel:
    one: list
    killed: dict
    ell: dict
    hit_by_step: dict
    branches: np.ndarray
    actions: np.ndarray
    target: int = TARGET_ID

    def terms(self, r: int, a: int):
        if not 1 <= r <= 100 or not 0 <= a < 20:
            raise ValueError('nonterminal r/action required')
        u = a // 5
        tau = a % 5 + 1
        L = min(r, tau)
        ell = self.ell[u, L]
        fail = np.asarray(self.killed[u, L].sum(axis=1)).ravel() if L == r else np.zeros(512)
        cost = (1 + 0.01 * sum(INPUTS[u])) * ell + 100 * fail
        return (ell, fail, cost, self.killed[u, L] if L < r else None)

def build_model():
    one = []
    killed = {}
    ell = {}
    hits = {}
    alive = np.ones(512)
    alive[TARGET_ID] = 0
    for ui, u in enumerate(INPUTS):
        rr = []
        cc = []
        dd = []
        for s in range(512):
            for j, p in transition_distribution(s, u).items():
                rr.append(s)
                cc.append(j)
                dd.append(p)
        P = csr_matrix((dd, (rr, cc)), shape=(512, 512))
        one.append(P)
        Q = P.multiply(alive[:, None]).multiply(alive[None, :]).tocsr()
        power = eye(512, format='csr').multiply(alive[:, None]).tocsr()
        e = np.zeros(512)
        h = []
        pt = P[:, TARGET_ID].toarray().ravel() * alive
        for l in range(1, 6):
            e = e + np.asarray(power.sum(axis=1)).ravel()
            h.append(power @ pt)
            power = power @ Q
            power.eliminate_zeros()
            killed[ui, l] = power.copy()
            ell[ui, l] = e.copy()
            hits[ui, l] = np.stack(h, axis=1)
    return StoppingModel(one, killed, ell, hits, branch_table(), ACTIONS.copy())

def simulate_hold(branches, state, action, remaining, uniforms):
    if state == TARGET_ID or remaining <= 0:
        raise ValueError('no action at terminal state')
    path = []
    u = int(action) // 5
    tau = int(action) % 5 + 1
    for z in np.asarray(uniforms)[:min(remaining, tau)]:
        state = int(branches[state, u, int(z[0] >= 0.7), int(z[1] >= 0.6)])
        path.append(state)
        if state == TARGET_ID:
            break
    if not path:
        raise ValueError('missing plant uniforms')
    return (state, len(path), state == TARGET_ID, path)
