from __future__ import annotations
import time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.optimize import linprog
from .model import INPUTS, TARGET_ID
from .io_contract import dump, npz
FIELDS = ['T_cap', 'updates', 'input_exposure', 'failure_probability', 'scheduled_duration_total']

def q_metrics(m, r, V):
    q = np.empty((512, 20, 5), dtype=np.float64)
    for a in range(20):
        ell, fail, cost, cont = m.terms(r, a)
        q[:, a, 0] = ell
        q[:, a, 1] = 1.0
        q[:, a, 2] = sum(INPUTS[a // 5]) * ell
        q[:, a, 3] = fail
        q[:, a, 4] = a % 5 + 1
        if cont is not None:
            q[:, a, :] += cont @ V[r - (a % 5 + 1)]
    q[TARGET_ID, :, :] = 0
    return q

def summary(v):
    T, K, A, F, S = np.mean(v, axis=0)
    return {'T_cap': float(T), 'updates': float(K), 'input_exposure': float(A), 'failure_probability': float(F), 'hit_probability': float(1 - F), 'J': float(T + 0.01 * A + 100 * F), 'budget_numerator': float(K - 0.4 * T), 'rate': float(K / T) if T > 0 else None, 'coded_hit_time': float(T + F), 'successful_hit_time': float((T - 100 * F) / (1 - F)) if F < 1 else None, 'scheduled_mean_hold': float(S / K) if K > 0 else None, 'realized_mean_hold': float(T / K) if K > 0 else None}

def evaluate(m, policy, H=100):
    V = np.zeros((H + 1, 512, 5))
    for r in range(1, H + 1):
        q = q_metrics(m, r, V)
        if policy.ndim == 2:
            V[r] = q[np.arange(512), policy[r]]
        else:
            V[r] = np.einsum('sa,saf->sf', policy[r], q)
        V[r, TARGET_ID] = 0
    return (V, summary(V[H]))

def dynamic_program(m, actions, H=100, multiplier=0.0, tie=True):
    V = np.zeros((H + 1, 512, 5))
    policy = np.zeros((H + 1, 512), np.int16)
    pure = np.zeros((H + 1, 512))
    actions = np.array(actions)
    for r in range(1, H + 1):
        q = q_metrics(m, r, V)
        cost = q[:, :, 0] + 0.01 * q[:, :, 2] + 100 * q[:, :, 3] + multiplier * (q[:, :, 1] - 0.4 * q[:, :, 0])
        restricted = cost[:, actions]
        mins = restricted.min(axis=1)
        if tie:
            mask = restricted <= mins[:, None] + 1e-10
            Ks = np.where(mask, q[:, actions, 1], np.inf)
            minK = Ks.min(axis=1)
            candidates = mask & (Ks == minK[:, None])
            choice = actions[candidates.argmax(axis=1)]
        else:
            choice = actions[restricted.argmin(axis=1)]
        policy[r] = choice
        V[r] = q[np.arange(512), choice]
        V[r, TARGET_ID] = 0
        policy[r, TARGET_ID] = 0
        cq = np.empty((512, len(actions)))
        for j, a in enumerate(actions):
            ell, fail, base, cont = m.terms(r, int(a))
            cq[:, j] = base + multiplier * (1 - 0.4 * ell)
            if cont is not None:
                cq[:, j] += cont @ pure[r - (a % 5 + 1)]
        pure[r] = cq.min(axis=1)
        pure[r, TARGET_ID] = 0
    s = summary(V[H])
    s['pure_optimum'] = float(pure[H].mean())
    s['tie_objective_gap'] = float(s['J'] + multiplier * s['budget_numerator'] - s['pure_optimum'])
    return (policy, V, pure, s)

def reachable_pairs(m, H=100):
    reach = np.zeros((H + 1, 512), bool)
    reach[H] = True
    reach[:, TARGET_ID] = False
    for r in range(H, 1, -1):
        for u in range(4):
            for tau in range(1, min(5, r - 1) + 1):
                reach[r - tau] |= np.asarray(m.killed[u, tau].T @ reach[r].astype(float)).ravel() > 0
    reach[:, TARGET_ID] = False
    pairs = np.array([(r, int(s)) for r in range(H, 0, -1) for s in np.flatnonzero(reach[r])], np.int16)
    rid = np.full((H + 1, 512), -1, np.int32)
    rid[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
    return (pairs, rid)

def occupation_problem(m, H=100):
    pairs, rid = reachable_pairs(m, H)
    n = len(pairs)
    nv = n * 20
    fields = np.zeros((nv, 5))
    rows = [np.repeat(np.arange(n, dtype=np.int32), 20)]
    cols = [np.arange(nv, dtype=np.int32)]
    data = [np.ones(nv)]
    b = (pairs[:, 0] == H).astype(float) / 512.0
    for r in range(1, H + 1):
        ss = np.flatnonzero(rid[r] >= 0)
        ii = rid[r, ss]
        for a in range(20):
            jj = ii * 20 + a
            ell, fail, cost, cont = m.terms(r, a)
            fields[jj, 0] = ell[ss]
            fields[jj, 1] = 1.0
            fields[jj, 2] = sum(INPUTS[a // 5]) * ell[ss]
            fields[jj, 3] = fail[ss]
            fields[jj, 4] = a % 5 + 1
            if cont is not None:
                sub = cont[ss].tocoo()
                dest = rid[r - a % 5 - 1, sub.col]
                if np.any(dest < 0):
                    raise AssertionError('reachability pruned a reachable node')
                rows.append(dest)
                cols.append(ii[sub.row] * 20 + a)
                data.append(-sub.data)
    flow = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, nv)).tocsr()
    c = fields[:, 0] + 0.01 * fields[:, 2] + 100 * fields[:, 3]
    g = fields[:, 1] - 0.4 * fields[:, 0]
    return (pairs, flow, b, fields, c, g)

def recover(pairs, y, H=100):
    p = np.zeros((H + 1, 512, 20))
    p[:, :, 0] = 1.0
    mass = y.reshape(-1, 20).sum(axis=1)
    active = mass > 0
    rr, ss = pairs[active].T
    p[rr, ss] = y.reshape(-1, 20)[active] / mass[active, None]
    return p

def check_lp(result, A, b, Aub, bub, c):
    if not result.success:
        raise RuntimeError(result.message)
    x = result.x
    ye = result.eqlin.marginals
    yu = result.ineqlin.marginals
    reduced = c - A.T @ ye - Aub.T @ yu
    return {'status': int(result.status), 'message': result.message, 'iterations': int(result.nit), 'objective': float(c @ x), 'flow_max_abs': float(np.max(np.abs(A @ x - b))), 'inequality_max_violation': float(max(0, np.max(Aub @ x - bub))), 'negative_occupation': float(max(0, -x.min())), 'dual_max_violation': float(max(0, -reduced.min(), yu.max())), 'primal_dual_gap': float(c @ x - b @ ye - bub @ yu), 'reduced_cost_complementarity': float(np.max(np.abs(x * reduced))), 'inequality_complementarity': float(np.max(np.abs(yu * (Aub @ x - bub)))), 'inequality_marginals': yu.tolist()}
