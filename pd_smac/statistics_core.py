from __future__ import annotations
import math
import numpy as np

def linear_quantile(values, q):
    a = np.sort(np.asarray(values, dtype=np.float64))
    if a.ndim != 1 or len(a) == 0 or (not np.isfinite(a).all()):
        raise ValueError('Nonempty finite vector required')
    q = np.asarray(q, dtype=np.float64)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('Quantile probabilities must lie in [0,1]')
    h = (len(a) - 1) * q
    lo = np.floor(h).astype(int)
    hi = np.ceil(h).astype(int)
    return a[lo] + (h - lo) * (a[hi] - a[lo])

def summarize(values, indices=None, ddof=1):
    x = np.asarray(values, dtype=np.float64)
    if not isinstance(ddof, int) or ddof < 0 or x.ndim != 1 or (not np.isfinite(x).all()) or (len(x) <= ddof):
        raise ValueError('Invalid statistical unit vector')
    mean = float(np.mean(x, dtype=np.float64))
    sd = float(np.sqrt(np.sum((x - mean) ** 2, dtype=np.float64) / (len(x) - ddof)))
    result = {'mean': mean, 'sd': sd, 'n': len(x), 'ddof': ddof}
    if indices is not None:
        indices = np.asarray(indices)
        if indices.ndim != 2 or not np.issubdtype(indices.dtype, np.integer) or len(indices) == 0 or (indices.shape[1] != len(x)) or (np.min(indices) < 0) or (np.max(indices) >= len(x)):
            raise ValueError('Resampling unit mismatch')
        b = np.empty(len(indices))
        for st in range(0, len(indices), 256):
            b[st:st + 256] = np.mean(x[indices[st:st + 256]], axis=1, dtype=np.float64)
        result['bootstrap95'] = linear_quantile(b, [0.025, 0.975]).tolist()
    else:
        result['bootstrap95'] = None
    return result

def stationary(P):
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    if P.shape != (n, n) or not np.isfinite(P).all() or np.min(P) < -1e-12 or (np.max(abs(P.sum(1) - 1)) > 1e-10):
        raise ValueError('Invalid kernel')
    a = P.T - np.eye(n)
    a[-1, :] = 1
    b = np.zeros(n)
    b[-1] = 1
    d = np.linalg.solve(a, b)
    if np.min(d) < -1e-10 or np.max(abs(d @ P - d)) > 1e-10:
        raise ArithmeticError('Invariant-law failure')
    return d

def softmax(w, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    q = np.asarray(w, dtype=np.float64) / temperature
    q = q - q.max(axis=-1, keepdims=True)
    q = np.exp(q)
    return q / q.sum(axis=-1, keepdims=True)

def project(w, bound=12.0):
    w = np.asarray(w, dtype=np.float64)
    if w.ndim < 1 or w.shape[-1] < 1 or (not np.isfinite(w).all()) or (not np.isfinite(bound)) or (bound <= 0):
        raise ValueError('Finite rows and positive projection bound required')
    lo = w.min(axis=-1, keepdims=True) - bound
    hi = w.max(axis=-1, keepdims=True) + bound
    for _ in range(100):
        mid = (lo + hi) / 2
        s = np.clip(w - mid, -bound, bound).sum(axis=-1, keepdims=True)
        lo = np.where(s > 0, mid, lo)
        hi = np.where(s > 0, hi, mid)
    return np.clip(w - (lo + hi) / 2, -bound, bound)

def stationary_metrics(P, pi, occ, reward, duration, exposure):
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 2 or not np.isfinite(pi).all() or np.min(pi) < 0 or (np.max(abs(pi.sum(1) - 1)) > 1e-10) or np.any(np.asarray(duration) <= 0):
        raise ValueError('Invalid policy simplex or duration')
    kernel = np.einsum('sa,sat->st', pi, P)
    d = stationary(kernel)
    tau = np.broadcast_to(duration, pi.shape)
    T = float(np.sum(d[:, None] * pi * tau))
    weights = d[:, None] * pi
    vals = {'occupancy': float(np.sum(weights * occ) / T), 'reward': float(np.sum(weights * reward) / T), 'rate': 1.0 / T, 'mean_duration': T, 'exposure_rate': float(np.sum(weights * exposure) / T)}
    return (vals, d, kernel)

def actor_residual(w, pi, d, kernel, P, reward, duration, lam, gamma, temperature=1.0):
    tau = np.broadcast_to(duration, pi.shape)
    T = float(np.sum(d[:, None] * pi * tau))
    r = reward - lam
    rho = float(np.sum(d[:, None] * pi * r) / T)
    a = np.eye(len(d)) - kernel
    rhs = np.sum(pi * (r - rho * tau), axis=1)
    a[0, :] = 0
    a[0, 0] = 1
    rhs[0] = 0
    h = np.linalg.solve(a, rhs)
    adv = r - rho * tau + np.einsum('sat,t->sa', P, h) - h[:, None]
    grad = d[:, None] * pi * (adv - np.sum(pi * adv, axis=1, keepdims=True)) / (T * temperature)
    return ({'actor_residual_rate': float(np.linalg.norm(project(w + grad) - w)), 'actor_residual_numerator': float(np.linalg.norm(project(w + T * grad) - w)), 'primal': max(0.0, 1 / T - gamma), 'complementarity': abs(lam * (1 / T - gamma))}, grad, h)

def biological_counts(z, horizon, target_index):
    states = z['states']
    inputs = z['inputs']
    dec = z['decisions']
    n = len(states)
    if states.shape[1] != horizon + 1 or inputs.shape != (n, horizon):
        raise ValueError('Persistent horizon mismatch')
    target = states[:, 1:, target_index] == 0
    hit = target.any(1)
    first = np.where(hit, np.argmax(target, axis=1) + 1, horizon + 1)
    ep = dec[:, 0].astype(int)
    K = np.bincount(ep, minlength=n)
    S = np.bincount(ep, weights=dec[:, 5], minlength=n)
    executed = np.bincount(ep, weights=dec[:, 6], minlength=n)
    if not np.all(executed == horizon) or not np.all(K > 0):
        raise AssertionError('Executed hold accounting failure')
    cc = np.column_stack([np.full(n, horizon), K, S, inputs.sum(1), target.sum(1), hit, first]).astype(np.int64)
    if not np.array_equal(cc, z['counts']):
        raise AssertionError('Saved counts disagree with independently counted states/inputs/decisions')
    return cc

def biological_metrics(counts, gamma=0.4):
    T, K, S, A, O, hit, coded = np.asarray(counts, dtype=float).T
    if np.any(T <= 0) or np.any(K <= 0):
        raise ValueError('Nonpositive denominator')
    values = {'hit_probability': float(np.mean(hit)), 'coded_hitting_time': float(np.mean(coded)), 'occupancy': float(sum(O) / sum(T)), 'reward': float((sum(O) - 0.06 * sum(A)) / sum(T)), 'rate': float(sum(K) / sum(T)), 'scheduled_hold': float(sum(S) / sum(K)), 'realized_hold': float(sum(T) / sum(K)), 'input_on_fraction': float(sum(A) / sum(T)), 'residual': float(sum(K) / sum(T) - gamma)}
    individual = {'hit_probability': hit, 'coded_hitting_time': coded, 'occupancy': O / T, 'reward': (O - 0.06 * A) / T, 'rate': K / T, 'scheduled_hold': S / K, 'realized_hold': T / K, 'input_on_fraction': A / T, 'residual': K / T - gamma}
    return (values, individual)

def first_arrival_counts(z, horizon=100, target=507):
    states = z['states']
    n = len(states)
    at = states == target
    hit = at.any(axis=1)
    first = np.where(hit, np.argmax(at, axis=1), horizon)
    T = first.astype(int)
    if not np.array_equal(states[:, 0], np.repeat(np.arange(512), 4)):
        raise AssertionError('Initial-state stratification changed')
    da = z['decision_actions']
    ha = z['held_actions']
    mask = np.arange(horizon)[None, :] < T[:, None]
    dm = (da >= 0) & mask
    K = dm.sum(1)
    S = np.where(dm, da % 5 + 1, 0).sum(1)
    input_index = np.maximum(ha, 0) // 5
    A = np.where(mask, input_index // 2 + input_index % 2, 0).sum(1)
    for field, v in [('T_cap', T), ('updates', K), ('scheduled_duration_total', S), ('input_exposure', A), ('hit', hit), ('coded_hit_time', np.where(hit, T, horizon + 1))]:
        if not np.array_equal(v, z[field]):
            raise AssertionError('First-arrival raw mismatch ' + field)
    if np.any(K[T == 0]) or np.any(A[T == 0]) or np.any(S[T == 0]):
        raise AssertionError('Nonzero initially targeted accounting')
    return {'T': T, 'K': K, 'S': S, 'A': A, 'hit': hit}

def first_arrival_metrics(c):
    T, K, S, A, hit = (c[k].astype(float) for k in ['T', 'K', 'S', 'A', 'hit'])
    F = 1 - hit
    n = len(T)
    success = hit.astype(bool)
    v = {'hit_probability': float(hit.mean()), 'T_cap': float(T.mean()), 'coded_hit_time': float(np.where(success, T, 101).mean()), 'successful_hit_time': float(T[success].mean()) if success.any() else None, 'updates': float(K.mean()), 'input_exposure': float(A.mean()), 'J': float((T + 0.01 * A + 100 * F).mean()), 'budget_numerator': float((K - 0.4 * T).mean()), 'rate': float(K.sum() / T.sum()) if T.sum() > 0 else None, 'scheduled_mean_hold': float(S.sum() / K.sum()) if K.sum() > 0 else None, 'realized_mean_hold': float(T.sum() / K.sum()) if K.sum() > 0 else None}
    extra = {'mean_individual_rate_positive_time': float(np.mean(K[T > 0] / T[T > 0])) if np.any(T > 0) else None, 'initial_target_count': int((T == 0).sum()), 'failure_count': int(F.sum()), 'episodes': n, 'total_T': int(T.sum()), 'total_K': int(K.sum()), 'total_S': int(S.sum()), 'total_A': int(A.sum())}
    return (v, extra)
