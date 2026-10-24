from __future__ import annotations
import numpy as np
from numba import njit

@njit(cache=False, fastmath=False)
def project_row(v):
    n = len(v)
    center = np.sum(v) / n
    z = v - center
    bound = 12.0
    if np.max(np.abs(z)) <= bound:
        x = z.copy()
        lam = center
    else:
        lo = np.min(z) - bound
        hi = np.max(z) + bound
        for _ in range(100):
            mid = lo / 2 + hi / 2
            f = 0.0
            for j in range(n):
                f += min(bound, max(-bound, z[j] - mid))
            if f > 0:
                lo = mid
            else:
                hi = mid
        shift = lo / 2 + hi / 2
        for _ in range(2):
            count = 0
            num = 0.0
            for j in range(n):
                y = z[j] - shift
                if y > -bound and y < bound:
                    count += 1
                    num += z[j]
                elif y >= bound:
                    num += bound
                else:
                    num -= bound
            if count > 0:
                shift = num / count
        x = np.empty(n)
        for j in range(n):
            x[j] = min(bound, max(-bound, z[j] - shift))
        lam = shift + center
    rowerr = abs(np.sum(x))
    normal = 0.0
    stationarity = 0.0
    comp = 0.0
    for j in range(n):
        normal = max(normal, abs(x[j] - min(bound, max(-bound, v[j] - lam))))
        q = x[j] - v[j] + lam
        lower = max(q, 0.0) if x[j] <= -bound + 1e-12 else 0.0
        upper = max(-q, 0.0) if x[j] >= bound - 1e-12 else 0.0
        stationarity = max(stationarity, abs(q + upper - lower))
        comp = max(comp, abs(lower * (x[j] + bound)), abs(upper * (bound - x[j])))
    if rowerr > 1e-10 or normal > 1e-09 or stationarity > 1e-09 or (comp > 1e-09):
        raise ArithmeticError('projection certificate failed')
    return (x, np.array([rowerr, normal, stationarity, comp]))

@njit(cache=False, fastmath=False)
def probs_row(w, temp):
    p = np.exp((w - np.max(w)) / temp)
    return p / np.sum(p)

@njit(cache=False, fastmath=False)
def categorical(p, u):
    c = 0.0
    for i in range(len(p)):
        c += p[i]
        if u < c:
            return i
    if u < 1.0 and abs(c - 1) < 1e-12:
        return len(p) - 1
    raise ArithmeticError('invalid categorical distribution')

@njit(cache=False, fastmath=False)
def core(P, actions, target, reset_tape, actor_tape, plant_tape, episodes=1200):
    W = np.zeros((4, 10))
    h = np.zeros(4)
    eta = 0.0
    lam = 0.1
    cap = episodes * 299
    events_i = np.empty((cap, 7), np.int64)
    events_f = np.empty((cap, 8))
    states = np.empty(cap, np.int8)
    episode_i = np.empty((episodes, 5), np.int64)
    episode_f = np.empty((episodes, 4))
    checkpoint_W = np.empty((4, 4, 10))
    checkpoint_h = np.empty((4, 4))
    checkpoint_f = np.empty((4, 3))
    checkpoint_counters = np.empty((4, 2), np.int64)
    n = 0
    tglobal = 0
    maxcert = np.zeros(4)
    clipped_updates = 0
    maxlam = lam
    cap_count = 0
    for ep in range(1, episodes + 1):
        s = categorical(np.full(4, 0.25), reset_tape[ep - 1])
        start = s
        t = 0
        k = 0
        epocc = 0
        epr = 0.0
        fac = (1.0 + ep / 1000.0) ** 0.6
        at = 0.12 / fac
        av = 0.16 / fac
        ae = 0.05 / fac
        al = 0.08 / fac
        temp = max(0.15, 1.0 - 0.85 * ep / 1200.0)
        while t < 295:
            p = probs_row(W[s], temp)
            ai = categorical(p, actor_tape[n])
            u = actions[ai, 0]
            tau = actions[ai, 1]
            ns = s
            oc = 0
            offset = tglobal
            for _ in range(tau):
                ns = categorical(P[u, ns], plant_tape[tglobal])
                states[tglobal] = ns
                tglobal += 1
                oc += int(target[ns])
            R = oc - 0.02 * u * tau
            delta = R - lam - eta * tau + h[ns] - h[s]
            eta += ae * delta
            h[s] += av * delta
            h -= h[0]
            score = -p
            score[ai] += 1
            candidate = W[s] + at * delta * score / temp
            if np.max(np.abs(candidate - np.mean(candidate))) > 12:
                clipped_updates += 1
            W[s], cert = project_row(candidate)
            for j in range(4):
                maxcert[j] = max(maxcert[j], cert[j])
            lam = min(50.0, max(0.0, lam + al * (1.0 - 0.3 * tau)))
            maxlam = max(maxlam, lam)
            if lam >= 50.0 - 1e-10:
                cap_count += 1
            if not np.isfinite(eta) or not np.all(np.isfinite(W)) or (not np.all(np.isfinite(h))):
                raise ArithmeticError('nonfinite learner state')
            events_i[n] = np.array([ep, s, ai, ns, offset, tau, oc])
            events_f[n] = np.array([R, delta, eta, lam, h[0], h[1], h[2], h[3]])
            n += 1
            t += tau
            k += 1
            epocc += oc
            epr += R
            s = ns
        episode_i[ep - 1] = np.array([start, s, t, k, epocc])
        episode_f[ep - 1] = np.array([epr, lam, eta, temp])
        if ep % 300 == 0:
            cp = ep // 300 - 1
            checkpoint_W[cp] = W
            checkpoint_h[cp] = h
            checkpoint_f[cp] = np.array([eta, lam, temp])
            checkpoint_counters[cp] = np.array([n, tglobal])
    return (events_i[:n].copy(), events_f[:n].copy(), states[:tglobal].copy(), episode_i, episode_f, checkpoint_W, checkpoint_h, checkpoint_f, checkpoint_counters, maxcert, np.array([clipped_updates, maxlam, cap_count]))
