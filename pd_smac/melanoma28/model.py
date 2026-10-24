from __future__ import annotations
import itertools, math, json, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

def mi(x, y):
    out = 0.0
    for a in (0, 1):
        pa = np.mean(x == a)
        if pa == 0:
            continue
        for b in (0, 1):
            pb = np.mean(y == b)
            p = np.mean((x == a) & (y == b))
            if p > 0 and pb > 0:
                out += p * math.log(p / (pa * pb) + 1e-12, 2)
    return out

def fit_rule(obs, target, predictors):
    groups = defaultdict(list)
    for row in obs:
        groups[tuple((int(row[p]) for p in predictors))].append(int(row[target]))
    default = int(obs[:, target].mean() >= 0.5)
    table = {key: int(sum(v) >= len(v) / 2) for key, v in groups.items()}
    pred = np.array([table.get(tuple((int(row[p]) for p in predictors)), default) for row in obs], dtype=np.int8)
    return {'predictors': list(map(int, predictors)), 'table': {','.join(map(str, k)): int(v) for k, v in table.items()}, 'default': default, 'acc': float(np.mean(pred == obs[:, target]))}

class Model:

    def __init__(self, root: Path, config: dict):
        for item in config['model']['inputs']:
            p = root / item['path']
            if p.stat().st_size != item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Input identity failed: ' + str(p))
        unique = pd.read_csv(root / config['model']['inputs'][0]['path'])
        manifest = pd.read_csv(root / config['model']['inputs'][1]['path'])
        self.sample_columns = [c for c in unique.columns if c.startswith('T')]
        self.obs = unique[self.sample_columns].to_numpy(dtype=np.int8).T
        self.p = int(manifest.loc[manifest.is_control_gene == 1, 'local_index'].iloc[0])
        self.w = int(manifest.loc[manifest.is_target_gene == 1, 'local_index'].iloc[0])
        if self.obs.shape[1] != 28 or self.p != 0 or self.w != 1 or (not np.isin(self.obs, [0, 1]).all()):
            raise ValueError('Invalid 28-node input')
        self.reset_indices = np.flatnonzero(self.obs[:, self.w] == 1)
        if self.reset_indices.tolist() != config['model']['reset_pool_sample_indices']:
            raise ValueError('Reset pool changed')
        self.base = []
        self.controlled = []
        for t in range(28):
            top = [j for _, j in sorted(((mi(self.obs[:, j], self.obs[:, t]), j) for j in range(28) if j != t), reverse=True)[:6]]
            cand = [fit_rule(self.obs, t, p) for k in (1, 2, 3) for p in itertools.combinations(top, k)]
            cand.sort(key=lambda r: (r['acc'], -len(r['predictors'])), reverse=True)
            best = cand[0]
            second = next((r for r in cand[1:] if r['predictors'] != best['predictors']), best)
            base = [best, second]
            self.base.append(base)
            if t == self.p:
                ctrl = base
            else:
                topc = sorted(set(top + [self.p]))
                cp = [fit_rule(self.obs, t, p) for k in (1, 2, 3) for p in itertools.combinations(topc, k) if self.p in p]
                cp.sort(key=lambda r: (r['acc'], -len(r['predictors'])), reverse=True)
                if t == self.w:
                    first = fit_rule(self.obs, t, (self.p,))
                    second = next((r for r in cp if r['predictors'] != first['predictors']), cp[0])
                    ctrl = [first, second]
                else:
                    first = cp[0]
                    second = next((r for r in cp[1:] if r['predictors'] != first['predictors']), first)
                    ctrl = [first, second]
            self.controlled.append(ctrl)
        self.responsive = np.array([i for i, bank in enumerate(self.controlled) if any((self.p in r['predictors'] for r in bank))], dtype=np.int64)
        self.wpred = np.array(self.base[self.w][0]['predictors'][:3], dtype=np.int64)
        self.pred = np.zeros((28, 2, 3), dtype=np.int64)
        self.length = np.zeros((28, 2), dtype=np.int64)
        self.table = np.zeros((28, 2, 8), dtype=np.int8)
        self.pfirst = np.zeros(28)
        for i, bank in enumerate(self.base):
            weights = np.array([max(0.001, r['acc']) for r in bank], dtype=float)
            self.pfirst[i] = (weights / weights.sum())[0]
            for j, r in enumerate(bank):
                l = len(r['predictors'])
                self.length[i, j] = l
                self.pred[i, j, :l] = r['predictors']
                for a in range(2 ** l):
                    key = ','.join((str(a >> l - 1 - k & 1) for k in range(l)))
                    self.table[i, j, a] = r['table'].get(key, r['default'])
        self.dim = 6 + len(self.wpred)
        self.record = {'kind': 'frozen_input_fitted_model', 'state_dimension': 28, 'sample_columns': self.sample_columns, 'observations': self.obs.tolist(), 'reset_indices': self.reset_indices.tolist(), 'control_index': self.p, 'target_index': self.w, 'base_banks': self.base, 'controlled_banks_for_feature_definition_only': self.controlled, 'feature': {'dtype': 'float32', 'dimension': self.dim, 'W_PREDS': self.wpred.tolist(), 'RESPONSIVE_IDX': self.responsive.tolist(), 'order': ['1', 'WNT5A', 'pirin'] + [f'gene_{i}' for i in self.wpred] + ['mean_all', 'mean_responsive', 'population_std_responsive']}, 'rule_first_probability': self.pfirst.tolist(), 'semantics': 'synchronous base-bank update; xor pirin with held input at every plant step'}
        self.signature = hashlib.sha256(json.dumps(self.record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def reset(self, u):
        i = self.reset_indices[min(int(u * len(self.reset_indices)), len(self.reset_indices) - 1)]
        s = self.obs[i].copy()
        s[self.w] = 1
        return (s, int(i))

    def step(self, s, u, tape):
        return step(s, int(u), tape, self.pred, self.length, self.table, self.pfirst)

    def features(self, s):
        return features(s, self.wpred, self.responsive)

@njit(cache=False)
def step(s, u, tape, pred, length, table, pfirst):
    ns = np.empty(28, dtype=np.int8)
    for i in range(28):
        j = 0 if tape[i] < pfirst[i] else 1
        code = 0
        for k in range(length[i, j]):
            code = 2 * code + s[pred[i, j, k]]
        ns[i] = table[i, j, code]
    ns[0] = ns[0] ^ u
    return ns

@njit(cache=False)
def features(s, wpred, responsive):
    f = np.empty(6 + len(wpred), dtype=np.float32)
    f[0] = 1.0
    f[1] = s[1]
    f[2] = s[0]
    for k in range(len(wpred)):
        f[3 + k] = s[wpred[k]]
    m = 0.0
    for v in s:
        m += v
    f[-3] = m / 28
    m = 0.0
    for i in responsive:
        m += s[i]
    m /= len(responsive)
    f[-2] = m
    variance = 0.0
    for i in responsive:
        variance += (s[i] - m) ** 2
    f[-1] = math.sqrt(variance / len(responsive))
    return f

@njit(cache=False)
def batch_step(states, actions, tapes, pred, length, table, pfirst):
    out = np.empty_like(states)
    for k in range(len(states)):
        out[k] = step(states[k], actions[k], tapes[k], pred, length, table, pfirst)
    return out

@njit(cache=False)
def batch_features(states, wpred, responsive):
    out = np.empty((len(states), 6 + len(wpred)), dtype=np.float32)
    for k in range(len(states)):
        out[k] = features(states[k], wpred, responsive)
    return out
