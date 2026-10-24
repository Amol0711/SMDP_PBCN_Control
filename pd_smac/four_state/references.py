from __future__ import annotations
import csv, itertools, json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from .model import Model, evaluate, deterministic

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def tie(rows, objective):
    best = max((r[objective] for r in rows))
    first = [r for r in rows if r[objective] >= best - 1e-10]
    rate = min((r['rate'] for r in first))
    second = [r for r in first if r['rate'] <= rate + 1e-10]
    chosen = min(second, key=lambda r: tuple(r['actions']))
    return {'selected': chosen, 'optimal_objective': best, 'objective_ties': len(first), 'minimum_rate_among_objective_ties': rate, 'rate_ties': len(second), 'rule': 'objective within 1e-10; rate within 1e-10; canonical lexicographic action tuple'}

def enumerate_all(model: Model, out: Path) -> tuple[list, dict]:
    rows = []
    for ids in itertools.product(range(10), repeat=4):
        e = evaluate(model, deterministic(ids))
        rows.append({**{k: e[k] for k in ['occupancy', 'reward', 'rate', 'mean_duration', 'exposure_rate', 'feasible', 'stationary_check_abs']}, 'actions': list(ids), 'irreducible': e['chain']['irreducible'], 'unichain': e['chain']['unichain'], 'recurrent_classes': e['chain']['recurrent_classes'], 'stationary': e['stationary']})
    fields = ['policy_index', 'a_S', 'a_B', 'a_O', 'a_L', 'occupancy', 'reward', 'rate', 'mean_duration', 'exposure_rate', 'feasible', 'irreducible', 'unichain', 'stationary_check_abs']
    with (out / 'enumeration.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({'policy_index': i, **dict(zip(fields[1:5], r['actions'])), **{k: r[k] for k in fields[5:]}})
    feasible = [r for r in rows if r['feasible']]
    fixed = []
    for t in range(1, 6):
        subset = [r for r in rows if all((model.actions[a, 1] == t for a in r['actions']))]
        fixed.append({'duration': t, 'policy_count': len(subset), 'occupancy': tie(subset, 'occupancy'), 'reward': tie(subset, 'reward')})
    summary = {'count': len(rows), 'feasible_count': len(feasible), 'reducible_count': sum((not r['irreducible'] for r in rows)), 'multichain_count': sum((not r['unichain'] for r in rows)), 'max_stationary_check_abs': max((r['stationary_check_abs'] for r in rows)), 'deterministic': {o: tie(feasible, o) for o in ['occupancy', 'reward']}, 'fixed': fixed}
    dump(out / 'enumeration_summary.json', summary)
    return (rows, summary)

def solve_lp(model: Model, objective: str, enumerated: list, out: Path) -> dict:
    g = getattr(model, 'occupancy' if objective == 'occupancy' else 'rewards').reshape(-1)
    F = (np.eye(4)[:, None, :] - model.transitions).reshape(40, 4).T
    T = model.duration.ravel()
    Aeq = np.vstack((F[:3], T))
    beq = np.array([0.0, 0, 0, 1])
    options = {'primal_feasibility_tolerance': 1e-10, 'dual_feasibility_tolerance': 1e-10, 'ipm_optimality_tolerance': 1e-12}
    res = linprog(-g, A_ub=np.ones((1, 40)), b_ub=[model.budget], A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs-ds', options=options)
    if not res.success:
        raise RuntimeError(f'LP failed: {res.message}')
    y = res.x
    z = -res.eqlin.marginals
    lam = float(-res.ineqlin.marginals[0])
    rho = float(z[-1])
    h = np.r_[z[:3], 0.0]
    h -= h[0]
    slack = F.T @ h + rho * T + lam - g
    primal = float(g @ y)
    dual = rho + model.budget * lam
    pi = np.zeros((4, 10))
    Y = y.reshape(4, 10)
    for s in range(4):
        mass = Y[s].sum()
        if mass > 0:
            pi[s] = Y[s] / mass
        else:
            pi[s, 4] = 1
    recovered = evaluate(model, pi)
    values = np.array([r['occupancy' if objective == 'occupancy' else 'reward'] for r in enumerated])
    rates = np.array([r['rate'] for r in enumerated])
    dr = linprog([1.0, model.budget], A_ub=np.c_[-np.ones(len(rates)), -rates], b_ub=-values, bounds=[(None, None), (0, None)], method='highs-ds', options=options)
    if not dr.success:
        raise RuntimeError('independent deterministic-policy dual failed')
    tight = np.flatnonzero(abs(values - lam * rates - rho) < 1e-09)
    checks = {'flow_abs': float(np.max(abs(F @ y))), 'duration_normalization_abs': float(abs(T @ y - 1)), 'update_violation': float(max(0, y.sum() - model.budget)), 'negative_measure': float(max(0, -min(y))), 'dual_violation': float(max(0, -min(slack))), 'duality_gap_abs': abs(primal - dual), 'action_complementarity_abs': float(np.max(abs(y * slack))), 'budget_complementarity_abs': abs(lam * (model.budget - y.sum())), 'recovered_measure_abs': float(np.max(abs(np.array(recovered['occupation_measure']).ravel() - y))), 'recovered_objective_abs': abs(recovered['occupancy' if objective == 'occupancy' else 'reward'] - primal), 'deterministic_dual_objective_abs': abs(float(dr.fun) - primal), 'deterministic_dual_multiplier_abs': abs(float(dr.x[1]) - lam), 'max_deterministic_bound_violation': float(max(0, np.max(values - lam * rates - rho)))}
    if any((v > 1e-08 for v in checks.values())):
        raise ArithmeticError(checks)
    result = {'objective': objective, 'policy_class': 'unrestricted stationary randomized macro-policy', 'optimal_value': primal, 'dual_bound': dual, 'lambda': lam, 'rho': rho, 'value_h': h.tolist(), 'occupation_measure': Y.tolist(), 'dual_slacks': slack.reshape(4, 10).tolist(), 'recovered': recovered, 'zero_mass_rows': [s for s in range(4) if Y[s].sum() == 0], 'independent_deterministic_dual': {'rho': float(dr.x[0]), 'lambda': float(dr.x[1]), 'objective': float(dr.fun), 'tight_policy_indices': tight.tolist(), 'tight_policies': [enumerated[i] for i in tight]}, 'checks': checks, 'solver': {'method': 'highs-ds', 'status': int(res.status), 'message': res.message, 'iterations': int(res.nit), 'options': options}, 'marginal_sign': 'lambda=-ineqlin.marginals[0] for minimization of negative reward', 'not_an_optimum_over_compact_full_support_actor_class': True}
    dump(out / f'lp_{objective}.json', result)
    return result
