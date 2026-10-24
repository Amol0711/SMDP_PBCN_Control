from fractions import Fraction as F
from .exact_arithmetic import solve, evaluate
import numpy as np

def exact_lp_certificate(m, reference):
    Y = reference['occupation_measure']
    support = [(i, a) for i in range(4) for a in range(10) if Y[i][a] > 0]
    if len(support) != 5:
        raise ValueError('unexpected nondegenerate support; diagnose rather than substitute')
    obj = m['O'] if reference['objective'] == 'occupancy' else m['G']
    flow = lambda j, i, a: F(i == j) - m['Q'][i][a][j]
    A = [[flow(j, i, a) for i, a in support] for j in range(3)] + [[F(m['actions'][a][1]) for i, a in support], [F(1)] * 5]
    b = [F(0)] * 3 + [F(1), m['budget']]
    ys = solve(A, b)
    y = [[F(0)] * 10 for _ in range(4)]
    for (i, a), v in zip(support, ys):
        y[i][a] = v
    H = [[flow(j, i, a) for j in range(1, 4)] + [F(m['actions'][a][1]), F(1)] for i, a in support]
    hh = solve(H, [obj[i][a] for i, a in support])
    h = [F(0)] + hh[:3]
    rho, lam = hh[3:]
    slack = [[sum((flow(j, i, a) * h[j] for j in range(4))) + rho * m['actions'][a][1] + lam - obj[i][a] for a in range(10)] for i in range(4)]
    primal = sum((y[i][a] * obj[i][a] for i in range(4) for a in range(10)))
    dual = rho + m['budget'] * lam
    checks = {'all40_y_nonnegative': all((v >= 0 for row in y for v in row)), 'all4_flow_equalities': all((sum((y[i][a] * flow(j, i, a) for i in range(4) for a in range(10))) == 0 for j in range(4))), 'plant_time_exactly1': sum((y[i][a] * m['actions'][a][1] for i in range(4) for a in range(10))) == 1, 'update_budget_exact': sum((sum(row) for row in y)) == m['budget'], 'lambda_nonnegative': lam >= 0, 'all40_dual_inequalities': all((v >= 0 for row in slack for v in row)), 'all40_action_complementarity': all((y[i][a] * slack[i][a] == 0 for i in range(4) for a in range(10))), 'strong_duality_exact': primal == dual}
    pi = [[v / sum(row) for v in row] for row in y]
    e = evaluate(pi, m)
    checks['recovered_exact_rate'] = e['metrics']['rate'] == m['budget']
    checks['recovered_exact_objective'] = e['metrics']['occupancy' if reference['objective'] == 'occupancy' else 'reward'] == primal
    if not all(checks.values()):
        raise ArithmeticError(checks)
    return ({'objective': reference['objective'], 'class': 'unrestricted stationary randomized macro-policy, not compact softmax', 'support': support, 'optimal_value_exact': str(primal), 'multiplier_exact': str(lam), 'rho_exact': str(rho), 'h_exact': [str(v) for v in h], 'occupation_measure_exact': [[str(v) for v in row] for row in y], 'policy_exact': [[str(v) for v in row] for row in pi], 'dual_slacks_exact': [[str(v) for v in row] for row in slack], 'checks': checks, 'recovered_metrics_exact': rational_pack(e), 'stored_objective_absolute_difference': abs(float(primal) - reference['optimal_value']), 'stored_multiplier_absolute_difference': abs(float(lam) - reference['lambda']), 'stored_measure_absolute_difference': float(np.max(abs(array(y) - array(Y))))}, primal)

def array(A):return np.asarray(A,dtype=float)
def rational_pack(e):return {k:str(v)for k,v in e['metrics'].items()}
