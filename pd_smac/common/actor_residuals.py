from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import re
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .actor_geometry import finite_array, positive_scalar, require_domain, domain_errors, actor_mapping, PROJECTION_ID
from .actor_scores import softmax_rows
from .actor_steps import multiplier_step

def array_hash(a: ArrayLike) -> str:
    x = finite_array(a, 'hashed array')
    header = json.dumps({'dtype': '<f8', 'shape': list(x.shape)}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(header.encode() + b'\x00' + np.ascontiguousarray(x, dtype='<f8').tobytes()).hexdigest()

@dataclass(frozen=True)
class DiagnosticContext:
    study_id: str
    protocol_sha256: str
    checkpoint_id: str
    actor_contract_id: str
    objective_id: str
    objective_family: str = 'continuing_rate'
    policy_mode: str = 'stochastic'
    record_kind: str = 'scientific_diagnostic'

    def validate(self, family: str) -> None:
        if any((not isinstance(v, str) or not v for v in asdict(self).values())):
            raise ValueError('all context fields must be nonempty strings')
        if not re.fullmatch('[0-9a-f]{64}', self.protocol_sha256):
            raise ValueError('protocol_sha256 is not SHA-256 hex')
        if self.policy_mode != 'stochastic':
            raise ValueError('actor/multiplier diagnostics cannot use a greedy extraction')
        if self.objective_family != family:
            raise ValueError('diagnostic objective-family mismatch')
        if self.record_kind not in {'scientific_diagnostic', 'synthetic_unit_test'}:
            raise ValueError('unsupported record kind')

@dataclass(frozen=True)
class ExactPolicyEvaluation:
    context: DiagnosticContext
    theta: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    gradient_rate: NDArray[np.float64]
    mean_duration: float
    mean_reward: float
    mean_expenditure: float
    reward_rate: float
    update_rate: float
    penalized_rate: float
    rate_lagrangian: float
    multiplier: float
    budget: float
    temperature: float
    bound: float
    model_sha256: str
    checks: dict
    stationary: NDArray[np.float64]
    value: NDArray[np.float64]

def _matrix(x: ArrayLike, shape: tuple, name: str, *, positive=False, nonnegative=False):
    a = finite_array(x, name, min_ndim=0)
    try:
        a = np.broadcast_to(a, shape).copy()
    except ValueError as e:
        raise ValueError(f'{name} has incompatible shape') from e
    if positive and np.any(a <= 0):
        raise ValueError(f'{name} must be positive')
    if nonnegative and np.any(a < 0):
        raise ValueError(f'{name} must be nonnegative')
    return a

def evaluate_tabular_smdp(theta: ArrayLike, transitions: ArrayLike, rewards: ArrayLike, durations: ArrayLike, expenditures: ArrayLike, *, multiplier: float, budget: float, context: DiagnosticContext, temperature: float=1.0, bound: float=12.0, reference_state: int=0) -> ExactPolicyEvaluation:
    context.validate('continuing_rate')
    w = require_domain(theta, bound)
    if w.ndim != 2:
        raise ValueError('tabular theta must have shape (states,actions)')
    n, m = w.shape
    if not isinstance(reference_state, (int, np.integer)) or not 0 <= reference_state < n:
        raise ValueError('invalid reference state')
    P = finite_array(transitions, 'transitions', min_ndim=3)
    if P.shape != (n, m, n) or np.any(P < 0) or np.max(np.abs(P.sum(axis=-1) - 1)) > 1e-12:
        raise ValueError('invalid stochastic macro-transition tensor')
    R = _matrix(rewards, (n, m), 'rewards')
    T = _matrix(durations, (n, m), 'durations', positive=True)
    C = _matrix(expenditures, (n, m), 'expenditures', nonnegative=True)
    if not np.isfinite(multiplier) or multiplier < 0 or (not np.isfinite(budget)) or (budget < 0):
        raise ValueError('multiplier and budget must be finite and nonnegative')
    temp = positive_scalar(temperature, 'temperature')
    pi = softmax_rows(w, temp)
    if np.any(pi <= 0):
        raise ArithmeticError('full-support actor underflow')
    kernel = np.einsum('sa,san->sn', pi, P)
    A = kernel.T - np.eye(n)
    A[-1] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    try:
        d = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError as e:
        raise ArithmeticError('unique invariant solve failed; do not report exact rates') from e
    stationary_error = float(np.max(np.abs(d @ kernel - d)))
    if np.min(d) < -1e-14 or abs(d.sum() - 1) > 1e-12 or stationary_error > 1e-10:
        raise ArithmeticError('invariant-law residual/positivity check failed')
    weight = d[:, None] * pi
    tb = float(np.sum(weight * T))
    gb = float(np.sum(weight * R))
    cb = float(np.sum(weight * C))
    eta = gb / tb
    kappa = cb / tb
    eta_pen = eta - multiplier * kappa
    centered = R - multiplier * C - eta_pen * T
    b = np.sum(pi * centered, axis=1)
    H = np.eye(n) - kernel
    q = b.copy()
    H[reference_state] = 0
    H[reference_state, reference_state] = 1
    q[reference_state] = 0
    try:
        h = np.linalg.solve(H, q)
    except np.linalg.LinAlgError as e:
        raise ArithmeticError('differential-value solve failed') from e
    value_error = float(np.max(np.abs((np.eye(n) - kernel) @ h - b)))
    if value_error > 1e-09 or abs(h[reference_state]) > 1e-12:
        raise ArithmeticError('differential-value residual failed')
    advantage = centered + np.einsum('san,n->sa', P, h) - h[:, None]
    conditional_mean = np.sum(pi * advantage, axis=1, keepdims=True)
    g = weight * (advantage - conditional_mean) / (tb * temp)
    model_sig = hashlib.sha256('|'.join((array_hash(x) for x in [P, R, T, C])).encode()).hexdigest()
    for x in [w, pi, g, d, h]:
        x.setflags(write=False)
    checks = {'stationary_equation_abs': stationary_error, 'stationary_sum_abs': float(abs(d.sum() - 1)), 'min_stationary_mass': float(np.min(d)), 'value_equation_abs': value_error, 'conditional_advantage_mean_abs': float(np.max(np.abs(conditional_mean))), 'gradient_row_sum_abs': float(np.max(np.abs(g.sum(axis=-1))))}
    return ExactPolicyEvaluation(context, w, pi, g, tb, gb, cb, eta, kappa, eta_pen, eta - multiplier * (kappa - budget), float(multiplier), float(budget), temp, float(bound), model_sig, checks, d, h)

def _dual(multiplier: float, residual: float, upper: float | None):
    return abs(multiplier_step(multiplier, residual, 1.0, upper) - multiplier)

def continuing_diagnostics(e: ExactPolicyEvaluation, *, multiplier_upper: float, dual_recursion: str, actor_threshold: float=1e-05, complementarity_threshold: float=1e-06) -> dict:
    e.context.validate('continuing_rate')
    upper = positive_scalar(multiplier_upper, 'multiplier_upper')
    if e.multiplier > upper or dual_recursion not in {'rate', 'duration_scaled_numerator'}:
        raise ValueError('dual contract mismatch')
    actor_threshold = positive_scalar(actor_threshold, 'actor_threshold')
    complementarity_threshold = positive_scalar(complementarity_threshold, 'complementarity_threshold')
    r = e.update_rate - e.budget
    nu = e.mean_expenditure - e.budget * e.mean_duration
    ar = float(np.linalg.norm(actor_mapping(e.theta, e.gradient_rate, e.bound)))
    an = float(np.linalg.norm(actor_mapping(e.theta, e.mean_duration * e.gradient_rate, e.bound)))
    rr = {'actor_stationarity_rate': ar, 'actor_stationarity_numerator_field': an, 'unprojected_rate_gradient_norm': float(np.linalg.norm(e.gradient_rate)), 'rate_residual': r, 'numerator_residual': nu, 'primal_feasibility_rate': max(0.0, r), 'primal_feasibility_numerator': max(0.0, nu), 'complementarity_rate': abs(e.multiplier * r), 'complementarity_numerator': abs(e.multiplier * nu), 'dual_mapping_rate': _dual(e.multiplier, r, upper), 'dual_mapping_numerator': _dual(e.multiplier, nu, upper)}
    return {'schema_version': '1.0.0', 'context': asdict(e.context), 'identity': {'parameters_sha256': array_hash(e.theta), 'probabilities_sha256': array_hash(e.probabilities), 'model_sha256': e.model_sha256, 'temperature': e.temperature, 'coordinates': 'ambient_rows_restricted_zero_sum_box', 'projection_id': PROJECTION_ID, 'bound': e.bound, 'mapping_step': 1.0, 'gradient_objective': 'eta-lambda*kappa; same derivative as rate Lagrangian', 'gradient_source': 'exact_supplied_model'}, 'metrics': {'reward_rate': e.reward_rate, 'update_rate': e.update_rate, 'penalized_rate_eta_lambda': e.penalized_rate, 'rate_lagrangian': e.rate_lagrangian, 'mean_duration': e.mean_duration, 'mean_reward': e.mean_reward, 'mean_expenditure': e.mean_expenditure, 'budget': e.budget, 'multiplier': e.multiplier, 'multiplier_upper': upper}, 'residuals': rr, 'dual_recursion': dual_recursion, 'diagnostic_thresholds': {'actor_stationarity_rate': actor_threshold, 'complementarity_rate': complementarity_threshold, 'primal_feasibility_rate': 1e-10, 'multiplier_cap_distance': 1e-10}, 'domain_checks': domain_errors(e.theta, e.bound), 'exact_solve_checks': e.checks, 'flags': {'actor_rate_below_threshold': ar <= actor_threshold, 'feasible_with_numerical_tolerance': r <= 1e-10, 'complementarity_rate_below_threshold': abs(e.multiplier * r) <= complementarity_threshold, 'multiplier_cap_active': e.multiplier >= upper - 1e-10, 'complete_kkt_certificate': False, 'stochastic_convergence_certificate': False, 'global_optimality_certificate': False}, 'scope': 'finite same-policy diagnostic; flags are not experiment-acceptance tests'}

def episodic_diagnostics(theta: ArrayLike, expected_lagrangian_gradient: ArrayLike | None, *, expected_cost: float, expected_updates: float, expected_time: float, multiplier: float, budget: float, multiplier_upper: float, context: DiagnosticContext, bound: float=12.0) -> dict:
    context.validate('stopping_total')
    w = require_domain(theta, bound)
    scalars = [expected_cost, expected_updates, expected_time, multiplier, budget]
    if any((not np.isfinite(x) or x < 0 for x in scalars)):
        raise ValueError('invalid episodic moments')
    if expected_time == 0 and expected_updates != 0:
        raise ValueError('zero time cannot contain updates')
    upper = positive_scalar(multiplier_upper, 'multiplier_upper')
    if multiplier > upper:
        raise ValueError('multiplier above cap')
    f = expected_updates - budget * expected_time
    a = None if expected_lagrangian_gradient is None else float(np.linalg.norm(actor_mapping(w, expected_lagrangian_gradient, bound)))
    return {'schema_version': '1.0.0', 'context': asdict(context), 'identity': {'parameters_sha256': array_hash(w), 'projection_id': PROJECTION_ID, 'coordinates': 'ambient_rows_restricted_zero_sum_box', 'bound': bound, 'temperature': 1.0, 'mapping_step': 1.0, 'gradient_objective': '-E[cost]-lambda*(E[K]-Gamma*E[T_cap])', 'gradient_source': 'not_supplied' if a is None else 'caller_supplied_exact_expectation'}, 'metrics': {'expected_cost': expected_cost, 'expected_updates': expected_updates, 'expected_time': expected_time, 'budget': budget, 'multiplier': multiplier, 'multiplier_upper': upper, 'auxiliary_update_rate': None if expected_time == 0 else expected_updates / expected_time, 'undefined_rate_reason': 'zero expected plant time' if expected_time == 0 else None}, 'residuals': {'actor_stationarity_episodic': a, 'primal_total': max(0.0, f), 'episodic_constraint': f, 'complementarity_total': abs(multiplier * f), 'dual_mapping_total': _dual(multiplier, f, upper)}, 'flags': {'multiplier_cap_active': multiplier >= upper - 1e-10, 'complete_kkt_certificate': False, 'stochastic_convergence_certificate': False, 'global_optimality_certificate': False}, 'scope': 'finite stopping-objective diagnostic; not a continuing-rate theorem instance'}

def assert_same_policy_objective(*records: dict) -> None:
    if not records:
        raise ValueError('no records')

    def key(r):
        c = r['context']
        i = r['identity']
        if c['policy_mode'] != 'stochastic':
            raise ValueError('greedy policy has no stochastic-actor certificate')
        return (c['study_id'], c['protocol_sha256'], c['checkpoint_id'], c['actor_contract_id'], c['objective_id'], c['objective_family'], c['policy_mode'], c['record_kind'], i['parameters_sha256'], i.get('probabilities_sha256'), i.get('model_sha256'), i['projection_id'], i['bound'], i['temperature'], i['mapping_step'], i['gradient_objective'], r['metrics']['multiplier'], r['metrics']['budget'], r['metrics']['multiplier_upper'])
    reference = key(records[0])
    if any((key(r) != reference for r in records[1:])):
        raise ValueError('do not combine diagnostics from different policies, objectives, or scales')
