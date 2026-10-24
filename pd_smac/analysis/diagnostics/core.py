from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import re
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray
from . import tabular as tabular
VERSION = '1.0.0'
NAMESPACE = 'pbcn-complementary-diagnostics'
SHA = re.compile('[0-9a-f]{64}')
RECORD_KINDS = {'synthetic_unit_test', 'synthetic_example', 'scientific_diagnostic'}
MODES = {'stochastic', 'synthetic_policy', 'unrestricted_reference', 'greedy_extraction'}
CERT_FLAGS = {'exact_zero_certified': False, 'complete_kkt_certificate': False, 'stochastic_convergence_certificate': False, 'global_optimality_certificate': False}

def digest_json(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def _scalar(x: Any, name: str, *, positive: bool=False, nonnegative: bool=False) -> float:
    if isinstance(x, (bool, np.bool_)) or not np.isscalar(x):
        raise ValueError(f'{name} must be a real finite scalar, not boolean')
    try:
        y = float(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f'invalid {name}') from e
    if not np.isfinite(y) or (positive and y <= 0) or (nonnegative and y < 0):
        raise ValueError(f'invalid {name}')
    return y

def _array(x: ArrayLike, name: str, shape: tuple | None=None) -> NDArray[np.float64]:
    raw = np.asarray(x)
    if raw.dtype.kind not in 'iuf' or not raw.size:
        raise ValueError(f'{name} must be nonempty real numerical data')
    a = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(a)) or (shape is not None and a.shape != shape):
        raise ValueError(f'invalid shape or nonfinite {name}')
    a.setflags(write=False)
    return a

def _expanded(x: ArrayLike, name: str, shape: tuple) -> NDArray[np.float64]:
    a = _array(x, name)
    try:
        return _array(np.broadcast_to(a, shape), name, shape)
    except ValueError as e:
        raise ValueError(f'{name} cannot broadcast to {shape}') from e

@dataclass(frozen=True)
class Tolerances:
    probability_sum_abs: float = 1e-12
    stationary_abs: float = 1e-10
    stationary_comparison_abs: float = 1e-10
    value_abs: float = 1e-09
    normalization_abs: float = 1e-12
    advantage_display_abs: float = 1e-10

    def validate(self) -> None:
        for k, v in asdict(self).items():
            _scalar(v, k, positive=True)

@dataclass(frozen=True)
class Context:
    study_id: str
    protocol_sha256: str
    checkpoint_id: str
    actor_contract_id: str
    objective_id: str
    policy_mode: str = 'stochastic'
    record_kind: str = 'synthetic_unit_test'
    objective_family: str = 'continuing_rate'

    def validate(self, require_continuing: bool=True) -> None:
        if any((not isinstance(v, str) or not v.strip() for v in asdict(self).values())):
            raise ValueError('context fields must be nonempty strings')
        if not SHA.fullmatch(self.protocol_sha256):
            raise ValueError('invalid protocol hash')
        if self.policy_mode not in MODES or self.record_kind not in RECORD_KINDS:
            raise ValueError('invalid policy mode or record kind')
        if self.objective_family not in {'continuing_rate', 'stopping_total', 'empirical_finite_reset'}:
            raise ValueError('unsupported objective family')
        if require_continuing and self.objective_family != 'continuing_rate':
            raise ValueError('continuing advantages cannot be applied to stopping/empirical objectives')

@dataclass(frozen=True)
class Model:
    transitions: NDArray[np.float64]
    reward: NDArray[np.float64]
    duration: NDArray[np.float64]
    expenditure: NDArray[np.float64]
    objective_id: str
    reward_unit: str
    plant_time_unit: str
    expenditure_unit: str
    state_labels: tuple[str, ...]
    action_labels: tuple[str, ...]
    model_sha256: str
    objective_sha256: str
    array_hashes: dict[str, str]

def model_from_arrays(transitions: ArrayLike, reward: ArrayLike, duration: ArrayLike, expenditure: ArrayLike, *, objective_id: str, state_labels: list[str] | tuple[str, ...] | None=None, action_labels: list[str] | tuple[str, ...] | None=None, reward_unit: str='task_reward', plant_time_unit: str='plant_step', expenditure_unit: str='initiated_update', tolerances: Tolerances=Tolerances()) -> Model:
    tolerances.validate()
    P = _array(transitions, 'transitions')
    if P.ndim != 3 or P.shape[0] != P.shape[2]:
        raise ValueError('P must have shape states,actions,states')
    n, m, _ = P.shape
    G, T, C = (_expanded(x, name, (n, m)) for x, name in [(reward, 'reward'), (duration, 'duration'), (expenditure, 'expenditure')])
    if np.any(P < 0) or np.max(np.abs(P.sum(2) - 1)) > tolerances.probability_sum_abs:
        raise ValueError('invalid stochastic macro-transition rows; no clipping or renormalization')
    if np.any(T <= 0) or np.any(C < 0):
        raise ValueError('durations positive and expenditure nonnegative required')
    strings = [objective_id, reward_unit, plant_time_unit, expenditure_unit]
    if any((not isinstance(s, str) or not s.strip() for s in strings)):
        raise ValueError('missing objective/units')
    labels = []
    for supplied, count, prefix in [(state_labels, n, 's'), (action_labels, m, 'a')]:
        lab = tuple(supplied) if supplied is not None else tuple((f'{prefix}{i}' for i in range(count)))
        if len(lab) != count or len(set(lab)) != count or any((not isinstance(s, str) or not s for s in lab)):
            raise ValueError('invalid or duplicate labels')
        labels.append(lab)
    ah = {k: tabular.array_hash(x) for k, x in zip(['P', 'G', 'T', 'C'], [P, G, T, C])}
    oh = digest_json({'objective_id': objective_id, 'reward_sha256': ah['G'], 'reward_unit': reward_unit, 'plant_time_unit': plant_time_unit})
    mh = digest_json({'arrays': ah, 'objective_sha256': oh, 'expenditure_unit': expenditure_unit, 'state_labels': labels[0], 'action_labels': labels[1]})
    return Model(P, G, T, C, *strings, *labels, mh, oh, ah)

def _check_model(m: Model, tol: Tolerances) -> None:
    current = model_from_arrays(m.transitions, m.reward, m.duration, m.expenditure, objective_id=m.objective_id, state_labels=m.state_labels, action_labels=m.action_labels, reward_unit=m.reward_unit, plant_time_unit=m.plant_time_unit, expenditure_unit=m.expenditure_unit, tolerances=tol)
    if current.model_sha256 != m.model_sha256 or current.objective_sha256 != m.objective_sha256 or current.array_hashes != m.array_hashes:
        raise ValueError('model or objective arrays modified after identity binding')

def policy_hash(pi: ArrayLike) -> str:
    return tabular.array_hash(_array(pi, 'probabilities'))

@dataclass(frozen=True)
class MultiplierBinding:
    value: float
    kind: str
    source_id: str
    policy_sha256: str
    objective_sha256: str
    model_sha256: str
    checkpoint_id: str
    policy_mode: str

def bind_multiplier(value: float, *, model: Model, probabilities: ArrayLike, context: Context, kind: str, source_id: str) -> MultiplierBinding:
    context.validate()
    if context.objective_id != model.objective_id:
        raise ValueError('objective ID does not match model')
    if kind not in {'policy_owned', 'fixed_evaluation', 'reference_value'}:
        raise ValueError('invalid multiplier provenance kind')
    if kind == 'policy_owned' and context.policy_mode not in {'stochastic', 'synthetic_policy'}:
        raise ValueError('a greedy extraction/reference cannot inherit an actor-owned multiplier')
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError('multiplier source required')
    return MultiplierBinding(_scalar(value, 'multiplier', nonnegative=True), kind, source_id, policy_hash(probabilities), model.objective_sha256, model.model_sha256, context.checkpoint_id, context.policy_mode)

def _chain_classes(P: NDArray) -> dict:
    n = len(P)
    reach = (P > 0) | np.eye(n, dtype=bool)
    for k in range(n):
        reach |= reach[:, k, None] & reach[None, k, :]
    groups = []
    left = set(range(n))
    while left:
        i = min(left)
        group = sorted((j for j in left if reach[i, j] and reach[j, i]))
        groups.append(group)
        left.difference_update(group)
    closed = [g for g in groups if all((P[i, j] == 0 for i in g for j in range(n) if j not in g))]
    return {'communicating_classes': groups, 'recurrent_classes': closed, 'irreducible': len(groups) == 1, 'unichain': len(closed) == 1, 'transient_states': [i for g in groups if g not in closed for i in g], 'graph_convention': 'strictly positive represented floating probabilities; no threshold'}

@dataclass(frozen=True)
class Evaluation:
    context: Context
    model: Model
    multiplier: MultiplierBinding
    probabilities: NDArray
    stationary: NDArray
    value: NDArray
    advantage: NDArray
    conditional_mean: NDArray
    metrics: dict
    checks: dict
    chain: dict
    reference_state: int
    tolerances: Tolerances

def evaluate_policy(model: Model, probabilities: ArrayLike, *, context: Context, multiplier: MultiplierBinding, budget: float, reference_state: int | None=None, tolerances: Tolerances=Tolerances()) -> Evaluation:
    context.validate()
    tolerances.validate()
    _check_model(model, tolerances)
    pi = _array(probabilities, 'probabilities', model.reward.shape)
    if np.any(pi < 0) or np.max(np.abs(pi.sum(1) - 1)) > tolerances.probability_sum_abs:
        raise ValueError('invalid policy rows; no clipping or renormalization')
    if context.objective_id != model.objective_id:
        raise ValueError('objective mismatch')
    expected = (policy_hash(pi), model.objective_sha256, model.model_sha256, context.checkpoint_id, context.policy_mode)
    if (multiplier.policy_sha256, multiplier.objective_sha256, multiplier.model_sha256, multiplier.checkpoint_id, multiplier.policy_mode) != expected:
        raise ValueError('multiplier/policy/objective/checkpoint mismatch')
    if multiplier.kind not in {'policy_owned', 'fixed_evaluation', 'reference_value'} or not multiplier.source_id:
        raise ValueError('invalid multiplier source')
    if multiplier.kind == 'policy_owned' and context.policy_mode not in {'stochastic', 'synthetic_policy'}:
        raise ValueError('cannot attach a parent actor multiplier to a greedy/reference policy')
    lam = _scalar(multiplier.value, 'multiplier', nonnegative=True)
    gamma = _scalar(budget, 'budget', nonnegative=True)
    P = np.einsum('sa,saj->sj', pi, model.transitions)
    chain = _chain_classes(P)
    if not chain['unichain']:
        raise ArithmeticError('multichain invariant law needs initial-class treatment; no default mixture')
    rec = chain['recurrent_classes'][0]
    n = len(P)
    if reference_state is None:
        reference_state = rec[0]
    if isinstance(reference_state, (bool, np.bool_)) or not isinstance(reference_state, (int, np.integer)) or reference_state not in rec:
        raise ValueError('value reference must be a recurrent state in the unique class')
    A = P[np.ix_(rec, rec)].T - np.eye(len(rec))
    b = np.zeros(len(rec))
    A[-1] = 1
    b[-1] = 1
    try:
        dr = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        raise ArithmeticError('invariant-law solve failed') from e
    if np.any(dr <= 0) or not np.all(np.isfinite(dr)):
        raise ArithmeticError('recurrent stationary positivity lost numerically')
    d = np.zeros(n)
    d[rec] = dr
    independent = np.linalg.lstsq(np.vstack((P.T - np.eye(n), np.ones((1, n)))), np.r_[np.zeros(n), 1], rcond=None)[0]
    checks = {'stationary_equation_abs': float(np.max(np.abs(d @ P - d))), 'stationary_sum_abs': float(abs(d.sum() - 1)), 'independent_stationary_difference_abs': float(np.max(np.abs(d - independent))), 'minimum_stationary_mass': float(d.min())}
    if checks['stationary_equation_abs'] > tolerances.stationary_abs or checks['stationary_sum_abs'] > tolerances.normalization_abs or checks['independent_stationary_difference_abs'] > tolerances.stationary_comparison_abs:
        raise ArithmeticError('invariant-law validation failed')
    w = d[:, None] * pi
    tb, gb, cb = [float(np.sum(w * x)) for x in (model.duration, model.reward, model.expenditure)]
    if tb <= 0 or not np.all(np.isfinite([tb, gb, cb])):
        raise ArithmeticError('invalid model moments')
    eta, kappa = (gb / tb, cb / tb)
    rho = eta - lam * kappa
    centered = model.reward - lam * model.expenditure - rho * model.duration
    rhs = np.sum(pi * centered, axis=1)
    H = np.eye(n) - P
    q = rhs.copy()
    H[reference_state] = 0
    H[reference_state, reference_state] = 1
    q[reference_state] = 0
    try:
        h = np.linalg.solve(H, q)
    except np.linalg.LinAlgError as e:
        raise ArithmeticError('same-policy differential-value solve failed') from e
    if not np.all(np.isfinite(h)):
        raise ArithmeticError('nonfinite value solve')
    adv = centered + np.einsum('saj,j->sa', model.transitions, h) - h[:, None]
    means = np.sum(pi * adv, axis=1)
    checks.update({'value_equation_abs': float(np.max(np.abs((np.eye(n) - P) @ h - rhs))), 'conditional_advantage_mean_abs': float(np.max(np.abs(means))), 'reference_normalization_abs': float(abs(h[reference_state]))})
    if max(checks['value_equation_abs'], checks['conditional_advantage_mean_abs']) > tolerances.value_abs or checks['reference_normalization_abs'] > tolerances.normalization_abs:
        raise ArithmeticError('same-policy Poisson/advantage centering validation failed')
    metrics = {'mean_duration': tb, 'mean_reward': gb, 'mean_expenditure': cb, 'reward_rate': eta, 'update_rate': kappa, 'penalized_rate_eta_lambda': rho, 'rate_lagrangian': eta - lam * (kappa - gamma), 'budget': gamma, 'multiplier': lam}
    if not np.all(np.isfinite(list(metrics.values()))):
        raise ArithmeticError('nonfinite rates')
    for x in [pi, d, h, adv, means]:
        x.setflags(write=False)
    return Evaluation(context, model, multiplier, pi, d, h, adv, means, metrics, checks, chain, int(reference_state), tolerances)

def scientific_value(x: float | None, tolerance: float, *, reason: str | None=None) -> dict:
    tol = _scalar(tolerance, 'display tolerance', positive=True)
    if x is None:
        if not reason:
            raise ValueError('null diagnostics require a reason')
        return {'value': None, 'scientific': None, 'absolute_display_tolerance': tol, 'zero_status': 'unavailable', 'exact_zero_certified': False, 'reason': reason}
    y = _scalar(x, 'displayed value')
    return {'value': y, 'scientific': format(y, '.16e'), 'absolute_display_tolerance': tol, 'zero_status': 'floating_zero_not_certified' if y == 0 else 'nonzero_within_display_tolerance' if abs(y) <= tol else 'nonzero_above_display_tolerance', 'exact_zero_certified': False, 'reason': None}

def _identity(e: Evaluation) -> dict:
    return {'model_sha256': e.model.model_sha256, 'objective_sha256': e.model.objective_sha256, 'probabilities_sha256': policy_hash(e.probabilities), 'model_array_hashes': e.model.array_hashes, 'state_labels': list(e.model.state_labels), 'action_labels': list(e.model.action_labels), 'invariant_convention': 'decision_epoch', 'reward_unit': e.model.reward_unit, 'plant_time_unit': e.model.plant_time_unit, 'expenditure_unit': e.model.expenditure_unit, 'parameters_sha256': None, 'temperature': None, 'temperature_hex': None, 'coordinates': None, 'projection_id': None, 'bound': None, 'mapping_step': 1.0, 'reference_state': e.reference_state, 'model_precision': 'supplied_float64'}

def _key(record: dict) -> str:
    fields = {k: record[k] for k in ['context', 'identity', 'multiplier_binding', 'metrics', 'tolerances']}
    old = record.get('tabular_record')
    fields['tabular_contract'] = None if old is None else {'dual_recursion': old['dual_recursion'], 'multiplier_upper': old['metrics']['multiplier_upper'], 'diagnostic_thresholds': old['diagnostic_thresholds']}
    return digest_json(fields)

def _record(e: Evaluation, old: dict | None=None) -> dict:
    maximum = e.advantage.max(1)
    r_adv = float(e.stationary @ maximum)
    allowance = float(e.stationary @ np.abs(e.conditional_mean))
    if np.min(maximum - e.conditional_mean) < -e.tolerances.value_abs or r_adv < -allowance - e.tolerances.value_abs:
        raise ArithmeticError('advantage maximum violates numerical centering/nonnegativity')
    r = e.metrics['update_rate'] - e.metrics['budget']
    nu = e.metrics['mean_expenditure'] - e.metrics['budget'] * e.metrics['mean_duration']
    rr = {'R_act': None, 'R_act_duration_scaled': None, 'unprojected_rate_gradient_norm': None, 'signed_excess': r, 'numerator_excess': nu, 'R_pri': max(0.0, r), 'R_comp': abs(e.multiplier.value * r), 'R_adv': r_adv}
    no_actor = 'no supported differentiable actor coordinates supplied'
    identity = _identity(e)
    if old is not None:
        r0 = old['residuals']
        rr.update({'R_act': r0['actor_stationarity_rate'], 'R_act_duration_scaled': r0['actor_stationarity_numerator_field'], 'unprojected_rate_gradient_norm': r0['unprojected_rate_gradient_norm']})
        rr.update({'signed_excess': r0['rate_residual'], 'numerator_excess': r0['numerator_residual'], 'R_pri': r0['primal_feasibility_rate'], 'R_comp': r0['complementarity_rate']})
        identity.update({k: old['identity'][k] for k in ['parameters_sha256', 'temperature', 'coordinates', 'projection_id', 'bound', 'mapping_step']})
        identity['temperature_hex'] = float(identity['temperature']).hex()
        no_actor = None
    scopes = {**CERT_FLAGS, 'actor_coordinate_independent_R_adv': True, 'reward_scale_invariant_R_adv': False, 'compact_logit_KKT_residual_R_adv': False, 'all_states_positive_represented_stationary_mass': bool(np.all(e.stationary > 0)), 'full_action_support': bool(np.all(e.probabilities > 0)), 'zero_mass_states': np.flatnonzero(e.stationary == 0).tolist(), 'minimum_stationary_mass': float(e.stationary.min()), 'numerical_not_interval_certified': True, 'optional_performance_bound_computed': False}
    displays = {k: scientific_value(v, 1e-05 if k.startswith('R_act') else 1e-06 if k == 'R_comp' else e.tolerances.advantage_display_abs, reason=no_actor if v is None else None) for k, v in rr.items()}
    result = {'schema_version': VERSION, 'schema_namespace': NAMESPACE, 'status': 'available', 'context': asdict(e.context), 'identity': identity, 'multiplier_binding': asdict(e.multiplier), 'metrics': e.metrics, 'residuals': rr, 'display': displays, 'tolerances': asdict(e.tolerances), 'exact_evaluation': {'probabilities': e.probabilities.tolist(), 'stationary': e.stationary.tolist(), 'differential_value': e.value.tolist(), 'advantage': e.advantage.tolist(), 'conditional_advantage_means': e.conditional_mean.tolist(), 'statewise_maximum_advantage': maximum.tolist(), 'weighted_centering_allowance': allowance, 'checks': e.checks, 'chain': e.chain}, 'tabular_record': old, 'scope': scopes, 'unavailable_reason': None}
    result['evaluation_key'] = _key(result)
    return result

def diagnose_policy(model: Model, probabilities: ArrayLike, **kwargs: Any) -> dict:
    return _record(evaluate_policy(model, probabilities, **kwargs))

def diagnose_tabular(model: Model, theta: ArrayLike, *, temperature: float, bound: float, context: Context, multiplier: MultiplierBinding, budget: float, multiplier_upper: float, dual_recursion: str, reference_state: int | None=None, tolerances: Tolerances=Tolerances()) -> dict:
    if context.policy_mode != 'stochastic':
        raise ValueError('tabular adapter requires stochastic actor identity')
    W = tabular.require_domain(theta, bound)
    temp = _scalar(temperature, 'temperature', positive=True)
    pi = tabular.softmax_rows(W, temp)
    if np.any(pi <= 0):
        raise ArithmeticError('full-support softmax underflow; no deterministic substitution')
    e = evaluate_policy(model, pi, context=context, multiplier=multiplier, budget=budget, reference_state=reference_state, tolerances=tolerances)
    kind = 'synthetic_unit_test' if context.record_kind in {'synthetic_unit_test', 'synthetic_example'} else 'scientific_diagnostic'
    cx = tabular.DiagnosticContext(context.study_id, context.protocol_sha256, context.checkpoint_id, context.actor_contract_id, context.objective_id, record_kind=kind)
    old_e = tabular.evaluate_tabular_smdp(W, model.transitions, model.reward, model.duration, model.expenditure, multiplier=multiplier.value, budget=budget, context=cx, temperature=temp, bound=bound, reference_state=e.reference_state)
    if np.max(abs(old_e.stationary - e.stationary)) > tolerances.stationary_comparison_abs or np.max(abs(old_e.value - e.value)) > tolerances.value_abs:
        raise ArithmeticError('independent evaluator disagrees with preserved tabular_diagnostic component')
    for name in ['mean_duration', 'mean_reward', 'mean_expenditure', 'reward_rate', 'update_rate']:
        if abs(getattr(old_e, name) - e.metrics[name]) > tolerances.value_abs:
            raise ArithmeticError('tabular_diagnostic same-policy metric discrepancy')
    old = tabular.continuing_diagnostics(old_e, multiplier_upper=multiplier_upper, dual_recursion=dual_recursion)
    return _record(e, old)

def unavailable_record(*, context: Context, reason: str) -> dict:
    context.validate(require_continuing=False)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('a material unavailability reason is required')
    fields = ['R_act', 'R_act_duration_scaled', 'unprojected_rate_gradient_norm', 'signed_excess', 'numerator_excess', 'R_pri', 'R_comp', 'R_adv']
    return {'schema_version': VERSION, 'schema_namespace': NAMESPACE, 'status': 'unavailable', 'context': asdict(context), 'identity': None, 'multiplier_binding': None, 'metrics': None, 'residuals': {k: None for k in fields}, 'display': {k: scientific_value(None, 1e-10, reason=reason) for k in fields}, 'tolerances': asdict(Tolerances()), 'exact_evaluation': None, 'tabular_record': None, 'scope': None, 'unavailable_reason': reason, 'evaluation_key': None}

def assert_same_evaluation(*records: dict) -> None:
    if not records or any((r.get('status') != 'available' for r in records)):
        raise ValueError('same-evaluation comparison requires available records')
    for r in records:
        if r['evaluation_key'] != _key(r):
            raise ValueError('diagnostic identity was modified')
    if len({r['evaluation_key'] for r in records}) != 1:
        raise ValueError('do not combine different policies, objectives, temperatures or multipliers')

def _finite_json(x: Any) -> None:
    if isinstance(x, float) and (not np.isfinite(x)):
        raise ValueError('JSON numerical values must be finite')
    if isinstance(x, dict):
        for y in x.values():
            _finite_json(y)
    if isinstance(x, list):
        for y in x:
            _finite_json(y)

def validate_record(record: dict, *, schema_path: Path | None=None) -> None:
    import jsonschema
    _finite_json(record)
    schema = json.loads((schema_path or Path(__file__).with_name('diagnostic_record.schema.json')).read_text())
    jsonschema.Draft202012Validator(schema).validate(record)
    Tolerances(**record['tolerances']).validate()
    if record['status'] == 'unavailable':
        if any((v is not None for v in record['residuals'].values())):
            raise ValueError('unavailable is not zero')
        for z in record['display'].values():
            if z != scientific_value(None, z['absolute_display_tolerance'], reason=record['unavailable_reason']):
                raise ValueError('unavailable display must retain its explicit reason and null value')
        return
    r = record
    e = r['exact_evaluation']
    m = r['metrics']
    i = r['identity']
    b = r['multiplier_binding']
    assert_same_evaluation(r)
    pi = np.array(e['probabilities'])
    d = np.array(e['stationary'])
    A = np.array(e['advantage'])
    if pi.ndim != 2 or d.shape != (len(pi),) or A.shape != pi.shape or np.any(pi < 0) or np.any(d < 0):
        raise ValueError('invalid probability/advantage dimensions')
    if policy_hash(pi) != i['probabilities_sha256'] or b['policy_sha256'] != i['probabilities_sha256'] or b['objective_sha256'] != i['objective_sha256'] or (b['model_sha256'] != i['model_sha256']) or (b['checkpoint_id'] != r['context']['checkpoint_id']) or (b['policy_mode'] != r['context']['policy_mode']) or (b['value'] != m['multiplier']):
        raise ValueError('bound identity mismatch')
    if r['scope']['all_states_positive_represented_stationary_mass'] != bool(np.all(d > 0)) or r['scope']['full_action_support'] != bool(np.all(pi > 0)) or r['scope']['zero_mass_states'] != np.flatnonzero(d == 0).tolist() or (r['scope']['minimum_stationary_mass'] != float(d.min())):
        raise ValueError('invariant-mass/support qualification mismatch')
    if b['kind'] == 'policy_owned' and r['context']['policy_mode'] not in {'stochastic', 'synthetic_policy'}:
        raise ValueError('actor multiplier transferred to a greedy/reference record')
    t = r['tolerances']
    tol = t['value_abs']
    if np.max(abs(pi.sum(1) - 1)) > t['probability_sum_abs'] or abs(d.sum() - 1) > t['normalization_abs']:
        raise ValueError('non-normalized probabilities')
    checks = [(float(d @ A.max(1)), r['residuals']['R_adv']), (m['mean_reward'] / m['mean_duration'], m['reward_rate']), (m['mean_expenditure'] / m['mean_duration'], m['update_rate']), (m['reward_rate'] - m['multiplier'] * m['update_rate'], m['penalized_rate_eta_lambda']), (m['reward_rate'] - m['multiplier'] * (m['update_rate'] - m['budget']), m['rate_lagrangian']), (m['mean_expenditure'] - m['budget'] * m['mean_duration'], r['residuals']['numerator_excess']), (m['update_rate'] - m['budget'], r['residuals']['signed_excess']), (max(0.0, m['update_rate'] - m['budget']), r['residuals']['R_pri']), (abs(m['multiplier'] * (m['update_rate'] - m['budget'])), r['residuals']['R_comp'])]
    if any((abs(a - bb) > tol for a, bb in checks)):
        raise ValueError('inconsistent objective/rate/residual')
    if np.max(abs((pi * A).sum(1) - np.array(e['conditional_advantage_means']))) > tol or np.max(abs((pi * A).sum(1))) > tol:
        raise ValueError('rowwise advantage centering fails')
    if np.max(abs(A.max(1) - np.array(e['statewise_maximum_advantage']))) > tol:
        raise ValueError('statewise maxima inconsistent')
    for k, v in r['residuals'].items():
        z = r['display'][k]
        expected = scientific_value(v, z['absolute_display_tolerance'], reason=z['reason'])
        if z != expected:
            raise ValueError('rounded display/zero classification mismatch')
    if r['tabular_record'] is None:
        if any((r['residuals'][k] is not None for k in ['R_act', 'R_act_duration_scaled', 'unprojected_rate_gradient_norm'])):
            raise ValueError('unsupported actor diagnostics must remain null')
        if any((i[k] is not None for k in ['parameters_sha256', 'temperature', 'temperature_hex', 'coordinates', 'projection_id', 'bound'])):
            raise ValueError('policy-only evaluation cannot invent actor coordinates')
    if r['tabular_record'] is not None:
        old = r['tabular_record']
        tabular_schema = json.loads((tabular.COMMON_PARENT / 'common/actor_diagnostic_record.schema.json').read_text())
        jsonschema.Draft202012Validator(tabular_schema).validate(old)
        for new, previous in [('R_act', 'actor_stationarity_rate'), ('R_act_duration_scaled', 'actor_stationarity_numerator_field'), ('unprojected_rate_gradient_norm', 'unprojected_rate_gradient_norm'), ('signed_excess', 'rate_residual'), ('numerator_excess', 'numerator_residual'), ('R_pri', 'primal_feasibility_rate'), ('R_comp', 'complementarity_rate')]:
            if r['residuals'][new] != old['residuals'][previous]:
                raise ValueError('preserved residual changed')
        expectedtabular_model = hashlib.sha256('|'.join((i['model_array_hashes'][k] for k in ['P', 'G', 'T', 'C'])).encode()).hexdigest()
        if old['identity']['model_sha256'] != expectedtabular_model:
            raise ValueError('tabular model arrays do not match objective-bound sidecar')
        if old['identity']['probabilities_sha256'] != i['probabilities_sha256'] or old['context']['objective_id'] != r['context']['objective_id'] or old['context']['checkpoint_id'] != r['context']['checkpoint_id'] or (old['metrics']['multiplier'] != b['value']) or (old['metrics']['budget'] != m['budget']):
            raise ValueError('tabular residuals from another policy/objective')
        for k in ['study_id', 'protocol_sha256', 'checkpoint_id', 'actor_contract_id', 'objective_id', 'objective_family', 'policy_mode']:
            if old['context'][k] != r['context'][k]:
                raise ValueError('tabular context/coordinate ownership mismatch')
        expected_kind = 'synthetic_unit_test' if r['context']['record_kind'] in {'synthetic_unit_test', 'synthetic_example'} else 'scientific_diagnostic'
        if old['context']['record_kind'] != expected_kind:
            raise ValueError('tabular synthetic/scientific classification mismatch')
        if old['diagnostic_thresholds'] != {'actor_stationarity_rate': 1e-05, 'complementarity_rate': 1e-06, 'primal_feasibility_rate': 1e-10, 'multiplier_cap_distance': 1e-10}:
            raise ValueError('this fixed_configuration adapter preserves the original tabular_diagnostic diagnostic thresholds')
        if old['metrics']['multiplier_upper'] < m['multiplier']:
            raise ValueError('multiplier exceeds its declared cap')
        for k in ['parameters_sha256', 'temperature', 'projection_id', 'coordinates', 'mapping_step', 'bound']:
            if i[k] != old['identity'][k]:
                raise ValueError('tabular actor-coordinate mismatch')
        if i['temperature_hex'] != float(i['temperature']).hex():
            raise ValueError('temperature decimal/hex mismatch')
