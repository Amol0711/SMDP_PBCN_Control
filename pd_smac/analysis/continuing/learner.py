from __future__ import annotations
import copy, sys
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .contracts import Protocol, ROOT, integer, scalar, execution_keys, digest, source_identity, verify_production_authorization
from .model import Plant, Hold, held_sample, categorical
from ...common.actor_scores import softmax_rows, tabular_score
from ...common.actor_geometry import project_zero_sum_box, projection_certificate, require_domain, FEASIBILITY_ATOL, OPTIMALITY_ATOL
XI = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
XI.setflags(write=False)

@dataclass(frozen=True)
class Gains:
    beta: float
    beta_eta: float
    alpha: float
    zeta: float

    def asdict(self):
        return dict(beta=self.beta, beta_eta=self.beta_eta, alpha=self.alpha, zeta=self.zeta)

def gains(k: int, protocol: Protocol) -> Gains:
    k = integer(k, 'global decision k')
    g = protocol.config['gains']
    z = float(1 + k)
    return Gains(g['beta0'] * z ** (-g['p_fast']), g['eta0'] * z ** (-g['p_fast']), g['alpha0'] * z ** (-g['p_slow']), g['zeta0'] * z ** (-g['p_slow']))

@dataclass
class Parameters:
    W: np.ndarray
    v: np.ndarray
    eta: float
    multiplier: float

    def clone(self):
        return Parameters(self.W.copy(), self.v.copy(), float(self.eta), float(self.multiplier))

    def serial(self):
        return {'W': self.W.tolist(), 'v': self.v.tolist(), 'eta': self.eta, 'multiplier': self.multiplier}

def validate_parameters(p: Parameters, protocol: Protocol) -> None:
    for x, shape, name in [(p.W, (4, 10), 'W'), (p.v, (3,), 'v')]:
        if not isinstance(x, np.ndarray) or x.dtype != np.dtype('float64') or x.shape != shape or (not np.all(np.isfinite(x))):
            raise ValueError('invalid ' + name)
    require_domain(p.W, protocol.config['actor']['bound'])
    eta = scalar(p.eta, 'eta')
    lam = scalar(p.multiplier, 'lambda')
    if not 0 <= lam <= protocol.config['dual']['upper_bound']:
        raise ValueError('lambda outside cap')
    limit = protocol.common['operational']['safety_max_abs_unprojected_float']
    if max(abs(eta), float(np.max(abs(p.v)))) > limit:
        raise ArithmeticError('unprojected safety failure; no clipping')

@dataclass(frozen=True)
class StepResult:
    parameters: Parameters
    delta: float
    score: np.ndarray
    gains: Gains
    dual_numerator: float
    projection_method: str
    projection_errors: dict

def projection_adapter(v: np.ndarray, bound: float) -> tuple[np.ndarray, str, dict]:
    z = np.asarray(v, dtype=np.float64)
    if z.shape != (4, 10) or not np.all(np.isfinite(z)):
        raise ArithmeticError('invalid actor trial')
    mean = z.mean(axis=1)
    centered = z - mean[:, None]
    if np.max(abs(centered)) <= bound:
        cert = projection_certificate(z, centered, mean, bound)
        x = centered
        method = 'certified_affine_interior'
    else:
        x, cert = project_zero_sum_box(z, bound, return_certificate=True)
        method = 'certified_active_box'
    errors = {name: float(getattr(cert, name)) for name in ['row_sum_error', 'bound_violation', 'normal_cone_error', 'stationarity_error', 'complementarity_error']}
    if max(errors['row_sum_error'], errors['bound_violation']) > FEASIBILITY_ATOL or max((errors[k] for k in ['normal_cone_error', 'stationarity_error', 'complementarity_error'])) > OPTIMALITY_ATOL:
        raise ArithmeticError('actor projection certificate failed')
    return (x, method, errors)

def sampled_update(p: Parameters, hold: Hold, k: int, protocol: Protocol) -> StepResult:
    validate_parameters(p, protocol)
    k = integer(k, 'k')
    s = integer(hold.state, 'state', 0, 3)
    sp = integer(hold.next_state, 'next', 0, 3)
    a = integer(hold.action, 'action', 0, 9)
    u, t = protocol.config['actor']['action_order'][a]
    if hold.duration != t or hold.input_value != u or hold.target_count not in range(t + 1):
        raise ValueError('inconsistent complete hold')
    reward = scalar(hold.reward, 'reward')
    if abs(reward - (hold.target_count - protocol.config['objective']['penalty'] * u * t)) > 1e-12:
        raise ValueError('reward is not accumulated post-transition task reward')
    g = gains(k, protocol)
    temp = protocol.config['actor']['temperature']
    score = np.zeros((4, 10))
    score[s] = tabular_score(p.W[s], a, temp)
    with np.errstate(over='raise', invalid='raise'):
        delta = float(reward - p.multiplier - p.eta * t + float(p.v @ XI[sp]) - float(p.v @ XI[s]))
        if abs(delta) > protocol.common['operational']['safety_max_abs_unprojected_float']:
            raise ArithmeticError('TD safety fail-stop; no clipping')
        new_eta = float(p.eta + g.beta_eta * delta)
        new_v = p.v + g.beta * delta * XI[s]
        trial = p.W + g.alpha * delta * score
        W, method, errors = projection_adapter(trial, protocol.config['actor']['bound'])
        nu = float(1 - protocol.config['objective']['budget'] * t)
        lam = float(np.clip(p.multiplier + g.zeta * nu, 0.0, protocol.config['dual']['upper_bound']))
    result = Parameters(W, np.asarray(new_v, dtype=np.float64), new_eta, lam)
    validate_parameters(result, protocol)
    return StepResult(result, delta, score, g, nu, method, errors)

@dataclass
class TrainingState:
    parameters: Parameters
    current_state: int
    k: int
    plant_time: int
    target_count: int
    input_exposure: int
    reward_total: float
    tail_counts: dict
    offsets: dict
    rngs: dict
    seed_keys: dict
    seed_label: int
    mode: str
    initial_state: int
    initial_uniform: float

    def serial(self) -> dict:
        return {'parameters': self.parameters.serial(), 'current_state': int(self.current_state), 'k': int(self.k), 'plant_time': int(self.plant_time), 'target_count': int(self.target_count), 'input_exposure': int(self.input_exposure), 'reward_total': float(self.reward_total), 'tail_counts': dict(self.tail_counts), 'offsets': dict(self.offsets), 'rng_states': {n: copy.deepcopy(g.bit_generator.state) for n, g in self.rngs.items()}, 'seed_keys': copy.deepcopy(self.seed_keys), 'seed_label': self.seed_label, 'mode': self.mode, 'initial_state': self.initial_state, 'initial_uniform': self.initial_uniform}

    def identity(self):
        return digest(self.serial())

class SafetyFailure(ArithmeticError):

    def __init__(self, message: str, offending: dict):
        super().__init__(message)
        self.offending = offending

def initialize(protocol: Protocol, seed: int, mode: str='functional_test', *, production_permit: dict | None=None) -> TrainingState:
    keys = execution_keys(seed, mode)
    if mode == 'production':
        if production_permit is None:
            raise PermissionError('no production draws without validated fixed_configuration activation')
        verify_production_authorization(production_permit, protocol, digest(source_identity(protocol.root)))
    rngs = {n: np.random.Generator(np.random.PCG64(np.random.SeedSequence(key))) for n, key in keys.items()}
    uniform = float(rngs['initial_state'].random())
    s, tail = categorical(np.asarray(protocol.config['training']['initial_state_law']), uniform)
    p = Parameters(np.zeros((4, 10), dtype=np.float64), np.zeros(3, dtype=np.float64), 0.0, protocol.config['dual']['lambda_initial'])
    validate_parameters(p, protocol)
    return TrainingState(p, s, 0, 0, 0, 0, 0.0, {'initial_state': int(tail), 'actor': 0, 'plant': 0}, {'initial_state': 1, 'actor': 0, 'plant': 0}, rngs, keys, seed, mode, s, uniform)

def validate_state(s: TrainingState, protocol: Protocol) -> None:
    validate_parameters(s.parameters, protocol)
    integer(s.current_state, 'state', 0, 3)
    integer(s.k, 'k')
    integer(s.plant_time, 'plant_time')
    if not s.k <= s.plant_time <= 5 * s.k:
        raise ValueError('plant count outside complete-hold bounds')
    if s.offsets != {'initial_state': 1, 'actor': s.k, 'plant': s.plant_time}:
        raise ValueError('RNG offsets differ from counters')
    if s.seed_keys != execution_keys(s.seed_label, s.mode):
        raise ValueError('wrong seed streams')
    if not 0 <= s.target_count <= s.plant_time or not 0 <= s.input_exposure <= s.plant_time:
        raise ValueError('count bound violation')
    scalar(s.reward_total, 'total reward')
    for n in s.rngs:
        if s.rngs[n].bit_generator.state['bit_generator'] != 'PCG64':
            raise ValueError('wrong bit generator')
    r = np.random.Generator(np.random.PCG64(np.random.SeedSequence(s.seed_keys['initial_state'])))
    u = float(r.random())
    initial, _ = categorical(np.asarray(protocol.config['training']['initial_state_law']), u)
    if r.bit_generator.state != s.rngs['initial_state'].bit_generator.state or u != s.initial_uniform or initial != s.initial_state:
        raise ValueError('initial stream consumed after initialization')

def advance(state: TrainingState, model: Plant, protocol: Protocol) -> tuple[np.ndarray, np.ndarray, Hold, dict]:
    p = state.parameters
    s = state.current_state
    k = state.k
    before_rng = {n: copy.deepcopy(state.rngs[n].bit_generator.state) for n in ['actor', 'plant']}
    action_uniform = None
    us = None
    try:
        probs = softmax_rows(p.W[s], protocol.config['actor']['temperature'])
        if np.any(probs <= 0):
            raise ArithmeticError('softmax full support lost')
        action_uniform = float(state.rngs['actor'].random())
        a, at = categorical(probs, action_uniform)
        _, t = model.actions[a]
        us = state.rngs['plant'].random(t)
        hold = held_sample(model, s, a, us)
        step = sampled_update(p, hold, k, protocol)
    except Exception as e:
        for n, v in before_rng.items():
            state.rngs[n].bit_generator.state = v
        detail = {'k': k, 'state': s, 'action_uniform': action_uniform, 'plant_uniforms': None if us is None else us.tolist(), 'exception_class': type(e).__name__, 'message': str(e), 'state_mutated': False, 'rng_rolled_back': True}
        raise SafetyFailure(str(e), detail) from e
    ints = np.array([k, s, a, hold.next_state, state.plant_time, t, hold.target_count, hold.input_value], dtype=np.int64)
    floats = np.array([hold.reward, step.delta, p.eta, step.parameters.eta, p.multiplier, step.parameters.multiplier, *step.parameters.v, action_uniform, float(probs[a])], dtype=np.float64)
    state.parameters = step.parameters
    state.current_state = hold.next_state
    state.k += 1
    state.plant_time += t
    state.target_count += hold.target_count
    state.input_exposure += hold.input_value * t
    state.reward_total = float(state.target_count - model.penalty * state.input_exposure)
    state.offsets['actor'] += 1
    state.offsets['plant'] += t
    state.tail_counts['actor'] += int(at)
    state.tail_counts['plant'] += hold.tail_count
    return (ints, floats, hold, {'method': step.projection_method, 'errors': step.projection_errors})
