from __future__ import annotations
from dataclasses import replace
import sys
import numpy as np
from .contracts import ROOT, Protocol
from .learner import TrainingState
from .model import Plant, macro_arrays
from ..diagnostics import core as dc

def evaluate_checkpoint(state: TrainingState, plant: Plant, protocol: Protocol, *, include_greedy: bool=True) -> dict:
    before = state.identity()
    try:
        mac = macro_arrays(plant)
        model = dc.model_from_arrays(mac['Q'], mac['reward'], mac['duration'], mac['expenditure'], objective_id='occupancy_minus_0.02_exposure_rate', state_labels=['S', 'B', 'O', 'L'], action_labels=[f'u{u}_tau{t}' for u, t in plant.actions])
        checkpoint = f'{state.mode}__seed{state.seed_label}__k{state.k}__{before[:16]}'
        cx = dc.Context('four_state_continuing', protocol.sha256, checkpoint, 'four_state_reference_indicator_fixedT1_C12_v1', model.objective_id, record_kind='synthetic_unit_test' if state.mode == 'functional_test' else 'scientific_diagnostic')
        W = state.parameters.W.copy()
        pi = dc.tabular.softmax_rows(W, 1.0)
        bind = dc.bind_multiplier(state.parameters.multiplier, model=model, probabilities=pi, context=cx, kind='policy_owned', source_id='saved_checkpoint:' + before)
        record = dc.diagnose_tabular(model, W, temperature=1.0, bound=12.0, context=cx, multiplier=bind, budget=0.3, multiplier_upper=50.0, dual_recursion='duration_scaled_numerator', reference_state=0)
        dc.validate_record(record)
        h = np.array(record['exact_evaluation']['differential_value'])
        rho = record['metrics']['penalized_rate'] if 'penalized_rate' in record['metrics'] else record['metrics']['reward_rate'] - state.parameters.multiplier * record['metrics']['update_rate']
        d = np.array(record['exact_evaluation']['stationary'])
        weights = d[:, None] * pi
        tracking = {'rate_estimate': state.parameters.eta, 'exact_penalized_rate': rho, 'rate_abs_error': abs(state.parameters.eta - rho), 'v': state.parameters.v.tolist(), 'h_reference_normalized': h.tolist(), 'value_norm2_error': float(np.linalg.norm(state.parameters.v - h[1:])), 'value_norminf_error': float(np.max(abs(state.parameters.v - h[1:]))), 'finite_error_not_convergence_certificate': True}
        moments = {'occupancy': float(np.sum(weights * mac['occupancy']) / np.sum(weights * mac['duration'])), 'input_exposure_rate': float(np.sum(weights * (mac['duration'] * np.array([u for u, t in plant.actions])[None, :])) / np.sum(weights * mac['duration']))}
        result = {'checkpoint_id': checkpoint, 'training_state_sha256': before, 'record_scope': state.mode, 'stochastic': record, 'critic_tracking': tracking, 'stochastic_auxiliary_metrics': moments, 'greedy': None, 'training_rng_consumption': 0}
        if include_greedy:
            greedy = np.eye(10)[np.argmax(pi, axis=1)]
            gc = replace(cx, checkpoint_id=checkpoint + '__greedy', policy_mode='greedy_extraction')
            gb = dc.bind_multiplier(0.0, model=model, probabilities=greedy, context=gc, kind='fixed_evaluation', source_id='explicit_zero_not_parent_multiplier')
            gr = dc.diagnose_policy(model, greedy, context=gc, multiplier=gb, budget=0.3, reference_state=0)
            dc.validate_record(gr)
            result['greedy'] = gr
        return result
    finally:
        if state.identity() != before:
            raise RuntimeError('evaluation mutated training state or consumed RNG')
