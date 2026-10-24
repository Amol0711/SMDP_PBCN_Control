from __future__ import annotations
import argparse, csv, datetime, hashlib, importlib.metadata, json, os, platform, resource, sys, time, zipfile
from pathlib import Path
import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits
from .model import build, evaluate, deterministic
from .references import enumerate_all, solve_lp, dump
from .training import core, probs_row
from ..common.actor_residuals import DiagnosticContext, evaluate_tabular_smdp, continuing_diagnostics, array_hash
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text())
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()

def csvwrite(path, rows):
    if not rows:
        raise ValueError('empty records')
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def seed_key(purpose, seed, method=3, component=0):
    return [20260917, 12, purpose, method, seed, 300, 0, 0, component]

def tapes(seed, episodes):
    info = {}
    arrays = {}
    for name, code, n in [('reset', 23, episodes), ('actor', 21, episodes * 299), ('plant', 22, episodes * 299)]:
        key = seed_key(code, seed)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))
        before = rng.bit_generator.state
        arrays[name] = rng.random(n)
        info[name] = {'entropy_key': key, 'before': before, 'prefetched_count': n, 'prefetched_after': rng.bit_generator.state}
    info['init'] = {'entropy_key': seed_key(20, seed), 'random_draw_count': 0, 'initialization': 'W=0,h=0,eta=0,lambda=.1'}
    return (arrays, info)

def run_seed(out, m, cfg, seed, runid, protocol):
    folder = out / f'seed_{seed:02d}'
    folder.mkdir(exist_ok=False)
    start = utc()
    t0 = time.perf_counter_ns()
    a, info = tapes(seed, cfg['training']['episodes'])
    init_ns = time.perf_counter_ns() - t0
    tick = time.perf_counter_ns()
    result = core(m.plant, m.actions, m.target, a['reset'], a['actor'], a['plant'], cfg['training']['episodes'])
    training_ns = time.perf_counter_ns() - tick
    ei, ef, states, epi, epf, W, h, cf, counters, cert, extra = result
    data = dict(events_i=ei, events_f=ef, plant_states=states, episode_i=epi, episode_f=epf, W=W, h=h, checkpoint_f=cf, checkpoint_counters=counters, projection_max_residuals=cert, training_extrema=extra)
    np.savez_compressed(folder / 'training_records.npz', **data)
    consumed = {'reset': len(epi), 'actor': len(ei), 'plant': len(states)}
    for name in ['reset', 'actor', 'plant']:
        n = consumed[name]
        arr = a[name][:n]
        np.save(folder / f'{name}_tape.npy', arr, allow_pickle=False)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(info[name]['entropy_key'])))
        rng.bit_generator.advance(n)
        info[name].update({'consumed_count': n, 'consumed_after': rng.bit_generator.state, 'consumed_tape_sha256': sha(folder / f'{name}_tape.npy'), 'checkpoint_consumed_states': []})
        for cp, episode in enumerate(cfg['training']['checkpoint_episodes']):
            used = episode if name == 'reset' else int(counters[cp, 0 if name == 'actor' else 1])
            r = np.random.Generator(np.random.PCG64(np.random.SeedSequence(info[name]['entropy_key'])))
            r.bit_generator.advance(used)
            info[name]['checkpoint_consumed_states'].append({'episode': episode, 'draws': used, 'state': r.bit_generator.state})
    dump(folder / 'rng_records.json', info)
    evaluations = []
    diags = []
    ps = []
    greedy = []
    eval_ns = endpoint_ns = 0
    for cp, episode in enumerate(cfg['training']['checkpoint_episodes']):
        tick = time.perf_counter_ns()
        critic_eta, lam, temp = map(float, cf[cp])
        pid = f'{runid}:seed{seed}:episode{episode}:stochastic'
        ctx = DiagnosticContext('four_state', protocol, pid, 'four_state_tabular_tempered_v1', 'target_occupancy_minus_0.02_input_on_rate')
        e = evaluate_tabular_smdp(W[cp], m.transitions, m.rewards, m.duration, 1, multiplier=lam, budget=m.budget, context=ctx, temperature=temp)
        d = continuing_diagnostics(e, multiplier_upper=50, dual_recursion='duration_scaled_numerator')
        diag = d['residuals']
        exact = evaluate(m, e.probabilities)
        if max(abs(exact['reward'] - e.reward_rate), abs(exact['rate'] - e.update_rate)) > 1e-10:
            raise ArithmeticError('independent stochastic evaluation differs')
        ids = np.argmax(W[cp], axis=1)
        gr = evaluate(m, deterministic(ids))
        ps.append(e.probabilities.copy())
        greedy.append(ids.copy())
        diags.append(d)
        for mode, ex in [('stochastic', exact), ('greedy', gr)]:
            row = {'seed': seed, 'episode': episode, 'policy_mode': mode, 'policy_id': pid if mode == 'stochastic' else pid.replace(':stochastic', ':greedy'), 'occupancy': ex['occupancy'], 'reward': ex['reward'], 'rate': ex['rate'], 'mean_duration': ex['mean_duration'], 'exposure_rate': ex['exposure_rate'], 'rate_residual': ex['rate_residual'], 'feasible': ex['feasible'], 'temperature': temp, 'training_plant_steps': int(counters[cp, 1]), 'training_decisions': int(counters[cp, 0]), 'lambda': lam if mode == 'stochastic' else None, 'critic_eta': critic_eta if mode == 'stochastic' else None, 'actor_residual_rate': diag['actor_stationarity_rate'] if mode == 'stochastic' else None, 'actor_residual_numerator': diag['actor_stationarity_numerator_field'] if mode == 'stochastic' else None, 'primal': diag['primal_feasibility_rate'] if mode == 'stochastic' else None, 'complementarity': diag['complementarity_rate'] if mode == 'stochastic' else None, 'stationary_check_abs': ex['stationary_check_abs'], 'actions': ids.tolist() if mode == 'greedy' else None, 'stationary': ex['stationary'], 'policy': ex['policy']}
            evaluations.append(row)
        elapsed = time.perf_counter_ns() - tick
        if episode == 1200:
            endpoint_ns += elapsed
        else:
            eval_ns += elapsed
    tick = time.perf_counter_ns()
    np.savez_compressed(folder / 'policies.npz', W=W, h=h, critic_eta=cf[:, 0], multiplier=cf[:, 1], temperature=cf[:, 2], probabilities=np.array(ps), greedy_actions=np.array(greedy), episodes=np.array(cfg['training']['checkpoint_episodes']), counters=counters)
    dump(folder / 'evaluations.json', evaluations)
    with (folder / 'diagnostics.jsonl').open('w') as f:
        for d in diags:
            f.write(json.dumps(d, separators=(',', ':'), allow_nan=False) + '\n')
    episodes = [{'episode': i + 1, 'initial_state': int(r[0]), 'final_state': int(r[1]), 'plant_steps': int(r[2]), 'decisions': int(r[3]), 'occupancy_count': int(r[4]), 'reward_sum': float(epf[i, 0]), 'lambda': float(epf[i, 1]), 'critic_eta': float(epf[i, 2]), 'temperature': float(epf[i, 3])} for i, r in enumerate(epi)]
    csvwrite(folder / 'episodes.csv', episodes)
    timing = {'initialization_ns': init_ns, 'training_ns_excluding_eval_io': training_ns, 'checkpoint_evaluation_ns': eval_ns, 'endpoint_evaluation_ns': endpoint_ns, 'aggregation_ns': 0, 'serialization_ns': time.perf_counter_ns() - tick, 'end_to_end_ns': time.perf_counter_ns() - t0, 'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'clock': 'time.perf_counter_ns', 'rss_measurement_method': 'Linux getrusage ru_maxrss*1024, cumulative sequential process peak', 'timing_comparison_authorized': False}
    dump(folder / 'execution_metadata.json', {'seed': seed, 'run_id': f'{runid}_seed{seed:02d}', 'started_utc': start, 'ended_utc': utc(), 'timing': timing, 'counts': {'episodes_completed': len(epi), 'plant_steps': len(states), 'decision_epochs': len(ei), 'optimizer_updates': len(ei), 'initially_targeted_zero_step_episodes': 0}, 'projection_max_residuals': cert.tolist(), 'box_active_updates': int(extra[0]), 'maximum_lambda': float(extra[1]), 'upper_cap_updates': int(extra[2]), 'episode_plant_range': [int(epi[:, 2].min()), int(epi[:, 2].max())], 'inherited_protocol_unchanged': True})
    print(json.dumps({'seed': seed, 'plant_steps': len(states), 'decisions': len(ei), 'endpoints': [{k: r[k] for k in ['policy_mode', 'occupancy', 'reward', 'rate', 'lambda', 'actor_residual_rate']} for r in evaluations[-2:]]}), flush=True)
    return evaluations[-2:]
