from __future__ import annotations
import time
import numpy as np
from .model import TARGET_ID, INPUTS
from .io_contract import npz, dump, key, rng
from ..common.actor_scores import softmax_rows, tabular_score
from ..common.actor_steps import episodic_actor_step, multiplier_step

def episode_metrics(T, K, A, hit, scheduled):
    T = np.asarray(T)
    K = np.asarray(K)
    A = np.asarray(A)
    hit = np.asarray(hit, dtype=bool)
    scheduled = np.asarray(scheduled)
    cost = T + 0.01 * A + 100 * ~hit
    num = K - 0.4 * T
    totals = {'T_cap': int(T.sum()), 'updates': int(K.sum()), 'input_exposure': int(A.sum()), 'scheduled_duration_total': int(scheduled.sum()), 'hits': int(hit.sum()), 'episodes': len(T)}
    out = {'hit_probability': float(hit.mean()), 'T_cap': float(T.mean()), 'updates': float(K.mean()), 'input_exposure': float(A.mean()), 'J': float(cost.mean()), 'budget_numerator': float(num.mean()), 'rate': float(K.sum() / T.sum()) if T.sum() else None, 'coded_hit_time': float(np.where(hit, T, 101).mean()), 'successful_hit_time': float(T[hit].mean()) if hit.any() else None, 'scheduled_mean_hold': float(scheduled.sum() / K.sum()) if K.sum() else None, 'realized_mean_hold': float(T.sum() / K.sum()) if K.sum() else None, 'initial_target_count': int((T == 0).sum()), 'failure_count': int((~hit).sum()), 'feasible': bool(num.mean() <= 1e-12), 'totals': totals, 'mean_individual_rate_positive_time': float((K[T > 0] / T[T > 0]).mean()) if np.any(T > 0) else None, 'per_episode_zero_denominator': 'null (initial target), excluded only from individual positive-time-ratio diagnostic'}
    return out

def evaluation_tapes(seed, checkpoint, method):
    endpoint = checkpoint == 5000
    cp = 0 if endpoint else checkpoint
    pk = [key(40 if endpoint else 30, 0, seed, 0, cp, i) for i in range(2048)]
    ak = [key(41 if endpoint else 31, method, seed, 400 if method == 5 else 0, cp, i) for i in range(2048)]
    plant = np.stack([rng(k).random((100, 2)) for k in pk])
    actor = np.stack([rng(k).random(100) for k in ak])
    return (plant, actor, pk, ak)

def evaluate_rollouts(m, parameters, method, mode, seed, checkpoint, out):
    import hashlib
    plant, actor, pk, ak = evaluation_tapes(seed, checkpoint, method)
    n = 2048
    s = np.repeat(np.arange(512, dtype=np.int16), 4)
    hit = s == TARGET_ID
    T = np.zeros(n, np.int16)
    K = np.zeros(n, np.int16)
    A = np.zeros(n, np.int16)
    scheduled = np.zeros(n, np.int16)
    states = np.full((n, 101), -1, np.int16)
    states[:, 0] = s
    decisions = np.full((n, 100), -1, np.int8)
    held = np.full((n, 100), -1, np.int8)
    left = np.zeros(n, np.int16)
    current = np.zeros(n, np.int8)
    for t in range(100):
        live = np.flatnonzero(~hit)
        new = live[left[live] == 0]
        if len(new):
            rows = parameters[100 - t, s[new]]
            if method == 4:
                act = np.argmin(rows, axis=1)
            elif mode == 'greedy':
                act = np.argmax(rows, axis=1)
            else:
                prob = softmax_rows(rows)
                U = actor[new, K[new]]
                act = (np.cumsum(prob, axis=1) <= U[:, None]).sum(axis=1)
                act = np.minimum(act, 19)
            current[new] = act
            left[new] = act % 5 + 1
            decisions[new, t] = act
            K[new] += 1
            scheduled[new] += act % 5 + 1
        held[live, t] = current[live]
        u = current[live] // 5
        A[live] += np.array([sum(x) for x in INPUTS], np.int16)[u]
        s[live] = m.branches[s[live], u, (plant[live, t, 0] >= 0.7).astype(int), (plant[live, t, 1] >= 0.6).astype(int)]
        states[live, t + 1] = s[live]
        T[live] += 1
        left[live] -= 1
        hit[live] = s[live] == TARGET_ID
        if np.all(hit):
            break
    metrics = episode_metrics(T, K, A, hit, scheduled)
    npz(out.with_suffix('.npz'), states=states, decision_actions=decisions, held_actions=held, T_cap=T, updates=K, input_exposure=A, hit=hit, scheduled_duration_total=scheduled, first_hit_time=np.where(hit, T, -1), coded_hit_time=np.where(hit, T, 101), plant_keys=np.array(pk, np.int64), actor_keys=np.array(ak, np.int64))
    record = {'evaluation_id': out.name, 'method_code': method, 'mode': mode, 'seed_label': seed, 'checkpoint_episode': checkpoint, 'endpoint': checkpoint == 5000, 'metrics': metrics, 'plant_tape_sha256': hashlib.sha256(plant.tobytes()).hexdigest(), 'actor_tape_sha256': hashlib.sha256(actor.tobytes()).hexdigest(), 'plant_tape_shape': list(plant.shape), 'actor_tape_shape': list(actor.shape), 'rollout_design': 'four per initial state0..511; equal weights; initial target retained'}
    dump(out.with_suffix('.json'), record)
    return record
EVENT_DTYPE = np.dtype([('episode', 'i4'), ('state', 'i2'), ('remaining', 'i2'), ('action', 'i1'), ('ell', 'i1'), ('next_state', 'i2'), ('hit', '?'), ('failure', '?'), ('actor_uniform', 'f8'), ('explore_uniform', 'f8'), ('td', 'f8'), ('gain', 'f8'), ('prior_visits', 'i4'), ('before_value', 'f8'), ('bootstrap_value', 'f8'), ('lambda', 'f8')])

def run_learner(m, method, seed, folder, episodes=5000, checkpoints=(1000, 2000, 3000, 4000, 5000), evaluate_checkpoints=True):
    folder.mkdir(parents=True, exist_ok=False)
    tstart = time.perf_counter()
    budget = 400 if method == 5 else 0
    resetkey = key(23, method, seed, budget)
    actkey = key(21, method, seed, budget)
    explorekey = key(25, method, seed, budget)
    reset = rng(resetkey)
    actor = rng(actkey)
    explore = rng(explorekey)
    initial_states = {'reset': reset.bit_generator.state, 'actor': actor.bit_generator.state, 'exploration': explore.bit_generator.state}
    parameters = np.zeros((101, 512, 20))
    V = np.zeros((101, 512))
    visits = np.zeros((101, 512, 20), np.int32)
    lam = 0.1
    allstates = np.full((episodes, 101), -1, np.int16)
    epdata = np.zeros((episodes, 10))
    events = []
    evals = []
    checkpoint_paths = []
    eval_seconds = 0.0
    io_seconds = 0.0
    max_coordinate = 0.0
    for e in range(1, episodes + 1):
        s = int(reset.integers(0, 512))
        allstates[e - 1, 0] = s
        plant = rng(key(22, method, seed, budget, 0, e - 1)).random((100, 2))
        t = 0
        K = 0
        A = 0
        scheduled = 0
        initial = s == TARGET_ID
        hit = initial
        frozen = lam
        changes = []
        rows = []
        before_count = len(events)
        gainfactor = (1 + e / 1000) ** (-0.6)
        ag = 0.02 * gainfactor
        cg = 0.1 * gainfactor
        dg = 0.005 * gainfactor
        while not hit and t < 100:
            r = 100 - t
            w = parameters[r, s]
            au = float(actor.random())
            eu = float(explore.random()) if method == 4 else -1.0
            if method == 4:
                eps = max(0.05, 1 - e / (0.7 * 5000))
                a = int(np.floor(20 * au)) if eu < eps else int(np.argmin(w))
            else:
                p = softmax_rows(w)
                a = min(int(np.searchsorted(np.cumsum(p), au, side='right')), 19)
            u = a // 5
            tau = a % 5 + 1
            old = s
            ell = 0
            K += 1
            scheduled += tau
            for _ in range(min(tau, r)):
                s = int(m.branches[s, u, int(plant[t, 0] >= 0.7), int(plant[t, 1] >= 0.6)])
                t += 1
                ell += 1
                A += sum(INPUTS[u])
                allstates[e - 1, t] = s
                hit = s == TARGET_ID
                if hit:
                    break
            failure = not hit and t == 100
            terminal = hit or failure
            cost = ell + 0.01 * sum(INPUTS[u]) * ell + 100 * failure
            if method == 4:
                prior = int(visits[r, old, a])
                g = 0.5 / (1 + prior) ** 0.6
                before = float(parameters[r, old, a])
                boot = 0.0 if terminal else float(parameters[100 - t, s].min())
                td = cost + boot - before
                parameters[r, old, a] += g * td
                visits[r, old, a] += 1
            else:
                prior = 0
                g = cg
                before = float(V[r, old])
                boot = 0.0 if terminal else float(V[100 - t, s])
                td = -cost - frozen * (1 - 0.4 * ell) + boot - before
                V[r, old] += g * td
                rows.append((r, old))
                changes.append(td * tabular_score(w, a))
            events.append((e, old, r, a, ell, s, hit, failure, au, eu, td, g, prior, before, boot, frozen if method == 5 else 0.0))
        if method == 5 and (not initial):
            rr, ss = np.array(rows).T
            if len(set(rows)) != len(rows):
                raise AssertionError('time-augmented pair repeated unexpectedly')
            parameters[rr, ss] = episodic_actor_step(parameters[rr, ss], np.array(changes), ag)
            lam = multiplier_step(lam, K - 0.4 * t, dg, 50)
            max_coordinate = max(max_coordinate, float(np.max(np.abs(parameters[rr, ss]))))
        epdata[e - 1] = [t, K, A, int(hit), scheduled, frozen if method == 5 else 0, lam if method == 5 else 0, e, len(events) - before_count, s]
        if e in checkpoints:
            tic = time.perf_counter()
            cp = folder / f'checkpoint_{e:04d}.npz'
            contents = {'parameters': parameters, 'lambda': np.array(lam if method == 5 else 0), 'episode': np.array(e), 'critic': V} if method == 5 else {'parameters': parameters, 'visits': visits, 'episode': np.array(e)}
            npz(cp, **contents)
            checkpoint_paths.append(cp.name)
            io_seconds += time.perf_counter() - tic
            if evaluate_checkpoints:
                tic = time.perf_counter()
                for mode in ['stochastic', 'greedy'] if method == 5 else ['greedy']:
                    evals.append(evaluate_rollouts(m, parameters, method, mode, seed, e, folder / f'eval_{e:04d}_{mode}'))
                eval_seconds += time.perf_counter() - tic
        if e % 1000 == 0:
            print('training', method, seed, e, 'steps', int(epdata[:e, 0].sum()), 'lambda', lam if method == 5 else None, flush=True)
    npz(folder / 'training.npz', states=allstates, episodes=epdata, events=np.array(events, dtype=EVENT_DTYPE))
    record = {'method': 'PD_SMAC_FA' if method == 5 else 'Macro_Q', 'method_code': method, 'seed_label': seed, 'episodes': episodes, 'actual_plant_steps': int(epdata[:, 0].sum()), 'decision_epochs': len(events), 'initial_target_episodes': int((epdata[:, 0] == 0).sum()), 'training_horizon_failures': int((epdata[:, 3] == 0).sum()), 'training_record_columns': ['T_cap', 'K', 'A', 'hit', 'scheduled_total', 'lambda_before', 'lambda_after', 'episode', 'event_count', 'terminal_state'], 'stream_keys': {'reset': resetkey, 'actor': actkey, 'exploration': explorekey, 'plant_template': key(22, method, seed, budget, 0, 0)}, 'plant_rollout_field': 'zero-based episode index; reserve100x2 uniforms each episode, unused tail never executed', 'initial_rng_states': initial_states, 'final_rng_states': {'reset': reset.bit_generator.state, 'actor': actor.bit_generator.state, 'exploration': explore.bit_generator.state}, 'lambda_endpoint': float(lam) if method == 5 else None, 'multiplier_cap_episodes': int((epdata[:, 6] >= 50).sum()) if method == 5 else None, 'max_actor_abs': max_coordinate if method == 5 else None, 'checkpoint_files': checkpoint_paths, 'evaluation_record_ids': [x['evaluation_id'] for x in evals], 'wall_seconds_total': time.perf_counter() - tstart, 'wall_seconds_evaluation': eval_seconds, 'wall_seconds_checkpoint_serialization': io_seconds, 'performance_selection': False}
    dump(folder / 'run_record.json', record)
    return record
