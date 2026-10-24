from __future__ import annotations
import argparse, hashlib, json, os, platform, resource, sys, time
from pathlib import Path
from .model import Model
import numpy as np
import psutil, importlib.metadata
import torch
from torch import nn
CONFIG = json.loads(Path(__file__).with_name('config.json').read_text())
COMMON = json.loads((Path(__file__).resolve().parents[1] / 'common/config.json').read_text())
METHOD = {'fixed': 0, 'DDQN_PER': 1, 'TT_AC': 2, 'PD_SMAC': 3}
E = CONFIG['training']['episodes']
H = CONFIG['training']['episode_horizon']
N = CONFIG['model']['state_dimension']
NR = CONFIG['evaluation']['rollouts_per_seed']
A = len(CONFIG['model']['action_pairs'])
PURPOSE = COMMON['seeds']['purpose_codes']
PAIRS = np.array(CONFIG['model']['action_pairs'], dtype=np.int16)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False, default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist()) + '\n')

def arrhash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def key(purpose, method=0, seed=0, budget=0, checkpoint=0, rollout=0, component=0):
    return [20260917, 15, PURPOSE[purpose], METHOD[method] if isinstance(method, str) else method, int(seed), round(1000 * budget), int(checkpoint), int(rollout), int(component)]

def rng(k):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(k)))

def integer_seed(k):
    return int(np.random.SeedSequence(k).generate_state(1, dtype=np.uint64)[0]) % 2 ** 63

def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    if not np.isfinite(p).all() or min(p) < 0 or abs(p.sum() - 1) > 1e-06:
        raise ArithmeticError('Invalid probability row')
    return p / p.sum()

def categorical(p, u):
    return int(np.searchsorted(np.cumsum(normalize(p)), u, side='right'))

def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

class QNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)

def replay_sample(prior, beta, uniform):
    pr = np.asarray(prior, dtype=np.float64)
    if len(pr) == 0 or not np.isfinite(pr).all() or np.any(pr <= 0):
        raise ValueError('PER priorities must be positive and finite')
    u = np.asarray(uniform, dtype=np.float64)
    if not np.isfinite(u).all() or np.any(u < 0) or np.any(u >= 1):
        raise ValueError('PER uniforms outside [0,1)')
    prob = pr ** 0.6
    prob /= prob.sum()
    ix = np.searchsorted(np.cumsum(prob), u, side='right')
    weights = (len(pr) * prob[ix]) ** (-beta)
    return (ix, prob, (weights / weights.max()).astype(np.float32))

def update_priorities(prior, indices, pre_update_td):
    for index in np.unique(indices):
        prior[int(index)] = float(np.max(np.abs(pre_update_td[indices == index])) + 0.001)

def double_q_target(online, target_net, next_state, reward, terminal):
    with torch.no_grad():
        selected = torch.argmax(online(next_state), dim=1)
        return reward + 0.98 * (1 - terminal) * target_net(next_state).gather(1, selected[:, None]).squeeze(1)

def configure_cpu():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

def policy(method, mode, state, model, actor, q=None):
    phi = model.features(state)
    if method == 'fixed':
        u, tau = map(int, actor)
        return (u, tau, np.array([1.0]), None)
    if method == 'DDQN_PER':
        with torch.no_grad():
            z = q(torch.from_numpy(state.astype(np.float32))[None, :]).numpy()[0]
        u = int(np.argmax(z))
        return (u, 1, np.eye(2)[u], z)
    if method == 'TT_AC':
        z = actor @ phi
        p = sig(z)
        probs = np.array([1 - p, p])
        u = int(p >= 0.5)
        return (u, 1, probs, np.array([z]))
    z = actor @ phi
    probs = softmax(z)
    a = int(np.argmax(z))
    u, tau = PAIRS[a]
    return (int(u), int(tau), probs, z)

def make_eval_tapes(out, model):
    tape_dir = out / 'tapes'
    tape_dir.mkdir()
    meta = []
    tapes = {}
    for ep in CONFIG['training']['checkpoint_episodes']:
        phase = 'endpoint' if ep == E else 'checkpoint'
        coord = 0 if ep == E else ep
        for seed in CONFIG['training']['seed_labels']:
            plant = np.empty((NR, H, N))
            reset = np.empty(NR)
            keys = []
            for j in range(NR):
                pk = key(phase + '_plant', seed=seed, checkpoint=coord, rollout=j)
                rk = key(phase + '_reset', seed=seed, checkpoint=coord, rollout=j)
                plant[j] = rng(pk).random((H, N))
                reset[j] = rng(rk).random()
                keys.append({'rollout': j, 'plant': pk, 'reset': rk})
            name = f'e{ep:03d}_seed{seed:02d}'
            np.savez_compressed(tape_dir / (name + '.npz'), plant=plant, reset=reset)
            rec = {'tape_id': name, 'checkpoint_episode': ep, 'seed_label': seed, 'keys': keys, 'plant_array_sha256': arrhash(plant), 'reset_array_sha256': arrhash(reset), 'artifact_sha256': sha(tape_dir / (name + '.npz'))}
            meta.append(rec)
            tapes[ep, seed] = (plant, reset, name)
    dump(tape_dir / 'index.json', meta)
    return tapes
METRICS = ['hit_probability', 'coded_hitting_time', 'occupancy', 'reward', 'rate', 'scheduled_hold', 'realized_hold', 'input_on_fraction']

def summarize(raw, budget):
    n = len(raw['counts'])
    c = raw['counts'].astype(np.float64)
    T, K, S, on, occ, hit, coded = c.T
    vals = {'hit_probability': float(hit.mean()), 'coded_hitting_time': float(coded.mean()), 'occupancy': float(occ.sum() / T.sum()), 'reward': float((occ.sum() - 0.06 * on.sum()) / T.sum()), 'rate': float(K.sum() / T.sum()), 'scheduled_hold': float(S.sum() / K.sum()), 'realized_hold': float(T.sum() / K.sum()), 'input_on_fraction': float(on.sum() / T.sum())}
    vals['residual'] = vals['rate'] - budget
    vals['empirical_feasible'] = vals['residual'] <= 1e-12
    return vals

def evaluate(folder, ep, seed, method, budget, actor, model, tapes, q=None, mode='greedy'):
    folder.mkdir(exist_ok=True, parents=True)
    plant, reset, tapeid = tapes[ep, seed]
    states = np.empty((NR, H + 1, N), dtype=np.int8)
    inputs = np.empty((NR, H), dtype=np.int8)
    counts = np.zeros((NR, 7), dtype=np.int64)
    action_u = np.empty((NR, H))
    keys = []
    phase = 'endpoint' if ep == E else 'checkpoint'
    coord = 0 if ep == E else ep
    for j in range(NR):
        ak = key(phase + '_actor', method, seed, budget if method == 'PD_SMAC' else 0, coord, j)
        action_u[j] = rng(ak).random(H)
        keys.append(ak)
    decisions = []
    probs_all = []
    logits_all = []
    visited = {}
    clip = {'evaluated_decisions': 0, 'abs_logit_gt_30': 0, 'abs_logit_eq_30': 0, 'probability_zero_or_one': 0}
    maxabs = 0.0
    for j in range(NR):
        s, reset_idx = model.reset(reset[j])
        states[j, 0] = s
        t = 0
        k = 0
        scheduled = 0
        hit = 0
        coded = H + 1
        while t < H:
            u, tau, probs, z = policy(method, mode, s, model, actor, q)
            if method == 'TT_AC':
                p = probs[1]
                clip['evaluated_decisions'] += 1
                clip['abs_logit_gt_30'] += int(abs(z[0]) > 30)
                clip['abs_logit_eq_30'] += int(abs(z[0]) == 30)
                clip['probability_zero_or_one'] += int(p == 0 or p == 1)
                maxabs = max(maxabs, abs(float(z[0])))
            if mode == 'stochastic':
                a = categorical(probs, action_u[j, k])
                if method == 'TT_AC':
                    u = a
                    tau = 1
                else:
                    u, tau = map(int, PAIRS[a])
            else:
                a = int(np.argmax(z)) if method in ('PD_SMAC', 'DDQN_PER') else u
            ell = min(tau, H - t)
            skey = ''.join(map(str, s.tolist()))
            gu, gt, _, _ = policy(method, 'greedy', s, model, actor, q)
            visited[skey] = [gu, gt]
            for l in range(ell):
                s = model.step(s, u, plant[j, t + l])
                states[j, t + l + 1] = s
                inputs[j, t + l] = u
                if s[1] == 0 and (not hit):
                    hit = 1
                    coded = t + l + 1
            R = float(np.sum(states[j, t + 1:t + ell + 1, 1] == 0) - 0.06 * u * ell)
            decisions.append([j, k, t, a, u, tau, ell, R])
            probs_all.append(np.pad(normalize(probs), (0, A - len(probs)), constant_values=0))
            logits_all.append(np.pad(np.asarray(z if z is not None else [0.0], dtype=np.float64), (0, A - (len(z) if z is not None else 1)), constant_values=0))
            t += ell
            k += 1
            scheduled += tau
        counts[j] = [H, k, scheduled, int(inputs[j].sum()), int((states[j, 1:, 1] == 0).sum()), hit, coded]
    raw = {'states': states, 'inputs': inputs, 'counts': counts, 'decisions': np.array(decisions, dtype=np.float64), 'probabilities': np.array(probs_all), 'logits': np.array(logits_all), 'actor_uniform': action_u}
    filename = f'e{ep:03d}_{mode}'
    np.savez_compressed(folder / (filename + '.npz'), **raw)
    rec = {'evaluation_id': folder.parent.name + '__' + filename, 'checkpoint_id': folder.parent.name + f'__e{ep:03d}', 'method': method, 'policy_mode': mode, 'seed_label': seed, 'Gamma': budget, 'checkpoint_episode': ep, 'terminal_evaluation': ep == E, 'rollouts': NR, 'horizon': H, 'tape_id': tapeid, 'actor_keys': keys, 'actor_draw_sha256': arrhash(action_u), 'raw_file': filename + '.npz', 'raw_sha256': sha(folder / (filename + '.npz')), 'metrics': summarize(raw, budget), 'feature_signature': model.signature, 'actor_clip_audit': {**clip, 'max_abs_preclip_logit': maxabs}, 'stationarity': None, 'stationarity_null_reason': 'No exact gradient for this compact-feature finite-reset objective; no theorem certificate', 'matching_multiplier': None if mode != 'stochastic' or method != 'PD_SMAC' else 'checkpoint lambda, not a greedy certificate'}
    dump(folder / (filename + '.json'), rec)
    dump(folder / (filename + '_greedy_map.json'), {'scope': 'states visited in this evaluation; weights define the total-domain policy', 'evaluation_id': rec['evaluation_id'], 'map': visited})
    return rec
DECISION_COLUMNS = ['episode', 'start_time', 'decision_within_episode', 'action_index', 'u', 'scheduled', 'executed', 'reward', 'TD', 'eta_before', 'eta_after', 'lambda_before', 'lambda_after', 'actor_uniform', 'chosen_probability']

def train(out, method, seed, budget, model, tapes):
    name = f'{method}_seed{seed:02d}_g{round(1000 * budget):03d}'
    folder = out / name
    folder.mkdir()
    (folder / 'checkpoints').mkdir()
    t0 = time.perf_counter_ns()
    keys = {p: key(p, method, seed, budget) for p in ['train_init', 'train_actor', 'train_plant', 'train_reset', 'train_replay', 'train_exploration']}
    generators = {p: rng(k) for p, k in keys.items()}
    initial = {p: g.bit_generator.state for p, g in generators.items()}
    plant = generators['train_plant'].random((E, H, N))
    reset = generators['train_reset'].random(E)
    np.savez_compressed(folder / 'training_tapes.npz', plant=plant, reset=reset)
    v = np.zeros(model.dim, dtype=np.float32)
    actor = np.zeros((A, model.dim) if method == 'PD_SMAC' else (model.dim,), dtype=np.float32)
    eta = 0.0
    lam = 0.1 if method == 'PD_SMAC' else 0.0
    q = qt = opt = None
    torchkey = key('train_init', method, seed, budget, component=1)
    if method == 'DDQN_PER':
        torch.manual_seed(integer_seed(torchkey))
        q = QNet()
        qt = QNet()
        qt.load_state_dict(q.state_dict())
        opt = torch.optim.Adam(q.parameters(), lr=0.001)
    states = np.empty((E, H + 1, N), dtype=np.int8)
    inputs = np.empty((E, H), dtype=np.int8)
    decisions = []
    prerows = []
    qreplay = []
    replay_uniform = []
    exploration = []
    updates = 0
    dataidx = []
    prior = []
    pos = 0
    counts = np.empty((E, 7), dtype=np.int64)
    evals = []
    checkpoints = []
    train_ns = 0
    eval_ns = 0
    endpoint_ns = 0
    serialize_ns = 0
    init_ns = time.perf_counter_ns() - t0
    clip = {'count': 0, 'saturated': 0, 'kink': 0, 'rounded': 0, 'max_abs': 0.0}
    for ep in range(1, E + 1):
        clock = time.perf_counter_ns()
        s, _ = model.reset(reset[ep - 1])
        states[ep - 1, 0] = s
        t = 0
        k = 0
        scheduled = 0
        hit = 0
        coded = H + 1
        while t < H:
            phi = model.features(s)
            eb = float(eta)
            lb = float(lam)
            draw = float(generators['train_actor'].random()) if method != 'DDQN_PER' else 0.0
            if method == 'PD_SMAC':
                probs = softmax(actor @ phi)
                a = categorical(probs, draw)
                u, tau = map(int, PAIRS[a])
                prerows.append(np.r_[probs, np.zeros(0)])
            elif method == 'TT_AC':
                z = actor @ phi
                p = sig(z)
                probs = np.array([1 - p, p])
                a = categorical(probs, draw)
                u = a
                tau = 1
                prerows.append(np.r_[probs, np.zeros(A - 2)])
                clip['count'] += 1
                clip['saturated'] += int(abs(z) > 30)
                clip['kink'] += int(abs(z) == 30)
                clip['rounded'] += int(p == 0 or p == 1)
                clip['max_abs'] = max(clip['max_abs'], abs(float(z)))
            else:
                eps = max(0.05, 1 - ep / (0.7 * E))
                ex = generators['train_exploration'].random(2)
                exploration.append(ex)
                with torch.no_grad():
                    qs = q(torch.from_numpy(s.astype(np.float32))[None, :]).numpy()[0]
                a = int(ex[1] * 2) if ex[0] < eps else int(np.argmax(qs))
                u = a
                tau = 1
                probs = np.eye(2)[u]
                prerows.append(np.r_[qs, np.zeros(A - 2)])
            ell = min(tau, H - t)
            R = 0.0
            for l in range(ell):
                s = model.step(s, u, plant[ep - 1, t + l])
                states[ep - 1, t + l + 1] = s
                inputs[ep - 1, t + l] = u
                R += (1.0 if s[1] == 0 else 0.0) - 0.06 * u
                if s[1] == 0 and (not hit):
                    hit = 1
                    coded = t + l + 1
            if method != 'DDQN_PER':
                delta = R - lam - eta * ell + v @ model.features(s) - v @ phi
                if method == 'TT_AC':
                    eta += 0.012 * delta
                    v += 0.04 * delta * phi
                    actor += 0.01 * delta * (u - p) * phi
                else:
                    eta += 0.008 * delta
                    v += 0.02 * delta * phi
                    grad = -probs[:, None] * phi[None, :]
                    grad[a] += phi
                    actor += 0.004 * delta * grad
                    lam = max(0.0, lam + 0.008 * (1.0 - budget * ell))
                if not (np.isfinite(actor).all() and np.isfinite(v).all() and np.isfinite(eta) and np.isfinite(lam)):
                    raise ArithmeticError('Nonfinite actor-critic run ' + name)
            else:
                delta = 0.0
                stepid = (ep - 1) * H + t
                newp = max(prior) if prior else 1.0
                if len(dataidx) < 5000:
                    dataidx.append(stepid)
                    prior.append(newp)
                else:
                    dataidx[pos] = stepid
                    prior[pos] = newp
                    pos = (pos + 1) % 5000
                if len(dataidx) >= 32:
                    beta = min(1.0, 0.4 + 0.6 * ep / E)
                    ru = generators['train_replay'].random(32)
                    ix, prob, weights = replay_sample(prior, beta, ru)
                    di = np.asarray(dataidx)[ix]
                    e = di // H
                    tt = di % H
                    ss = torch.from_numpy(states[e, tt].astype(np.float32))
                    ns = torch.from_numpy(states[e, tt + 1].astype(np.float32))
                    aa = torch.from_numpy(inputs[e, tt].astype(np.int64))
                    rr = torch.from_numpy(((states[e, tt + 1, 1] == 0).astype(np.float64) - 0.06 * inputs[e, tt]).astype(np.float32))
                    done = torch.from_numpy((tt == H - 1).astype(np.float32))
                    ww = torch.from_numpy(weights)
                    qsa = q(ss).gather(1, aa[:, None]).squeeze(1)
                    target = double_q_target(q, qt, ns, rr, done)
                    td = target - qsa
                    loss = (ww * td.pow(2)).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tdn = td.detach().numpy()
                    if not np.isfinite(tdn).all() or not np.isfinite(float(loss.detach())):
                        raise ArithmeticError('Nonfinite DDQN ' + name)
                    update_priorities(prior, ix, tdn)
                    updates += 1
                    qreplay.append({'step': stepid, 'idx': ix, 'dataidx': di, 'probability': prob[ix], 'weights': weights, 'TD': tdn.copy(), 'target': target.numpy().copy(), 'loss': float(loss.detach())})
                    replay_uniform.append(ru)
                    if (stepid + 1) % 100 == 0:
                        qt.load_state_dict(q.state_dict())
            decisions.append([ep, t, k, a, u, tau, ell, R, float(delta), eb, float(eta), lb, float(lam), draw, float(normalize(probs)[a]) if method != 'DDQN_PER' else 1.0])
            t += ell
            k += 1
            scheduled += tau
        counts[ep - 1] = [H, k, scheduled, int(inputs[ep - 1].sum()), int((states[ep - 1, 1:, 1] == 0).sum()), hit, coded]
        train_ns += time.perf_counter_ns() - clock
        if ep in CONFIG['training']['checkpoint_episodes']:
            clock = time.perf_counter_ns()
            arrays = {'v': v.copy(), 'actor': actor.copy(), 'eta': np.array(eta), 'lambda': np.array(lam), 'episode': np.array(ep), 'plant_steps': np.array(ep * H), 'decisions': np.array(len(decisions))}
            if q is not None:
                arrays.update({'online.' + k: v.numpy().copy() for k, v in q.state_dict().items()})
                arrays.update({'target.' + k: v.numpy().copy() for k, v in qt.state_dict().items()})
                for i, param in enumerate(q.parameters()):
                    for k, val in opt.state[param].items():
                        arrays[f'optimizer.{i}.{k}'] = val.numpy().copy()
                arrays.update({'replay_dataidx': np.asarray(dataidx, dtype=np.int64), 'replay_priority': np.asarray(prior), 'replay_pos': np.array(pos)})
            ckpath = folder / 'checkpoints' / f'e{ep:03d}.npz'
            np.savez_compressed(ckpath, **arrays)
            ck = {'checkpoint_id': name + f'__e{ep:03d}', 'episode': ep, 'plant_steps': ep * H, 'decisions': len(decisions), 'actor_sha256': sha(ckpath), 'checkpoint_file': f'checkpoints/e{ep:03d}.npz', 'feature_signature': model.signature, 'method': method, 'seed_label': seed, 'Gamma': budget, 'lambda': float(lam) if method == 'PD_SMAC' else None, 'selection': 'prespecified complete-episode checkpoint; terminal selected'}
            checkpoints.append(ck)
            serialize_ns += time.perf_counter_ns() - clock
            clock = time.perf_counter_ns()
            if q is not None:
                q.eval()
            for mode in ['greedy'] if method == 'DDQN_PER' else ['greedy', 'stochastic']:
                evals.append(evaluate(folder / 'evaluation', ep, seed, method, budget if method == 'PD_SMAC' else 0.4, actor, model, tapes, q, mode))
            if q is not None:
                q.train()
            if ep == E:
                endpoint_ns += time.perf_counter_ns() - clock
            else:
                eval_ns += time.perf_counter_ns() - clock
    clock = time.perf_counter_ns()
    np.savez_compressed(folder / 'training_raw.npz', states=states, inputs=inputs, counts=counts, decisions=np.array(decisions), policy_values=np.array(prerows), exploration=np.array(exploration))
    if q is not None:
        np.savez_compressed(folder / 'replay_raw.npz', **{k: np.array([r[k] for r in qreplay]) for k in qreplay[0]}, uniform=np.array(replay_uniform))
    dump(folder / 'checkpoints.json', checkpoints)
    dump(folder / 'evaluations.json', evals)
    dump(folder / 'streams.json', {'keys': keys, 'initial_states': initial, 'final_states': {p: g.bit_generator.state for p, g in generators.items()}, 'torch_initialization_key': torchkey if q is not None else None, 'torch_integer_seed': integer_seed(torchkey) if q is not None else None})
    serialize_ns += time.perf_counter_ns() - clock
    total = time.perf_counter_ns() - t0
    receipt = {'run_id': name, 'method': method, 'seed_label': seed, 'Gamma': budget if method == 'PD_SMAC' else None, 'counts': {'episodes': E, 'plant_steps': E * H, 'decisions': len(decisions), 'optimizer_updates': updates}, 'clip_audit': clip, 'training_tape_hash': sha(folder / 'training_tapes.npz'), 'feature_signature': model.signature, 'decision_columns': DECISION_COLUMNS, 'count_columns': ['T', 'K', 'scheduled_total', 'input_on_steps', 'target_steps', 'hit', 'coded_hit_time'], 'timing': {'initialization_ns': init_ns, 'training_ns_excluding_eval_io': train_ns, 'checkpoint_evaluation_ns': eval_ns, 'endpoint_evaluation_ns': endpoint_ns, 'aggregation_ns': 0, 'serialization_ns': serialize_ns, 'end_to_end_ns': total, 'unallocated_driver_overhead_ns': total - init_ns - train_ns - eval_ns - endpoint_ns - serialize_ns, 'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'rss_measurement_method': 'Linux ru_maxrss cumulative sequential process; not comparative isolated RSS'}, 'status': 'completed', 'fresh_installation': False, 'performance_selection': False}
    dump(folder / 'execution.json', receipt)
    print(name, receipt['counts'], 'done', flush=True)
    return receipt
