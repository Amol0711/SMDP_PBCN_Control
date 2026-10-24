from __future__ import annotations
import copy, hashlib, json, os, sys, platform
from pathlib import Path
import numpy as np
import torch
from ...melanoma28 import run_study as original
from ...melanoma28.model import Model
from ..biological_context import schema_path, schema_version, PROTOCOL

def read(p):
    return json.loads(Path(p).read_text())

def dump(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False, default=lambda z: z.item() if isinstance(z, np.generic) else z.tolist()) + '\n')

def info(p):
    p = Path(p)
    with p.open('rb') as f:
        h = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'sha256': h, 'bytes': p.stat().st_size}

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def ah(a):
    a = np.asarray(a)
    return {'dtype': str(a.dtype), 'shape': list(a.shape), 'sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}

def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def eq(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and (a.tobytes() == b.tobytes())

def compare(a, b):
    keys = sorted(set(a) | set(b))
    errs = []
    n = 0
    for k in keys:
        if k not in a or k not in b:
            errs.append({'array': k, 'reason': 'missing'})
            continue
        n += 1
        if not eq(a[k], b[k]):
            row = {'array': k, 'shape_a': list(a[k].shape), 'shape_b': list(b[k].shape), 'dtype_a': str(a[k].dtype), 'dtype_b': str(b[k].dtype)}
            if a[k].shape == b[k].shape:
                row['max_abs'] = float(np.max(abs(a[k].astype(float) - b[k].astype(float)), initial=0))
            errs.append(row)
    return {'equal': not errs, 'compared': n, 'differences': errs}

def train_arrays(L):
    d = {'v': L['v'].copy(), 'actor': L['actor'].copy(), 'eta': np.array(L['eta']), 'lambda': np.array(L['lam']), 'episode': np.array(L['ep']), 'plant_steps': np.array(L['ep'] * 28), 'decisions': np.array(len(L['decisions']))}
    q = L['q']
    if q is not None:
        for label, net in [('online', q), ('target', L['qt'])]:
            d.update({label + '.' + k: v.detach().numpy().copy() for k, v in net.state_dict().items()})
        for i, param in enumerate(q.parameters()):
            for k, v in L['opt'].state[param].items():
                d[f'optimizer.{i}.{k}'] = v.detach().numpy().copy()
        d.update(replay_dataidx=np.asarray(L['dataidx'], dtype=np.int64), replay_priority=np.asarray(L['prior'], dtype=np.float64), replay_pos=np.array(L['pos']))
    return d

def state_payload(L):
    ep = int(L['ep'])
    a = train_arrays(L)
    for key in ['states', 'inputs', 'counts']:
        a['history.' + key] = L[key][:ep].copy()
    a.update({'history.decisions': np.array(L['decisions']), 'history.policy_values': np.array(L['prerows']), 'history.exploration': np.array(L['exploration']), 'tapes.plant': L['plant'].copy(), 'tapes.reset': L['reset'].copy(), 'torch_rng': torch.get_rng_state().numpy().copy()})
    q = L['q']
    groups = None
    if q is not None:
        groups = L['opt'].state_dict()['param_groups']
    meta = {'schema_version': schema_version('extension'), 'method': L['method'], 'seed_label': int(L['seed']), 'budget': float(L['budget']), 'episode': ep, 'next_episode': ep + 1, 'plant_steps': ep * 28, 'decisions': len(L['decisions']), 'optimizer_updates': int(L['updates']), 'replay_ring_position': int(L['pos']), 'training_keys': L['keys'], 'rng_states': {k: g.bit_generator.state for k, g in L['generators'].items()}, 'scalar_kinds': {'eta': 'python_float' if type(L['eta']) is float else str(np.asarray(L['eta']).dtype), 'lambda': 'python_float' if type(L['lam']) is float else str(np.asarray(L['lam']).dtype)}, 'torch_training': None if q is None else {'online': bool(q.training), 'target': bool(L['qt'].training)}, 'optimizer_param_groups': groups, 'feature_signature': L['model'].signature, 'protocol_sha256': PROTOCOL, 'scope': 'original_continuation_from_verified_episode240', 'boundary': 'completed_episode_before_next_reset', 'reserved_episodes': len(L['reset']), 'offsets': {'train_init': 0, 'train_plant': int(L['plant'].size), 'train_reset': len(L['reset']), 'train_actor': len(L['decisions']) if q is None else 0, 'train_exploration': 0 if q is None else ep * 28 * 2, 'train_replay': int(L['updates']) * 32}, 'continued_from_episode': 240, 'lineage': L['lineage']}
    return (a, meta)

def payload_digest(a, m):
    return digest({'arrays': {k: ah(v) for k, v in sorted(a.items())}, 'meta': m})

def save_state(folder, L):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    tmp = folder.with_name(folder.name + '.pending')
    if tmp.exists():
        raise FileExistsError(tmp)
    a, m = state_payload(L)
    folder.parent.mkdir(parents=True, exist_ok=True)
    tmp.mkdir()
    with (tmp / 'state_arrays.npz').open('wb') as f:
        np.savez_compressed(f, **a)
        f.flush()
        os.fsync(f.fileno())
    m['arrays_identity'] = {k: ah(v) for k, v in sorted(a.items())}
    dump(tmp / 'state.json', m)
    dump(tmp / 'manifest.json', {'files': {f: info(tmp / f) for f in ['state_arrays.npz', 'state.json']}, 'payload_sha256': payload_digest(a, {k: v for k, v in m.items() if k != 'arrays_identity'})})
    for f in ['state.json', 'manifest.json']:
        with (tmp / f).open('rb') as fp:
            os.fsync(fp.fileno())
    tmp.rename(folder)
    return m

def load_state(folder, model):
    folder = Path(folder)
    manifest = read(folder / 'manifest.json')
    if set((p.name for p in folder.iterdir())) != {'state_arrays.npz', 'state.json', 'manifest.json'}:
        raise ValueError('unexpected snapshot file')
    for f, z in manifest['files'].items():
        if info(folder / f) != z:
            raise ValueError('snapshot hash')
    a = arrays(folder / 'state_arrays.npz')
    m = read(folder / 'state.json')
    import jsonschema
    jsonschema.Draft202012Validator(read(schema_path('extension'))).validate(m)
    expected_actor = (10, 7) if m['method'] == 'PD_SMAC' else (7,)
    if a['actor'].shape != expected_actor or a['actor'].dtype != np.float32 or a['v'].shape != (7,) or (a['v'].dtype != np.float32):
        raise ValueError('actor/critic schema')
    if any((not np.isfinite(v).all() for v in a.values())):
        raise ValueError('nonfinite state')
    if m['training_keys'] != {k: original.key(k, m['method'], m['seed_label'], m['budget']) for k in m['training_keys']}:
        raise ValueError('wrong stream identity')
    if m['budget'] != (0.4 if m['method'] == 'PD_SMAC' else 0.0):
        raise ValueError('wrong budget')
    ep = m['episode']
    if a['history.states'].shape != (ep, 29, 28) or a['history.inputs'].shape != (ep, 28) or a['history.counts'].shape != (ep, 7):
        raise ValueError('history shape')
    if m['decisions'] != len(a['history.decisions']) or m['optimizer_updates'] != (ep * 28 - 31 if m['method'] == 'DDQN_PER' else 0):
        raise ValueError('update counters')
    if {k: ah(v) for k, v in sorted(a.items())} != m['arrays_identity']:
        raise ValueError('logical array identity')
    if payload_digest(a, {k: v for k, v in m.items() if k != 'arrays_identity'}) != manifest['payload_sha256']:
        raise ValueError('payload identity')
    if m['protocol_sha256'] != PROTOCOL or model.signature != m['feature_signature']:
        raise ValueError('protocol/model identity')
    if m['episode'] != int(a['episode']) or m['next_episode'] != m['episode'] + 1 or m['plant_steps'] != 28 * m['episode']:
        raise ValueError('counter identity')
    gens = {}
    for k, key in m['training_keys'].items():
        gen = original.rng(key)
        gen.bit_generator.advance(m['offsets'][k])
        if gen.bit_generator.state != m['rng_states'][k]:
            raise ValueError('RNG state/offset disagreement ' + k)
        gens[k] = gen
    L = {'method': m['method'], 'seed': m['seed_label'], 'budget': m['budget'], 'ep': m['episode'], 'actor': a['actor'].copy(), 'v': a['v'].copy(), 'eta': float(a['eta']) if m['scalar_kinds']['eta'] == 'python_float' else a['eta'][()], 'lam': float(a['lambda']) if m['scalar_kinds']['lambda'] == 'python_float' else a['lambda'][()], 'updates': m['optimizer_updates'], 'pos': m['replay_ring_position'], 'keys': m['training_keys'], 'generators': gens, 'model': model, 'plant': a['tapes.plant'].copy(), 'reset': a['tapes.reset'].copy(), 'states': a['history.states'].copy(), 'inputs': a['history.inputs'].copy(), 'counts': a['history.counts'].copy(), 'decisions': a['history.decisions'].tolist(), 'prerows': [r.copy() for r in a['history.policy_values']], 'exploration': [r.copy() for r in a['history.exploration']], 'dataidx': [], 'prior': [], 'q': None, 'qt': None, 'opt': None, 'new_replay_rows': [], 'lineage': m['lineage']}
    if m['method'] == 'DDQN_PER':
        q = original.QNet()
        qt = original.QNet()
        for label, net in [('online', q), ('target', qt)]:
            net.load_state_dict({k[len(label) + 1:]: torch.from_numpy(v.copy()) for k, v in a.items() if k.startswith(label + '.')})
        opt = torch.optim.Adam(q.parameters(), lr=0.001)
        s = {}
        for i, param in enumerate(q.parameters()):
            s[i] = {k.split('.', 2)[2]: torch.from_numpy(v.copy()) for k, v in a.items() if k.startswith(f'optimizer.{i}.')}
        opt.load_state_dict({'state': s, 'param_groups': m['optimizer_param_groups']})
        q.train(m['torch_training']['online'])
        qt.train(m['torch_training']['target'])
        L.update(q=q, qt=qt, opt=opt, dataidx=a['replay_dataidx'].tolist(), prior=a['replay_priority'].tolist())
    torch.set_rng_state(torch.from_numpy(a['torch_rng'].copy()))
    restored, mm = state_payload(L)
    if not compare(a, restored)['equal'] or {k: v for k, v in m.items() if k != 'arrays_identity'} != mm:
        raise ValueError('roundtrip state mismatch')
    return L

def schedules(ep):
    if not isinstance(ep, int) or ep < 1:
        raise ValueError('absolute episode required')
    return (max(0.05, 1 - ep / (0.7 * 240)), min(1.0, 0.4 + 0.6 * ep / 240))

def advance(L, stop_episode):
    if L.get('_authorization_checked') is not True:
        raise PermissionError('fixed_configuration authorization not checked')
    if not isinstance(stop_episode, int) or stop_episode < L['ep'] or stop_episode > 1200 or (L['ep'] < 240):
        raise ValueError('outside locked continuation horizon')
    if len(L['reset']) != 1200 or L['plant'].shape != (1200, 28, 28):
        raise ValueError('full tail reservation required')
    m = L['model']
    method = L['method']
    q = L['q']
    qt = L['qt']
    opt = L['opt']
    g = L['generators']
    actor = L['actor']
    v = L['v']
    eta = L['eta']
    lam = L['lam']
    budget = L['budget']
    end = stop_episode
    for ep in range(L['ep'] + 1, end + 1):
        for k, shape, dtype in [('states', (1, 29, 28), np.int8), ('inputs', (1, 28), np.int8), ('counts', (1, 7), np.int64)]:
            L[k] = np.concatenate((L[k], np.empty(shape, dtype=dtype)))
        states = L['states']
        inputs = L['inputs']
        s, _ = m.reset(L['reset'][ep - 1])
        states[ep - 1, 0] = s
        t = 0
        k = 0
        scheduled = 0
        hit = 0
        coded = 29
        while t < 28:
            phi = m.features(s)
            eb = float(eta)
            lb = float(lam)
            draw = float(g['train_actor'].random()) if method != 'DDQN_PER' else 0.0
            if method == 'PD_SMAC':
                probs = original.softmax(actor @ phi)
                a = original.categorical(probs, draw)
                u, tau = map(int, original.PAIRS[a])
                L['prerows'].append(np.r_[probs, np.zeros(0)])
            elif method == 'TT_AC':
                z = actor @ phi
                p = original.sig(z)
                probs = np.array([1 - p, p])
                a = original.categorical(probs, draw)
                u = a
                tau = 1
                L['prerows'].append(np.r_[probs, np.zeros(8)])
            else:
                eps, _ = schedules(ep)
                ex = g['train_exploration'].random(2)
                L['exploration'].append(ex)
                with torch.no_grad():
                    qs = q(torch.from_numpy(s.astype(np.float32))[None, :]).numpy()[0]
                a = int(ex[1] * 2) if ex[0] < eps else int(np.argmax(qs))
                u = a
                tau = 1
                probs = np.eye(2)[u]
                L['prerows'].append(np.r_[qs, np.zeros(8)])
            ell = min(tau, 28 - t)
            R = 0.0
            for l in range(ell):
                s = m.step(s, u, L['plant'][ep - 1, t + l])
                states[ep - 1, t + l + 1] = s
                inputs[ep - 1, t + l] = u
                R += (1.0 if s[1] == 0 else 0.0) - 0.06 * u
                if s[1] == 0 and (not hit):
                    hit = 1
                    coded = t + l + 1
            if method != 'DDQN_PER':
                delta = R - lam - eta * ell + v @ m.features(s) - v @ phi
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
                    raise ArithmeticError('nonfinite AC state')
                if max(float(np.max(np.abs(actor))), float(np.max(np.abs(v))), abs(float(eta)), abs(float(lam))) > 1000000000000.0:
                    raise ArithmeticError('magnitude safety stop; no critic projection')
            else:
                delta = 0.0
                stepid = (ep - 1) * 28 + t
                prior = L['prior']
                dataidx = L['dataidx']
                newp = max(prior) if prior else 1.0
                if len(dataidx) < 5000:
                    dataidx.append(stepid)
                    prior.append(newp)
                else:
                    dataidx[L['pos']] = stepid
                    prior[L['pos']] = newp
                    L['pos'] = (L['pos'] + 1) % 5000
                if len(dataidx) >= 32:
                    _, beta = schedules(ep)
                    prob = np.asarray(prior, dtype=np.float64) ** 0.6
                    prob /= prob.sum()
                    ru = g['train_replay'].random(32)
                    ix = np.searchsorted(np.cumsum(prob), ru, side='right')
                    di = np.asarray(dataidx)[ix]
                    weights = (len(dataidx) * prob[ix]) ** (-beta)
                    weights = (weights / weights.max()).astype(np.float32)
                    e = di // 28
                    tt = di % 28
                    ss = torch.from_numpy(states[e, tt].astype(np.float32))
                    ns = torch.from_numpy(states[e, tt + 1].astype(np.float32))
                    aa = torch.from_numpy(inputs[e, tt].astype(np.int64))
                    rr = torch.from_numpy(((states[e, tt + 1, 1] == 0).astype(np.float64) - 0.06 * inputs[e, tt]).astype(np.float32))
                    done = torch.from_numpy((tt == 27).astype(np.float32))
                    ww = torch.from_numpy(weights)
                    qsa = q(ss).gather(1, aa[:, None]).squeeze(1)
                    with torch.no_grad():
                        na = torch.argmax(q(ns), dim=1)
                        target = rr + 0.98 * (1 - done) * qt(ns).gather(1, na[:, None]).squeeze(1)
                    td = target - qsa
                    loss = (ww * td.pow(2)).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tdn = td.detach().numpy()
                    for pp in q.parameters():
                        pv = pp.detach().numpy()
                        if not np.isfinite(pv).all() or np.max(np.abs(pv), initial=0) > 1000000000000.0:
                            raise ArithmeticError('DDQN parameter safety stop')
                    if not np.isfinite(tdn).all() or not np.isfinite(float(loss.detach())):
                        raise ArithmeticError('nonfinite DDQN')
                    for index in np.unique(ix):
                        prior[int(index)] = float(np.max(np.abs(tdn[ix == index])) + 0.001)
                    L['updates'] += 1
                    L['new_replay_rows'].append({'step': stepid, 'idx': ix.copy(), 'dataidx': di.copy(), 'probability': prob[ix].copy(), 'weights': weights.copy(), 'TD': tdn.copy(), 'target': target.numpy().copy(), 'loss': float(loss.detach()), 'uniform': ru.copy()})
                    if (stepid + 1) % 100 == 0:
                        qt.load_state_dict(q.state_dict())
            L['decisions'].append([ep, t, k, a, u, tau, ell, R, float(delta), eb, float(eta), lb, float(lam), draw, float(original.normalize(probs)[a]) if method != 'DDQN_PER' else 1.0])
            t += ell
            k += 1
            scheduled += tau
        L['counts'][ep - 1] = [28, k, scheduled, int(inputs[ep - 1].sum()), int((states[ep - 1, 1:, 1] == 0).sum()), hit, coded]
        L.update(ep=ep, eta=eta, lam=lam)
        guard_state(L)
        if L.get('_pause_requested', False):
            return L
    return L

def guard_state(L):
    import resource
    vals = [L['actor'], L['v'], np.asarray(L['eta']), np.asarray(L['lam'])]
    if L['q'] is not None:
        vals.extend((p.detach().numpy() for net in [L['q'], L['qt']] for p in net.parameters()))
    if any((not np.isfinite(x).all() for x in vals)):
        raise ArithmeticError('nonfinite operational state')
    if any((np.max(np.abs(x), initial=0) > 1000000000000.0 for x in vals)):
        raise ArithmeticError('operational magnitude limit; no clipping')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4294967296:
        L['_pause_requested'] = True

def extend_reserved_tapes(L):
    if L['ep'] != 240 or len(L['reset']) != 240 or L['plant'].shape != (240, 28, 28):
        raise ValueError('tail may be reserved once at verified e240 only')
    p = L['generators']['train_plant'].random((960, 28, 28))
    z = L['generators']['train_reset'].random(960)
    L['plant'] = np.concatenate((L['plant'], p), axis=0)
    L['reset'] = np.concatenate((L['reset'], z), axis=0)
    return L
