from __future__ import annotations
import copy, hashlib, json, os, sys, platform
from pathlib import Path
import numpy as np
import torch
from ...melanoma28 import run_study as original
from ...melanoma28.model import Model
from ..biological_context import schema_path, schema_version, scope_value, PROTOCOL

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
    meta = {'schema_version': schema_version('prefix'), 'method': L['method'], 'seed_label': int(L['seed']), 'budget': float(L['budget']), 'episode': ep, 'next_episode': ep + 1, 'plant_steps': ep * 28, 'decisions': len(L['decisions']), 'optimizer_updates': int(L['updates']), 'replay_ring_position': int(L['pos']), 'training_keys': L['keys'], 'rng_states': {k: g.bit_generator.state for k, g in L['generators'].items()}, 'scalar_kinds': {'eta': 'python_float' if type(L['eta']) is float else str(np.asarray(L['eta']).dtype), 'lambda': 'python_float' if type(L['lam']) is float else str(np.asarray(L['lam']).dtype)}, 'torch_training': None if q is None else {'online': bool(q.training), 'target': bool(L['qt'].training)}, 'optimizer_param_groups': groups, 'feature_signature': L['model'].signature, 'protocol_sha256': PROTOCOL, 'scope': scope_value('prefix'), 'boundary': 'completed_episode_before_next_reset', 'reserved_episodes': len(L['reset']), 'offsets': {'train_init': 0, 'train_plant': int(L['plant'].size), 'train_reset': len(L['reset']), 'train_actor': len(L['decisions']) if q is None else 0, 'train_exploration': 0 if q is None else ep * 28 * 2, 'train_replay': int(L['updates']) * 32}, 'no_extended_training': ep <= 240}
    return (a, meta)

def payload_digest(a, m):
    return digest({'arrays': {k: ah(v) for k, v in sorted(a.items())}, 'meta': m})

def save_state(folder, L):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    a, m = state_payload(L)
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'state_arrays.npz', **a)
    m['arrays_identity'] = {k: ah(v) for k, v in sorted(a.items())}
    dump(folder / 'state.json', m)
    dump(folder / 'manifest.json', {'files': {f: info(folder / f) for f in ['state_arrays.npz', 'state.json']}, 'payload_sha256': payload_digest(a, {k: v for k, v in m.items() if k != 'arrays_identity'})})
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
    jsonschema.Draft202012Validator(read(schema_path('prefix'))).validate(m)
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
    L = {'method': m['method'], 'seed': m['seed_label'], 'budget': m['budget'], 'ep': m['episode'], 'actor': a['actor'].copy(), 'v': a['v'].copy(), 'eta': float(a['eta']) if m['scalar_kinds']['eta'] == 'python_float' else a['eta'][()], 'lam': float(a['lambda']) if m['scalar_kinds']['lambda'] == 'python_float' else a['lambda'][()], 'updates': m['optimizer_updates'], 'pos': m['replay_ring_position'], 'keys': m['training_keys'], 'generators': gens, 'model': model, 'plant': a['tapes.plant'].copy(), 'reset': a['tapes.reset'].copy(), 'states': a['history.states'].copy(), 'inputs': a['history.inputs'].copy(), 'counts': a['history.counts'].copy(), 'decisions': a['history.decisions'].tolist(), 'prerows': [r.copy() for r in a['history.policy_values']], 'exploration': [r.copy() for r in a['history.exploration']], 'dataidx': [], 'prior': [], 'q': None, 'qt': None, 'opt': None, 'new_replay_rows': []}
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
