from __future__ import annotations
from pathlib import Path
import hashlib, io, json, zipfile, sys
import numpy as np

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def npz(p, **arrays):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(arrays):
            b = io.BytesIO()
            np.save(b, np.asarray(arrays[name]), allow_pickle=False)
            zi = zipfile.ZipInfo(name + '.npy', (1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, b.getvalue())

def key(purpose, method, seed, budget=0, checkpoint=0, rollout=0, component=0):
    if 900 <= int(seed) <= 909:
        component = int(purpose)
        purpose = 10
    return [20260917, 14, int(purpose), int(method), int(seed), int(budget), int(checkpoint), int(rollout), int(component)]

def rng(k):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(k)))
