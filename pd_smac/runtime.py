"""Local file identities and execution receipts. No network operations."""
from __future__ import annotations
import hashlib, json, os, platform, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def output_dir(path):
    p = Path(path).resolve()
    if p.is_relative_to(ROOT):
        raise ValueError('Outputs must be outside the immutable companion directory.')
    p.mkdir(parents=True, exist_ok=False)
    return p

def environment():
    import importlib.metadata as meta
    packages = {}
    for name in ['numpy','scipy','numba','llvmlite','pandas','torch','psutil','threadpoolctl','jsonschema']:
        try: packages[name] = meta.version(name)
        except meta.PackageNotFoundError: packages[name] = None
    return {'python':sys.version, 'executable':sys.executable, 'platform':platform.platform(),
            'packages':packages, 'prefix':sys.prefix, 'base_prefix':sys.base_prefix,
            'virtual_environment':sys.prefix != sys.base_prefix,
            'thread_environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
            'fresh_installation_claim':False, 'public_release':False}

def configuration(study):
    if study not in ['four_state','lactose_meanfield','lactose_first_arrival','melanoma28','melanoma70']:
        raise ValueError('Unknown study')
    return load(ROOT/'pd_smac'/study/'config.json')

def protocol_hash(study):
    return hashlib.sha256(json.dumps({'study':configuration(study),'common':load(ROOT/'pd_smac/common/config.json')},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def verify_inputs(study, folder, acknowledged=False):
    if not acknowledged:
        raise PermissionError('Explicit acknowledgement of permission for local input use is required; this grants no rights.')
    if folder is None: raise ValueError('An external biological input directory is required.')
    if study not in ('melanoma28','melanoma70'): raise ValueError('Unknown biological study.')
    folder = Path(folder).resolve(); checks=[]
    if folder.is_relative_to(ROOT):
        raise ValueError('Biological inputs must remain outside the companion.')
    for item in load(ROOT/'manifests/external_inputs.json')['studies'][study]:
        path = folder/item['path']
        if not path.is_file(): raise FileNotFoundError(f'Required locally supplied input is missing: {path}')
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Biological input identity mismatch: '+str(path))
        checks.append({'relative_path':item['path'],'sha256':item['sha256'],'bytes':item['bytes']})
    return checks

def import_inputs(study, source, destination, acknowledged=False):
    if not acknowledged: raise PermissionError('Local-use permission must be acknowledged; redistribution is not authorized.')
    source=Path(source).resolve();dest=Path(destination).resolve()
    if dest.is_relative_to(ROOT):raise ValueError('Do not stage biological inputs inside the companion.')
    plan=[]
    for item in load(ROOT/'manifests/external_inputs.json')['studies'][study]:
        p=source/Path(item['path']).name
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('Source input is missing or not the required exact file: '+str(p))
        target=dest/item['path']
        if target.exists():raise FileExistsError('Input staging will not overwrite '+str(target))
        plan.append((p,target))
    for p,target in plan:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    return verify_inputs(study,dest,True)

def verify_package():
    """Authenticate exported payload, excluding only Python caches and Git metadata."""
    from pathlib import PurePosixPath
    m=load(ROOT/'MANIFEST_SHA256.json')
    actual=set()
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if p.is_symlink(): raise ValueError('Companion symlinks are not permitted.')
        if '.git' in rel.parts or '__pycache__' in rel.parts: continue
        if p.is_file() and rel.as_posix()!='MANIFEST_SHA256.json': actual.add(rel.as_posix())
    for name in m['files']:
        q=PurePosixPath(name)
        if not name or q.is_absolute() or '..' in q.parts or str(q)!=name or '\\' in name or ':' in name:
            raise ValueError('Unsafe manifest path.')
    if actual!=set(m['files']):
        raise ValueError({'missing':sorted(set(m['files'])-actual),'unexpected':sorted(actual-set(m['files']))})
    for rel,x in m['files'].items():
        p=ROOT/rel
        if p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:
            raise ValueError('Companion hash mismatch: '+rel)
    return {'status':'passed','file_count':len(m['files'])+1,'biological_inputs_bundled':False,'public_release':False,'scope':'transport integrity, not numerical reproduction or installation'}
