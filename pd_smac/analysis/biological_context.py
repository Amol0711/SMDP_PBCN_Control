"""External, hash-bound schemas. No biological file or credential is bundled."""
from pathlib import Path
from ..runtime import ROOT,load,sha
PROTOCOL=load(ROOT/'pd_smac/analysis/config/biological_protocol_identity.json')['protocol_sha256']
_STATE_ROOT=None

def bind_state_root(path):
 global _STATE_ROOT
 p=Path(path).resolve()
 if p.is_relative_to(ROOT):raise ValueError('State inputs must remain outside the companion.')
 expected=load(ROOT/'manifests/continuation_external.json')['files']
 for name in ['schemas/prefix.json','schemas/extension.json']:
  f=p/name;e=expected[name]
  if not f.is_file()or f.stat().st_size!=e['bytes']or sha(f)!=e['sha256']:raise ValueError('External schema identity mismatch '+name)
 _STATE_ROOT=p

def schema_path(kind):
 if _STATE_ROOT is None:raise PermissionError('Bind verified external schemas before loading complete biological state.')
 if kind not in ['prefix','extension']:raise ValueError('Unknown state schema')
 return _STATE_ROOT/'schemas'/(kind+'.json')
def schema_version(kind):return load(schema_path(kind))['properties']['schema_version']['const']
def scope_value(kind):return load(schema_path(kind))['properties']['scope']['const']
