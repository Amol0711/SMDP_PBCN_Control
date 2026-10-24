"""Portable configuration identity and validation-only execution contracts."""
from __future__ import annotations
import hashlib,json,os,platform,sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from ...runtime import ROOT
CONFIG=ROOT/'pd_smac/analysis/config'
ENV={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
def read(p):return json.loads(Path(p).read_text())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def info(p):return {'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def integer(x,name,lo=0,hi=None):
 if isinstance(x,(bool,np.bool_))or not isinstance(x,(int,np.integer))or x<lo or (hi is not None and x>hi):raise ValueError('invalid '+name)
 return int(x)
def scalar(x,name):
 if isinstance(x,(bool,np.bool_))or not np.isscalar(x):raise ValueError(name)
 y=float(x)
 if not np.isfinite(y):raise ValueError('nonfinite '+name)
 return y
@dataclass(frozen=True)
class Protocol:
 config:dict
 common:dict
 sha256:str
 root:Path

def load_protocol(root=ROOT):
 ident=read(CONFIG/'identities.json')
 for n,v in ident['portable_files'].items():
  if info(CONFIG/n)!=v:raise ValueError('portable protocol identity changed '+n)
 return Protocol(read(CONFIG/'four_state_continuing.json'),read(CONFIG/'common.json'),ident['original_continuing_protocol_sha256'],ROOT)

def verify_protocol_object(protocol):
 p=load_protocol()
 if protocol.config!=p.config or protocol.common!=p.common or protocol.sha256!=p.sha256:raise ValueError('in-memory protocol mutation')

def source_identity(root=ROOT):
 return {str(p.relative_to(ROOT)):info(p)for p in sorted((ROOT/'pd_smac/analysis/continuing').iterdir())if p.is_file()and p.suffix in ('.py','.json')}

def execution_keys(seed,mode):
 seed=integer(seed,'seed')
 if mode=='functional_test':
  if not 900<=seed<=909:raise ValueError('isolated debug seeds required')
  return {n:[20260919,212,10,3,seed,300,0,0,p]for n,p in [('actor',21),('plant',22),('initial_state',23)]}
 if mode=='production':
  if seed not in range(10):raise ValueError('production cohort is 0..9')
  return {n:[20260919,212,p,3,seed,300,0,0,0]for n,p in [('actor',21),('plant',22),('initial_state',23)]}
 raise ValueError('unknown execution mode')

def debug_limit(seed,stop):execution_keys(seed,'functional_test');integer(stop,'debug stop',1,2048)
def verify_production_authorization(*args,**kwargs):
 raise PermissionError('New scientific production is not authorized by this computational package.')
def environment():
 from ...runtime import environment as describe
 return describe()
