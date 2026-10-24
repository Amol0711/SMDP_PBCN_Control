"""Fail-closed host and input identity checks, without model or rollout execution."""
from pathlib import Path
import hashlib,json,os,platform,sys
from ..runtime import ROOT,load,sha

def actual_environment():
 import numpy as np,torch
 torch.set_num_threads(1)
 if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
 torch.use_deterministic_algorithms(True)
 return {'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform(),
 'cpu_model':next((s.split(':',1)[1].strip()for s in Path('/proc/cpuinfo').read_text().splitlines()if s.startswith('model name')),''),
 'logical_cores':os.cpu_count(),'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
 'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
 'thread_environment':{k:os.environ.get(k)for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']},
 'clean_installation':False}

def environment_check():
 observed=actual_environment();expected=load(ROOT/'pd_smac/analysis/config/recorded_environment.json')
 differences={k:{'actual':observed.get(k),'expected':expected.get(k)}for k in sorted(set(observed)|set(expected))if observed.get(k)!=expected.get(k)}
 return {'status':'eligible_environment_only'if not differences else 'blocked_environment','actual_environment':observed,'recorded_environment':expected,'differences':differences,
 'historical_environment_equality_required':True,'new_production_authorized':False,'isolated_installation_claim':False}

def require_environment():
 report=environment_check()
 if report['differences']:raise PermissionError('environment changed; revalidate same-host/prefix gates')
 return report

def verify_external(folder,acknowledged=False,scope='all'):
 if not acknowledged:raise PermissionError('Local-use responsibility must be acknowledged; this grants no redistribution rights.')
 p=Path(folder).resolve()
 if p.is_relative_to(ROOT):raise ValueError('External inputs must not be placed in the companion.')
 expected=load(ROOT/'manifests/continuation_external.json')['files'];checks=[]
 for name,e in expected.items():
  if scope=='authorization'and not name.startswith('authorization/'):continue
  f=p/name
  if not f.is_file()or f.is_symlink()or f.stat().st_size!=e['bytes']or sha(f)!=e['sha256']:raise ValueError('External input identity mismatch '+name)
  checks.append(name)
 return {'status':'passed','verified_files':len(checks),'scope':scope,'redistribution_rights_granted':False}

def require_continuation(folder,run_id,acknowledged=False):
 env=require_environment() # No biological state/model is loaded before this check.
 manifest=load(ROOT/'manifests/continuation_external.json')
 if run_id not in manifest['preselected_replay_run_ids']:raise PermissionError('Only preselected seed-zero validation replays are supported.')
 verification=verify_external(folder,acknowledged)
 p=Path(folder);desc=load(p/'authorization/descriptor.json');receipt=load(p/'authorization/receipt.json');matrix=load(p/'authorization/eligibility.json')
 # Byte identity above binds the original receipt, including its stage, protocol and evidence.
 if not desc['branch']['historical_continuation_eligible']:raise PermissionError('historical continuation blocked')
 if desc['tested_environment']!=env['actual_environment']:raise PermissionError('environment changed; revalidate same-host/prefix gates')
 if receipt['protocol_sha256']!=desc['protocol_sha256']or receipt['eligibility_sha256']!=desc['eligibility_sha256']:raise PermissionError('wrong protocol/evidence')
 if receipt.get('explicit_user_execution_authorization')is not True or receipt.get('favorable_result_condition')is not None:raise PermissionError('invalid historical authorization')
 if sorted(receipt['run_ids'])!=sorted(r['run_id']for r in matrix['rows'])or len(matrix['rows'])!=12:raise PermissionError('all twelve historical origins required')
 return {'environment':env,'inputs':verification,'descriptor':desc,'receipt':receipt,'matrix':matrix,'new_production_authorized':False}


def inspect_restart_files(folder,acknowledged=False):
 """Validate saved bytes, schemas, payloads and generator offsets; never build a plant."""
 import numpy as np,jsonschema
 verified=verify_external(folder,acknowledged);root=Path(folder);rows=[];array_count=0;rng_count=0
 manifest=load(ROOT/'manifests/continuation_external.json')['files']
 for name in sorted(manifest):
  if not name.endswith('/state.json'):continue
  d=root/Path(name).parent;m=load(d/'state.json');fm=load(d/'manifest.json')
  kind='prefix'if name.startswith('prefix/')else'extension';jsonschema.Draft202012Validator(load(root/'schemas'/(kind+'.json'))).validate(m)
  if set(fm['files'])!={'state.json','state_arrays.npz'}:raise ValueError('Unexpected restart manifest')
  for f,v in fm['files'].items():
   if {'bytes':(d/f).stat().st_size,'sha256':sha(d/f)}!=v:raise ValueError('Restart member identity mismatch')
  with np.load(d/'state_arrays.npz',allow_pickle=False)as z:
   identities={}
   for k in z.files:
    v=z[k]
    if v.dtype.hasobject or not np.isfinite(v).all():raise ValueError('Invalid typed state array')
    identities[k]={'dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes(order='C')).hexdigest()}
   if identities!=m['arrays_identity']:raise ValueError('Logical state-array identity mismatch')
   ep=m['episode']
   if int(z['episode'])!=ep or m['next_episode']!=ep+1 or m['plant_steps']!=28*ep:raise ValueError('Restart counter mismatch')
   if z['history.states'].shape!=(ep,29,28)or z['history.inputs'].shape!=(ep,28):raise ValueError('Restart history shape mismatch')
  meta={k:v for k,v in m.items()if k!='arrays_identity'}
  payload=hashlib.sha256(json.dumps({'arrays':identities,'meta':meta},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
  if payload!=fm['payload_sha256']:raise ValueError('Restart payload digest mismatch')
  for k,key in m['training_keys'].items():
   g=np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)));g.bit_generator.advance(m['offsets'][k])
   if g.bit_generator.state!=m['rng_states'][k]:raise ValueError('Saved random stream/offset disagreement')
   rng_count+=1
  array_count+=len(identities);rows.append({'alias':str(Path(name).parent),'episode':ep,'method':m['method'],'logical_arrays':len(identities),'schema_valid':True,'payload_sha256':payload})
 return {'status':'passed','input_files':verified['verified_files'],'saved_snapshots':len(rows),'logical_arrays':array_count,'rng_offset_checks':rng_count,'snapshots':rows,'model_constructed':False,'optimizer_restored':False,'training_executed':False,'numerical_continuation_roundtrip_claim':False,'redistribution_rights_granted':False}
