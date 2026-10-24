"""Complete-hold restart snapshots for isolated software validation."""
from pathlib import Path
import json,copy
import numpy as np
from .contracts import digest,info,read,verify_protocol_object,execution_keys
from .learner import Parameters,TrainingState,validate_state

def from_serial(raw,protocol):
 x=copy.deepcopy(raw);p=x.pop('parameters');states=x.pop('rng_states');rngs={}
 for name,key in x['seed_keys'].items():
  g=np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)));g.bit_generator.advance(x['offsets'][name])
  if g.bit_generator.state!=states[name]:raise ValueError('RNG offset/state mismatch '+name)
  rngs[name]=g
 state=TrainingState(parameters=Parameters(np.array(p['W'],dtype=np.float64),np.array(p['v'],dtype=np.float64),p['eta'],p['multiplier']),rngs=rngs,**x)
 validate_state(state,protocol)
 if state.serial()!=raw:raise ValueError('complete state roundtrip mismatch')
 return state

def save(path,state,protocol):
 verify_protocol_object(protocol)
 if state.mode!='functional_test':raise PermissionError('Only isolated functional-validation snapshots may be written.')
 path=Path(path)
 if path.exists():raise FileExistsError(path)
 path.mkdir(parents=True)
 record={'schema_version':'continuing-validation-state-1','state':state.serial(),'state_sha256':state.identity(),'original_protocol_sha256':protocol.sha256,'new_scientific_seed':False,'mode':'functional_test'}
 (path/'state.json').write_text(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+'\n')
 (path/'manifest.json').write_text(json.dumps({'files':{'state.json':info(path/'state.json')}},indent=2)+'\n')
 return record

def restore(path,protocol,seed=None):
 path=Path(path);manifest=read(path/'manifest.json')
 if set(p.name for p in path.iterdir())!={'state.json','manifest.json'}:raise ValueError('unexpected snapshot contents')
 if manifest!={'files':{'state.json':info(path/'state.json')}}:raise ValueError('restart file identity changed')
 r=read(path/'state.json')
 if r['original_protocol_sha256']!=protocol.sha256 or r['schema_version']!='continuing-validation-state-1':raise ValueError('wrong protocol/schema')
 if r['state']['mode']!='functional_test' or r['mode']!='functional_test':raise PermissionError('No production restart is enabled here.')
 if seed is not None and r['state']['seed_label']!=seed:raise ValueError('wrong restart seed')
 state=from_serial(r['state'],protocol)
 if state.identity()!=r['state_sha256']:raise ValueError('logical state hash mismatch')
 return state
