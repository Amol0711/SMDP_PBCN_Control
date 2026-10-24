"""Independent numerical operations. No command updates saved scientific evidence."""
from __future__ import annotations
import copy,csv,json,math,re,shutil,subprocess,time
from pathlib import Path
from ..runtime import ROOT,load,dump,sha,verify_inputs
SAVED=ROOT/'results/diagnostics'


def compare_values(actual,expected,atol=1e-10,rtol=1e-10,path=''):
 """Compare numerical/structural values; return counts and exact/tolerance agreement."""
 summary={'numeric_leaves':0,'exact_numeric_leaves':0,'max_abs_error':0.,'all_within_tolerance':True}
 def visit(a,b,p):
  if isinstance(b,dict):
   if not isinstance(a,dict)or set(a)!=set(b):raise AssertionError('key mismatch '+p)
   for k in b:visit(a[k],b[k],p+'/'+k)
  elif isinstance(b,list):
   if not isinstance(a,list)or len(a)!=len(b):raise AssertionError('length mismatch '+p)
   for i,(u,v)in enumerate(zip(a,b)):visit(u,v,p+'/'+str(i))
  elif isinstance(b,bool)or b is None or isinstance(b,str):
   if type(a)!=type(b)or a!=b:raise AssertionError('metadata/null mismatch '+p)
  elif isinstance(b,(int,float)):
   if isinstance(a,bool)or not isinstance(a,(int,float)):raise AssertionError('numeric type mismatch '+p)
   if not math.isfinite(a)or not math.isfinite(b):raise AssertionError('nonfinite '+p)
   e=abs(a-b);summary['numeric_leaves']+=1;summary['exact_numeric_leaves']+=int(a==b);summary['max_abs_error']=max(summary['max_abs_error'],e)
   if e>atol+rtol*abs(b):raise AssertionError((p,a,b,e,atol,rtol))
  elif a!=b:raise AssertionError('value mismatch '+p)
 visit(actual,expected,path);summary['exact']=summary['numeric_leaves']==summary['exact_numeric_leaves'];return summary


def witness(out):
 from fractions import Fraction as F
 from .exact_arithmetic import exact_model,witness_policy,evaluate,I,BITS
 from .certificates import exact_lp_certificate
 frozen=load(SAVED/'witness.json');model=exact_model(ROOT/'pd_smac/four_state/config.json');p=F(frozen['conditional_p_exact'])
 result=evaluate(witness_policy(p),model)
 enclosure={'metrics':{k:v.dump()for k,v in result['metrics'].items()},'stationary':[v.dump()for v in result['stationary']],
 'tree_cofactors':[v.dump()for v in result['tree_cofactors']],'occupation_measure':[[v.dump()for v in row]for row in result['occupation_measure']]}
 if enclosure!=frozen['enclosure']:raise AssertionError('witness interval output differs from frozen numerical evidence')
 if not result['metrics']['rate'].upper<F(3,10)-F('5e-9'):raise AssertionError('witness budget margin')
 certs={}
 for obj in ['occupancy','penalized_reward']:
  cert,value=exact_lp_certificate(model,load(SAVED/f'lp_{obj}.json'));certs[obj]=cert
 dump(out/'witness_enclosure.json',enclosure);dump(out/'active_support_certificates.json',certs)
 return {'status':'passed','interval_bits':BITS,'witness_interval_record_exact':True,'saved_active_support_certificates':len(certs),'fresh_optimization':False,
 'training_executed':False,'learned_policy_claim':False,'new_statistical_seeds':0}


def diagnostics(out):
 from .continuing.contracts import load_protocol
 from .continuing.learner import initialize
 from .continuing.model import Plant
 from .continuing.evaluator import evaluate_checkpoint
 p=load_protocol();s=initialize(p,900);before=s.identity();record=evaluate_checkpoint(s,Plant.from_protocol(p),p)
 if s.identity()!=before:raise AssertionError('diagnostics mutated state')
 dump(out/'diagnostics.json',record)
 return {'status':'passed','initial_state_diagnostic':True,'stochastic_and_greedy_separate':True,'training_steps':0,'state_rng_unchanged':True,
 'exact_model_float64_not_exact_arithmetic':True,'convergence_certificate':False,'new_statistical_seeds':0}


def continuing_train(out,seed=900,decisions=128,resume=None):
 import numpy as np
 from .continuing.contracts import load_protocol,debug_limit
 from .continuing.learner import initialize,advance,validate_state
 from .continuing.model import Plant
 from .continuing.io import restore,save
 from .continuing.evaluator import evaluate_checkpoint
 debug_limit(seed,decisions);p=load_protocol();s=initialize(p,seed)if resume is None else restore(resume,p,seed)
 if s.k>=decisions:raise ValueError('Stop must exceed the complete-hold restart boundary.')
 start=s.k;plant=Plant.from_protocol(p);ii=[];ff=[];states=[]
 while s.k<decisions:
  i,f,h,extra=advance(s,plant,p);ii.append(i);ff.append(f);states.extend(h.post_states)
 validate_state(s,p);np.savez_compressed(out/'trace.npz',integers=np.array(ii),floats=np.array(ff),states=np.array(states,dtype=np.int64))
 record=save(out/'checkpoint',s,p);before=s.identity();ev=evaluate_checkpoint(s,plant,p)
 if s.identity()!=before:raise AssertionError('evaluation interfered with training state')
 dump(out/'diagnostics.json',ev)
 return {'status':'passed','scope':'isolated_software_validation','seed_label':seed,'start_decision':start,'end_decision':s.k,'complete_holds_executed':s.k-start,
 'total_plant_steps':s.plant_time,'state_sha256':s.identity(),'restart':resume is not None,'production_schedule_changed':False,'new_statistical_seeds':0}


def saved_state(item,protocol):
 import numpy as np
 from .continuing.io import from_serial
 folder=ROOT/item['path'];m=load(folder/'state.json');raw=copy.deepcopy(m['state'])
 raw['parameters']={'W':np.load(folder/'W.npy',allow_pickle=False).tolist(),'v':np.load(folder/'v.npy',allow_pickle=False).tolist(),**m['parameter_scalars']}
 s=from_serial(raw,protocol)
 if s.identity()!=m['state_sha256']:raise ValueError('frozen logical state identity mismatch')
 return s


def continuing_evaluate(out,scope='all'):
 from .continuing.contracts import load_protocol
 from .continuing.model import Plant
 from .continuing.evaluator import evaluate_checkpoint
 p=load_protocol();plant=Plant.from_protocol(p);items=load(SAVED/'continuing/index.json');results=[]
 if scope=='endpoint':items=[x for x in items if x['decision']==1000000]
 for x in items:
  s=saved_state(x,p);before=s.identity();record=evaluate_checkpoint(s,plant,p);expected=load(ROOT/x['path']/'expected_diagnostics.json')
  summary=[]
  for mode in ['stochastic','greedy']:
   for field in ['metrics','residuals','exact_evaluation','scope','status','unavailable_reason']:
    summary.append(compare_values(record[mode][field],expected[mode][field],path=mode+'/'+field))
  for field in ['critic_tracking','stochastic_auxiliary_metrics','training_rng_consumption']:summary.append(compare_values(record[field],expected[field],path=field))
  if before!=s.identity():raise AssertionError('frozen evaluator modified complete state')
  name=f'seed_{x["seed"]:02d}_k{x["decision"]:09d}';dump(out/(name+'.json'),record)
  results.append({'seed':x['seed'],'decision':x['decision'],'numeric_leaves':sum(t['numeric_leaves']for t in summary),'exact_numeric_leaves':sum(t['exact_numeric_leaves']for t in summary),'max_abs_error':max(t['max_abs_error']for t in summary),'all_within_tolerance':True,'state_rng_unchanged':True})
 dump(out/'comparisons.json',results)
 return {'status':'passed','frozen_states':len(items),'policy_evaluations':2*len(items),'numeric_leaves':sum(x['numeric_leaves']for x in results),
 'exact_numeric_leaves':sum(x['exact_numeric_leaves']for x in results),'max_abs_error':max(x['max_abs_error']for x in results),'absolute_tolerance':1e-10,'relative_tolerance':1e-10,
 'training_steps':0,'state_rng_unchanged':True,'new_statistical_seeds':0}


def aggregate(out):
 import numpy as np
 rows=load(SAVED/'continuing/scalar_records.json');indices=np.load(SAVED/'continuing/bootstrap_indices.npy',allow_pickle=False)
 original=list(csv.DictReader((SAVED/'scalar_tables/continuing_curves.csv').open()));results=[];compared=0;exact=0;maxerr=0.
 from .continuing.contracts import load_protocol
 from .continuing.model import Plant,macro_arrays
 mac=macro_arrays(Plant.from_protocol(load_protocol()));greedy_occupancy={}
 for item in load(SAVED/'continuing/index.json'):
  folder=ROOT/item['path'];d=load(folder/'expected_diagnostics.json')['greedy']['exact_evaluation']['stationary']
  W=np.load(folder/'W.npy',allow_pickle=False);pi=np.eye(10)[np.argmax(W,axis=1)];weights=np.asarray(d)[:,None]*pi
  greedy_occupancy[(item['seed'],item['decision'])]=float(np.sum(weights*mac['occupancy'])/np.sum(weights*mac['duration']))
 metrics={
 'reward':lambda r,m:r[m]['reward_rate'],'rate':lambda r,m:r[m]['update_rate'],
 'lambda':lambda r,m:r[m]['multiplier'],'R_act':lambda r,m:r[m+'_residuals']['R_act'],
 'R_adv':lambda r,m:r[m+'_residuals']['R_adv'],'R_pri':lambda r,m:r[m+'_residuals']['R_pri'],
 'R_comp':lambda r,m:r[m+'_residuals']['R_comp'],
 'occupancy':lambda r,m:r['auxiliary']['occupancy']if m=='stochastic'else greedy_occupancy[(r['seed'],r['decision'])],
 'mean_duration':lambda r,m:r[m]['mean_duration'],
 'rate_tracking_abs':lambda r,m:r['critic_tracking']['rate_abs_error']if m=='stochastic'else None,
 'value_tracking_norm2':lambda r,m:r['critic_tracking']['value_norm2_error']if m=='stochastic'else None,
 'critic_rate_abs_error':lambda r,m:r['critic_tracking']['rate_abs_error']if m=='stochastic'else None,
 'critic_value_norm2_error':lambda r,m:r['critic_tracking']['value_norm2_error']if m=='stochastic'else None}
 for item in original:
  metric=item['metric'];match=re.fullmatch(r'continuing_summary/(stochastic|greedy)/k(\d+)',item['context_id'])
  if match is None or metric not in metrics:continue
  mode=match[1];k=int(match[2]);sample=[metrics[metric](r,mode)for r in sorted(rows,key=lambda z:z['seed'])if r['decision']==k]
  if any(v is None for v in sample):continue
  a=np.asarray(sample,dtype=np.float64);boot=np.mean(a[indices],axis=1);q=np.quantile(boot,[.025,.975],method='linear')
  observed={'value':float(np.mean(a)),'sample_sd':float(np.std(a,ddof=1)),'ci95_lower':float(q[0]),'ci95_upper':float(q[1]),'n':len(a)}
  checks={}
  for key,v in observed.items():
   old=float(item[key]);err=abs(v-old)
   if err>1e-10+1e-10*abs(old):raise AssertionError((item['context_id'],metric,key,v,old))
   checks[key]={'exact':v==old,'abs_error':err};compared+=1;exact+=int(v==old);maxerr=max(maxerr,err)
  results.append({'context':item['context_id'],'metric':metric,**observed,'checks':checks})
 if len(results)!=len(original):raise AssertionError('not all declared continuing-curve groups were aggregated')
 dump(out/'continuing_aggregation.json',results)
 return {'status':'passed','groups':len(results),'compared_fields':compared,'exact_fields':exact,'max_abs_error':maxerr,'absolute_tolerance':1e-10,'relative_tolerance':1e-10,
 'frozen_bootstrap_indices_reused':True,'new_bootstrap_sample':False,'new_statistical_seeds':0,'scope':'all continuing-curve groups; the other nine retained scalar tables are preserved, not recomputed'}


def figures(out,only=None):
 src=ROOT/'figures';names=load(src/'index.json')
 if only is not None:
  if only not in names:raise ValueError('unknown native numerical figure')
  names=[only]
 if not shutil.which('pdflatex'):raise RuntimeError('Installed pdfLaTeX, standalone, TikZ and PGFPlots required.')
 shutil.copytree(src/'data',out/'data');shutil.copy2(src/'preamble.tex',out)
 results=[]
 for name in names:
  for suffix in ['.tex','.tikz']:shutil.copy2(src/(name+suffix),out)
  for rep in range(3):
   q=subprocess.run(['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=out,capture_output=True,text=True,timeout=90)
   (out/f'{name}_pass{rep+1}.stdout.txt').write_text(q.stdout);(out/f'{name}_pass{rep+1}.stderr.txt').write_text(q.stderr)
   if q.returncode:raise RuntimeError('native numerical figure failed '+name)
  log=(out/(name+'.log')).read_text(errors='replace');checked=re.sub(r'Package shellesc Warning: Shell escape disabled on input line \d+\.','',log)
  warnings=[s for s in checked.splitlines()if 'Warning'in s or 'Overfull'in s or 'undefined references'in s]
  if warnings:raise RuntimeError({'figure':name,'warnings':warnings})
  f=out/(name+'.pdf');results.append({'name':name,'sha256':sha(f),'bytes':f.stat().st_size,'other_warnings':0})
 dump(out/'figure_manifest.json',results)
 return {'status':'passed','figures':results,'native_TikZ_PGFPlots':True,'scientific_values_changed':False,'training_executed':False}


def extended_train(out,state_dir,input_dir,run_id,stop=480,acknowledged=False):
 from .gate import require_continuation
 guard=require_continuation(state_dir,run_id,acknowledged)
 verify_inputs('melanoma28',input_dir,acknowledged)
 if stop not in [480,720,960,1200]:raise ValueError('Only locked extension endpoints are supported.')
 from .biological_context import bind_state_root
 bind_state_root(state_dir)
 from .biological import prefix as C,extension as E
 from ..melanoma28 import run_study as original
 from ..melanoma28.model import Model
 model=Model(Path(input_dir),original.CONFIG);origin=Path(state_dir)/'prefix'/run_id
 L=C.load_state(origin,model);row=next(x for x in guard['matrix']['rows']if x['run_id']==run_id)
 receipt=guard['receipt'];L['lineage']={'origin_snapshot_path':row['snapshot_path'],'origin_snapshot_manifest_sha256':row['snapshot_manifest']['sha256'],
 'implementation_sha256':receipt['implementation_sha256'],'authorization_sha256':sha(Path(state_dir)/'authorization/receipt.json'),'branch':'verified_original_continuation'}
 E.extend_reserved_tapes(L);L['_authorization_checked']=True
 for endpoint in [e for e in [480,720,960,1200]if e<=stop]:
  E.advance(L,endpoint);E.save_state(out/f'e{endpoint:04d}',L)
 dump(out/'environment_gate.json',guard['environment'])
 return {'status':'passed','role':'preselected_validation_replay','run_id':run_id,'start_episode':240,'end_episode':L['ep'],
 'checkpoint_evaluations_performed':False,'new_statistical_seeds':0,'outputs_contain_biological_states':True,'public_release_authorized':False}
