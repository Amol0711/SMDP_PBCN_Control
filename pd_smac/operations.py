"""Separate numerical training, saved-policy evaluation and aggregation paths."""
from __future__ import annotations
from pathlib import Path
import importlib, json, time
import numpy as np
from .runtime import ROOT, load, dump, configuration, protocol_hash, verify_inputs
from .lactose_first_arrival.io_contract import npz

def matched(actual, expected, tolerance=1e-10):
    checked=0; maximum=0.0
    for k,v in actual.items():
        if k not in expected or isinstance(v,(dict,list,str)):continue
        e=expected[k]
        if v is None or e is None:
            if v is not None or e is not None:raise AssertionError('Null identity mismatch: '+k)
        elif isinstance(v,(bool,np.bool_)):
            if bool(v)!=bool(e):raise AssertionError('Flag mismatch: '+k)
        else:
            error=abs(float(v)-float(e));maximum=max(maximum,error)
            if error>tolerance*max(1.,abs(float(e))):raise AssertionError((k,v,e,error))
        checked+=1
    return checked,maximum

def bio_tapes(module, records):
    """Same key per checkpoint/seed/rollout; generate only requested tapes."""
    cfg=module.CONFIG;nr=cfg['evaluation']['rollouts_per_seed'];H=cfg['training']['episode_horizon'];N=cfg['model']['state_dimension'];E=cfg['training']['episodes'];tapes={}
    for ep,seed in sorted({(r['episode'],r['seed']) for r in records}):
        phase='endpoint' if ep==E else 'checkpoint';coord=0 if ep==E else ep
        plant=np.empty((nr,H,N));reset=np.empty(nr)
        for j in range(nr):
            plant[j]=module.rng(module.key(phase+'_plant',seed=seed,checkpoint=coord,rollout=j)).random((H,N))
            reset[j]=module.rng(module.key(phase+'_reset',seed=seed,checkpoint=coord,rollout=j)).random()
        tapes[(ep,seed)]=(plant,reset,f'e{ep:03d}_seed{seed:02d}')
    return tapes

def train(study, out, seed, method, profile='full', inputs=None, acknowledged=False, budget=None):
    cfg=configuration(study)
    if profile=='smoke' and not 900<=seed<=909:raise ValueError('Smoke runs require isolated labels 900--909.')
    if profile=='full' and seed not in cfg['training']['seed_labels']:raise ValueError('Seed label outside the declared full profile.')
    if study=='four_state':
        if method not in (None,'PD_SMAC'): raise ValueError('Four-state training uses only PD_SMAC.')
        from .four_state.model import build
        from .four_state.run_study import run_seed, tapes
        from .four_state.training import core
        m=build()
        if profile=='full':
            result=run_seed(out,m,cfg,seed,'portable_replay',protocol_hash(study))
            return {'study':study,'profile':profile,'seed':seed,'endpoints':result,'record':'seed_%02d/training_records.npz'%seed}
        # A 300-episode debug run reaches exactly one declared checkpoint.
        a,keys=tapes(seed,300);r=core(m.plant,m.actions,m.target,a['reset'],a['actor'],a['plant'],300)
        npz(out/'training.npz',events_i=r[0],events_f=r[1],plant_states=r[2],episode_i=r[3],episode_f=r[4],W=r[5][:1],h=r[6][:1],checkpoint_f=r[7][:1],counters=r[8][:1])
        dump(out/'rng_keys.json',keys)
        return {'study':study,'profile':'isolated_smoke','seed':seed,'episodes':300,'plant_steps':len(r[2]),'decisions':len(r[0]),'active_results_updated':False}
    if study=='lactose_first_arrival':
        from .lactose_first_arrival.model import build_model
        from .lactose_first_arrival.learners import run_learner
        if method not in ['Macro_Q','PD_SMAC_FA']:raise ValueError('First-arrival method must be Macro_Q or PD_SMAC_FA.')
        episodes=20 if profile=='smoke' else cfg['training']['episodes']
        checkpoints=(episodes,) if profile=='smoke' else tuple(cfg['training']['checkpoint_episodes'])
        result=run_learner(build_model(),4 if method=='Macro_Q' else 5,seed,out/'training',episodes,checkpoints,evaluate_checkpoints=False)
        result['profile']=profile;result['evaluations_executed']=False
        return result
    if study in ['melanoma28','melanoma70']:
        if profile!='full':raise ValueError('Biological kernels are exposed only with their complete declared episode schedule.')
        verify_inputs(study,inputs,acknowledged)
        mod=importlib.import_module('.'+study+'.run_study',__package__);mod.configure_cpu()
        model=mod.Model(Path(inputs),cfg)
        if method not in ['DDQN_PER','TT_AC','PD_SMAC']:raise ValueError('Invalid biological method.')
        if method=='PD_SMAC':
            allowed=cfg['training'].get('budget_sweep',[cfg['objectives'].get('main_budget',.4)])
            # Validate against explicit configurations even when budget-sweep is in objectives.
            allowed=[.25,.4,.55] if study=='melanoma28' else [.4]
            if budget not in allowed:raise ValueError('Budget outside declared study values.')
        else:budget=0.
        tapes=bio_tapes(mod,[{'episode':ep,'seed':seed} for ep in cfg['training']['checkpoint_episodes']])
        result=mod.train(out,method,seed,budget,model,tapes)
        result['output_contains_biological_trajectories']=True
        result['public_redistribution_authorized']=False
        return result
    raise ValueError('Use meanfield for the deterministic lactose calculation.')

def meanfield(out, iterations=620):
    if iterations not in [2,620]:raise ValueError('Use 2 debug updates or the complete 620-update profile.')
    from .lactose_meanfield.model import build_model, strict_feasibility_witness
    from .common.actor_geometry import project_zero_sum_box
    from .common.actor_steps import meanfield_step,multiplier_step
    from .common.actor_residuals import DiagnosticContext,evaluate_tabular_smdp,continuing_diagnostics
    cfg=configuration('lactose_meanfield');m=build_model();key=[20260917,11,20,3,0,250,0,0,0]
    initial=np.random.Generator(np.random.PCG64(np.random.SeedSequence(key))).normal(0,.001,size=m.reward.shape)
    w=project_zero_sum_box(initial,12);lam=cfg['training']['initial_lambda'];W=[];V=[];D=[];G=[];L=[];rows=[]
    for i in range(iterations+1):
        ctx=DiagnosticContext('lactose_meanfield',protocol_hash('lactose_meanfield'),f'iteration{i}','lactose_tabular_meanfield_v1','target_occupancy_minus_0.01_input_L1_rate')
        e=evaluate_tabular_smdp(w,m.transitions,m.reward,m.duration,m.expenditure,multiplier=lam,budget=.25,context=ctx,reference_state=0)
        d=continuing_diagnostics(e,multiplier_upper=20,dual_recursion='rate')
        weight=e.stationary[:,None]*e.probabilities
        rows.append({'iteration':i,'eta':e.reward_rate,'occupancy':float(np.sum(weight*m.occupancy)/e.mean_duration),'input_exposure_rate':float(np.sum(weight*m.exposure)/e.mean_duration),'update_rate':e.update_rate,'rate_residual':e.update_rate-.25,'Tbar':e.mean_duration,'lambda':lam,**d['residuals'],**e.checks})
        W.append(w.copy());V.append(e.value.copy());D.append(e.stationary.copy());G.append(e.gradient_rate.copy());L.append(lam)
        if i<iterations:
            w=meanfield_step(w,e.gradient_rate,10.,12.);lam=multiplier_step(lam,e.update_rate-.25,.5,20.)
    npz(out/'all_iterates.npz',theta=W,value=V,stationary=D,gradient=G,initial_unprojected=initial,multiplier=L,iteration=np.arange(iterations+1))
    dump(out/'trace.json',rows);dump(out/'witness.json',strict_feasibility_witness(m)[1])
    checks={}; saved=np.load(ROOT/'results/lactose_meanfield/all_iterates.npz',allow_pickle=False)
    with np.load(out/'all_iterates.npz',allow_pickle=False)as actual:
        for name in actual.files:
            expected=saved[name] if name=='initial_unprojected' else saved[name][:iterations+1]
            error=float(np.max(np.abs(actual[name]-expected)))
            checks[name]={'max_abs_difference':error,'array_equal':bool(np.array_equal(actual[name],expected))}
            # Existing independent value/gradient agreement tolerance, not a bitwise claim.
            if error>1e-8:raise AssertionError(('saved mean-field comparison',name,error))
    if max(r['stationary_equation_abs'] for r in rows)>1e-10 or max(r['value_equation_abs'] for r in rows)>1e-9: raise AssertionError('Mean-field solve residual exceeded its declared threshold.')
    dump(out/'array_regression.json',checks)
    return {'iterations':iterations,'endpoint':rows[-1],'active_results_updated':False,'seed_key':key,'array_regression':checks,'agreement_tolerance':1e-8,'bitwise_reproduction':all(c['array_equal']for c in checks.values()),'near_actor_stationarity':rows[-1]['actor_stationarity_rate']<=1e-5,'kkt_certificate':False}


def _four_value(r):
    from .four_state.model import build,evaluate,deterministic
    with np.load(ROOT/r['policy'],allow_pickle=False)as z:
        index=int(np.flatnonzero(z['episodes']==r['episode'])[0])
        pi=z['probabilities'][index] if r['mode']=='stochastic' else deterministic(z['greedy_actions'][index])
    e=evaluate(build(),pi)
    return {k:e[k]for k in ['occupancy','reward','rate','mean_duration','exposure_rate','rate_residual','feasible']}

def selected_records(study='all', scope='endpoint'):
    return [r for r in load(ROOT/'results/evaluation_index.json') if (study=='all' or r['study']==study) and(scope=='all' or r['endpoint'])]

def evaluate_saved(study,out,scope='endpoint',inputs=None,acknowledged=False,limit=None):
    rows=selected_records(study,scope)
    if limit is not None:
        if limit <= 0: raise ValueError('Evaluation limit must be positive.')
        rows=rows[:limit]
    if not rows:raise ValueError('No matching saved policies.')
    models={};tapes={};results=[];scalar_checks=0;count_checks=0
    for s in sorted({r['study']for r in rows}):
        if s.startswith('melanoma'):
            verify_inputs(s,inputs,acknowledged);mod=importlib.import_module('.'+s+'.run_study',__package__)
            # Torch interop count is process-global; configure only once.
            if not any(k.startswith('melanoma')for k in models):mod.configure_cpu()
            models[s]=(mod,mod.Model(Path(inputs),configuration(s)));tapes[s]=bio_tapes(mod,[r for r in rows if r['study']==s])
        elif s=='lactose_first_arrival':
            from .lactose_first_arrival.model import build_model
            models[s]=build_model()
    for number,r in enumerate(rows):
        if r['study']=='four_state':v=_four_value(r)
        elif r['study']=='lactose_first_arrival':
            from .lactose_first_arrival.learners import evaluate_rollouts
            with np.load(ROOT/r['policy'],allow_pickle=False)as z:w=z['parameters']
            target=out/f'case{number:04d}'
            result=evaluate_rollouts(models[r['study']],w,r['method_code'],r['mode'],r['seed'],r['episode'],target);v=result['metrics']
            with np.load(target.with_suffix('.npz'),allow_pickle=False)as a,np.load(ROOT/r['counts'],allow_pickle=False)as b:
                for k in b.files:
                    if not np.array_equal(a[k],b[k]):raise AssertionError((r['id'],k,'saved-policy trajectory-count mismatch'))
                    count_checks+=1
        else:
            mod,m=models[r['study']];q=None
            if r['method']=='fixed':actor=r['fixed_action']
            else:
                with np.load(ROOT/r['policy'],allow_pickle=False)as z:
                    actor=z['actor']
                    if r['method']=='DDQN_PER':
                        import torch
                        q=mod.QNet();q.load_state_dict({k[7:]:torch.from_numpy(z[k])for k in z.files if k.startswith('online.')});q.eval()
            folder=out/f'case{number:04d}'/'evaluation'
            result=mod.evaluate(folder,r['episode'],r['seed'],r['method'],r['budget'],actor,m,tapes[r['study']],q,r['mode']);v=result['metrics']
            with np.load(folder/result['raw_file'],allow_pickle=False)as a,np.load(ROOT/r['counts'],allow_pickle=False)as b:
                if not np.array_equal(a['counts'],b['counts']):raise AssertionError(r['id']+' count mismatch')
                count_checks+=1
        n,error=matched(v,r['expected']);scalar_checks+=n
        results.append({'id':r['id'],'metrics':v,'max_scalar_difference':error})
    dump(out/'evaluation_results.json',results)
    return {'status':'passed','policies_evaluated':len(rows),'scalar_checks':scalar_checks,'count_array_checks':count_checks,'scope':scope,'biological_trajectories_generated':any(r['study'].startswith('melanoma')for r in rows),'active_results_updated':False}

def aggregate(out):
    from .statistics_core import biological_metrics,first_arrival_metrics,summarize
    rows=load(ROOT/'results/evaluation_index.json');values=[];checks=0;maxerror=0.;rollouts=0;allgroups={}
    for r in rows:
        if r['study']=='four_state':v=_four_value(r);individual=None
        elif r['study'].startswith('melanoma'):
            with np.load(ROOT/r['counts'],allow_pickle=False)as z:c=z['counts'];v,individual=biological_metrics(c,r['budget']);rollouts+=len(c)
            v['empirical_feasible']=v['residual']<=1e-12
        else:
            with np.load(ROOT/r['counts'],allow_pickle=False)as z:
                c={'T':z['T_cap'],'K':z['updates'],'S':z['scheduled_duration_total'],'A':z['input_exposure'],'hit':z['hit']};v,ex=first_arrival_metrics(c);rollouts+=len(c['T'])
            v.update({k:ex[k]for k in ['mean_individual_rate_positive_time','initial_target_count','failure_count']});v['feasible']=v['budget_numerator']<=1e-12;individual=None
        n,error=matched(v,r['expected']);checks+=n;maxerror=max(maxerror,error)
        values.append({'id':r['id'],'metrics':v})
        key=(r['study'],r['method'],r['mode'],r['budget'] if r['method'].startswith('PD') else None,r['episode'],tuple(r.get('fixed_action',[])))
        allgroups.setdefault(key,[]).append((r,v,individual))
    groups=[];canon=load(ROOT/'results/canonical_summary.json');reference_checks=0
    for key,items in allgroups.items():
        study,method,mode,budget,ep,fixed=key;items.sort(key=lambda x:x[0]['seed']);stats={}
        if method=='fixed':
            ids=np.load(ROOT/'results'/study/'bootstrap_indices.npz',allow_pickle=False)['fixed']
            names=[k for k in items[0][2] if k!='residual']
            for name in names:stats[name]=summarize(np.concatenate([x[2][name]for x in items]),ids,ddof=0)
            unit='rollouts within one fixed policy; population SD'
            ref=canon[study]['fixed'][f'u{fixed[0]}_tau{fixed[1]}']
        else:
            bp=ROOT/'results'/study/('bootstrap_indices.npy'if study=='four_state'else'bootstrap_indices.npz')
            zz=np.load(bp,allow_pickle=False);ids=zz if study=='four_state'else zz['training'if study.startswith('melanoma')else'indices']
            names=[k for k,v in items[0][1].items()if isinstance(v,(float,np.floating))and k not in ['mean_individual_rate_positive_time']]
            for name in names:stats[name]=summarize([x[1][name]for x in items],ids,ddof=1)
            unit='training-seed means; sample SD'
            if study=='four_state':ref=canon[study]['groups'][mode]
            elif study.startswith('melanoma'):ref=canon[study]['groups'][f'{method}_g{round(1000*(budget or 0)):03d}_{mode}']
            else:ref=canon[study]['groups'][method+'_'+mode]
        if items[0][0]['endpoint'] or 'checkpoints' in ref:
            if 'checkpoints' in ref: ref=ref['checkpoints'][str(ep)]
            refmetrics=ref['statistics']
            for name,stat in stats.items():
                if name not in refmetrics or not isinstance(refmetrics[name],dict):continue
                for field in ['mean','sd']:
                    if field in refmetrics[name]:
                        matched({field:stat[field]},{field:refmetrics[name][field]});reference_checks+=1
                if 'bootstrap95'in refmetrics[name]and refmetrics[name]['bootstrap95']is not None:
                    if np.max(abs(np.array(stat['bootstrap95'])-refmetrics[name]['bootstrap95']))>1e-10:raise AssertionError((key,name,'interval'))
                    reference_checks+=1
        groups.append({'study':study,'method':method,'mode':mode,'budget':budget,'episode':ep,'fixed_action':list(fixed),'unit':unit,'statistics':stats})
    dump(out/'conditional_evaluations.json',values);dump(out/'statistical_groups.json',groups)
    return {'status':'passed','conditional_evaluations':len(rows),'rollouts_reaggregated':rollouts,'groups':len(groups),'metric_scalar_checks':checks,'canonical_endpoint_summary_checks':reference_checks,'max_scalar_difference':maxerror,'training_executed':False,'plant_simulation_executed':False}
