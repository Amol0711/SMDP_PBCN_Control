"""Small-model reference computations and numerical certificate checks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .runtime import ROOT,load,dump
from .lactose_first_arrival.io_contract import npz

def first_arrival_forward(m,policy,H=100):
    mass=np.zeros((H+1,512));mass[H]=1/512;mass[:,m.target]=0;tot=np.zeros(5);occ=np.zeros((H+1,512,20))
    for r in range(H,0,-1):
        pi=np.eye(20)[policy[r]] if policy.ndim==2 else policy[r]
        occ[r]=mass[r,:,None]*pi
        for a in range(20):
            w=occ[r,:,a];ell,fail,_,cont=m.terms(r,a)
            tot[0]+=w@ell;tot[1]+=w.sum();tot[2]+=w@(ell*sum(m.actions[a,:2]));tot[3]+=w@fail;tot[4]+=(a%5+1)*w.sum()
            if cont is not None:mass[r-a%5-1]+=cont.T@w
    return tot,occ

def check_first_arrival(out):
    from .lactose_first_arrival.model import build_model
    from .lactose_first_arrival.references import evaluate
    from .operations import matched
    from scipy.sparse import csr_matrix
    m=build_model();folder=ROOT/'results/lactose_first_arrival/references';expected=load(folder/'metrics.json');results={}
    for name in ['FA_TT_exact','FA_macro_exact','FA_macro_constrained_exact']:
        with np.load(folder/(name+'.npz'),allow_pickle=False)as z:
            V,metrics=evaluate(m,z['policy']);tot,occ=first_arrival_forward(m,z['policy'])
            if np.max(abs(tot-V[100].mean(0)))>1e-8:raise AssertionError('Forward/backward disagreement')
            count,error=matched(metrics,expected[name],1e-8)
            detail={'metrics':metrics,'scalar_checks':count,'max_scalar_difference':error,'forward_backward_error':float(np.max(abs(tot-V[100].mean(0))))}
            if name=='FA_macro_constrained_exact':
                pairs=z['pairs'];y=z['occupation'];A=csr_matrix((z['flow_data'],z['flow_indices'],z['flow_indptr']),shape=tuple(z['flow_shape']));b=(pairs[:,0]==100)/512.;g=z['budget_coefficients'];c=z['cost']
                yy=occ[pairs[:,0],pairs[:,1]].ravel()
                with np.load(folder/'secondary_independent_dual.npz',allow_pickle=False)as dual:
                    lam,mu=dual['constraint_multipliers'];v=dual['flow_dual'];red=np.ones(len(y))-A.T@v+lam*g+mu*c
                    primary=float(c@z['primary_occupation']);gap=float(z['expected_fields'][:,1]@y-(b@v-mu*(primary+1e-10)))
                detail.update({'flow_error':float(np.max(abs(A@y-b))),'recovered_occupation_error':float(np.max(abs(yy-y))),'budget_violation':float(max(0,g@y)),'dual_reduced_cost_violation':float(max(0,-red.min())),'secondary_dual_gap':gap,'LP_columns_checked':len(y)})
                if detail['flow_error']>1e-9 or detail['recovered_occupation_error']>1e-9 or detail['budget_violation']>1e-10 or detail['dual_reduced_cost_violation']>1e-8 or abs(gap)>1e-8:raise AssertionError(detail)
            results[name]=detail
    dump(out/'references.json',results)
    return {'status':'passed','references_evaluated':3,'LP_columns_checked':340880,'optimization_executed':False}

def four_state_references(out):
    from .four_state.model import build
    from .four_state.references import enumerate_all,solve_lp
    from .operations import matched
    m=build();rows,summary=enumerate_all(m,out);lps={name:solve_lp(m,name,rows,out)for name in ['occupancy','penalized_reward']}
    expected=load(ROOT/'results/four_state/references/enumeration_summary.json')
    for key in ['count','feasible_count','reducible_count','multichain_count']:
        if summary[key]!=expected[key]:raise AssertionError('Enumeration count changed: '+key)
    for name,x in lps.items():matched({k:x[k]for k in ['optimal_value','lambda','rho']},load(ROOT/f'results/four_state/references/lp_{name}.json'),1e-8)
    return {'status':'passed','enumerated_policies':len(rows),'LP_objectives':list(lps),'optimization_executed':True}
