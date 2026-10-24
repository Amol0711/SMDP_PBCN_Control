"""One command interface for data verification, numerical studies and saved outputs.

No numerical module is imported for help, inventory, transport verification or
saved-artifact export. Numerical commands do not modify published evidence.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from .runtime import ROOT, dump, environment, import_inputs, load, output_dir, sha, verify_inputs, verify_package

STUDIES = {
    'four_state_reset': 'four_state',
    'lactose_first_arrival': 'lactose_first_arrival',
    'melanoma_xor': 'melanoma28',
    'melanoma_persistent': 'melanoma70',
}

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('verify', help='Verify immutable package inventory; no numerical dependencies.')
    sub.add_parser('outputs', help='List article and computational-guide output routes.')
    sub.add_parser('studies', help='List scientific cases and their configurations.')
    def out(name, help):
        q = sub.add_parser(name, help=help)
        q.add_argument('--out', type=Path, required=True)
        return q
    q = out('export-output', 'Export the saved files for one output; no recalculation.')
    q.add_argument('--id', required=True)
    q = out('figures', 'Compile the six current native numerical figures.')
    q.add_argument('--only', choices=load(ROOT/'figures/index.json'))
    out('guide', 'Compile the computational guide; no article or proofs bundled.')
    out('witness', 'Verify the saved interval witness and two support certificates; no optimization.')
    out('diagnostics', 'Evaluate a fixed synthetic initialization without training.')
    q = out('aggregate', 'Recompute aggregates from retained numerical records.')
    q.add_argument('--study', choices=['reported', 'four_state_continuing'], required=True)
    q = out('evaluate', 'Evaluate saved policies; biological cases require external inputs.')
    q.add_argument('--study', choices=[*STUDIES, 'four_state_continuing'], required=True)
    q.add_argument('--scope', choices=['all','endpoint'], default='endpoint')
    q.add_argument('--input-dir', type=Path)
    q.add_argument('--acknowledge-local-use-rights', action='store_true')
    q = out('train', 'Execute a declared study, not automatically authorize a new scientific run.')
    q.add_argument('--study', choices=list(STUDIES), required=True)
    q.add_argument('--method')
    q.add_argument('--seed', type=int, default=0)
    q.add_argument('--profile', choices=['full','smoke'], default='full')
    q.add_argument('--budget', type=float)
    q.add_argument('--input-dir', type=Path)
    q.add_argument('--acknowledge-local-use-rights', action='store_true')
    q = out('continuing-check', 'Bounded software-check trajectory, separate from the scientific cohort.')
    q.add_argument('--seed', type=int, default=900)
    q.add_argument('--decisions', type=int, default=128)
    q.add_argument('--resume', type=Path)
    q = out('meanfield', 'Run the separate deterministic continuing diagnostic.')
    q.add_argument('--iterations', type=int, choices=[2,620], default=620)
    q = out('references', 'Four-state references or first-arrival saved certificate checks.')
    q.add_argument('--study', choices=['four_state_reset','lactose_first_arrival'], required=True)
    q = sub.add_parser('inputs', help='Stage or verify author-supplied biological inputs; grants no rights.')
    q.add_argument('--study', choices=['melanoma_xor','melanoma_persistent'], required=True)
    q.add_argument('--input-dir', type=Path, required=True)
    q.add_argument('--source-dir', type=Path)
    q.add_argument('--acknowledge-local-use-rights', action='store_true')
    out('eligibility', 'Check the exact historical continuation environment; no override option.')
    for name in ['restart-inputs','restart-inspect']:
        q = out(name, 'Verify external restart identities; inspection is not live restoration.')
        q.add_argument('--state-dir', type=Path, required=True)
        q.add_argument('--acknowledge-local-use-rights', action='store_true')
    q = out('continuation-replay', 'Environment- and rights-gated historical biological continuation.')
    q.add_argument('--state-dir', type=Path, required=True)
    q.add_argument('--input-dir', type=Path, required=True)
    q.add_argument('--run-id', required=True)
    q.add_argument('--stop', type=int, choices=[480,720,960,1200], default=480)
    q.add_argument('--acknowledge-local-use-rights', action='store_true')
    return p

def export_output(out: Path, output_id: str) -> dict:
    index = load(ROOT/'manifests/paper_outputs.json')['outputs']
    matches = [r for r in index if r['id']==output_id]
    if len(matches)!=1: raise ValueError('Unknown output identifier: '+output_id)
    row = matches[0]
    if not row['export_files']:
        raise ValueError('This is a noncomputational article item; see its scope entry.')
    copied=[]
    for name in row['export_files']:
        src=(ROOT/name).resolve()
        if not src.is_relative_to(ROOT) or not src.is_file():
            raise ValueError('Invalid output dependency: '+name)
        dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        copied.append({'path':name,'sha256':sha(dst),'bytes':dst.stat().st_size})
    dump(out/'output.json',row)
    return {'status':'passed','id':output_id,'operation':'saved_artifact_export_only','files':copied,'fresh_training':False,'fresh_optimization':False}

def build_guide(out: Path) -> dict:
    if not shutil.which('pdflatex'): raise RuntimeError('An existing pdfLaTeX installation is required.')
    for src in (ROOT/'docs/guide_source').rglob('*'):
        if src.is_file():
            dst=out/src.relative_to(ROOT/'docs/guide_source');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for i in range(3):
        q=subprocess.run(['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','computational_guide.tex'],cwd=out,capture_output=True,text=True,timeout=180)
        (out/f'pass_{i+1}.stdout.txt').write_text(q.stdout);(out/f'pass_{i+1}.stderr.txt').write_text(q.stderr)
        if q.returncode:raise RuntimeError('Guide build failed; inspect the retained log.')
    log=(out/'computational_guide.log').read_text(errors='replace')
    if any(x in log for x in ['undefined references','Citation `','Overfull','Missing character:']):
        raise RuntimeError('Guide has an unresolved production diagnostic.')
    return {'status':'passed','pdf':'computational_guide.pdf','sha256':sha(out/'computational_guide.pdf'),'fresh_numerical_execution':False}

def main(argv=None) -> int:
    args=parser().parse_args(argv)
    verified=verify_package()
    if args.command=='verify':print(json.dumps(verified,indent=2));return 0
    if args.command in ('outputs','studies'):
        name='paper_outputs.json' if args.command=='outputs' else 'studies.json'
        print(json.dumps(load(ROOT/'manifests'/name),indent=2));return 0
    if args.command=='inputs':
        study=STUDIES[args.study]
        r=import_inputs(study,args.source_dir,args.input_dir,args.acknowledge_local_use_rights) if args.source_dir else verify_inputs(study,args.input_dir,args.acknowledge_local_use_rights)
        print(json.dumps({'status':'passed','files':r,'rights_granted':False},indent=2));return 0
    out=output_dir(args.out);start=time.perf_counter()
    dump(out/'invocation.json',{'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'package_manifest_sha256':sha(ROOT/'MANIFEST_SHA256.json'),'environment':environment(),'canonical_results_updated':False})
    try:
        if args.command=='export-output':r=export_output(out,args.id)
        elif args.command=='guide':r=build_guide(out)
        elif args.command=='figures':
            from .analysis.operations import figures
            r=figures(out,args.only)
        elif args.command=='eligibility':
            from .analysis.gate import environment_check
            r=environment_check();dump(out/'RESULT.json',r);print(json.dumps(r,indent=2));return 3 if r['differences'] else 0
        elif args.command in ('restart-inputs','restart-inspect'):
            from .analysis.gate import verify_external,inspect_restart_files
            r=(verify_external if args.command=='restart-inputs' else inspect_restart_files)(args.state_dir,args.acknowledge_local_use_rights)
        else:
            from . import operations as base
            from .analysis import operations as study
            if args.command in ('witness','diagnostics'):r=getattr(study,args.command)(out)
            elif args.command=='aggregate':r=(study.aggregate if args.study=='four_state_continuing' else base.aggregate)(out)
            elif args.command=='evaluate':
                r=study.continuing_evaluate(out,args.scope) if args.study=='four_state_continuing' else base.evaluate_saved(STUDIES[args.study],out,args.scope,args.input_dir,args.acknowledge_local_use_rights)
            elif args.command=='train':r=base.train(STUDIES[args.study],out,args.seed,args.method,args.profile,args.input_dir,args.acknowledge_local_use_rights,args.budget)
            elif args.command=='continuing-check':r=study.continuing_train(out,args.seed,args.decisions,args.resume)
            elif args.command=='meanfield':r=base.meanfield(out,args.iterations)
            elif args.command=='references':
                from .reference_operations import four_state_references,check_first_arrival
                r=(four_state_references if args.study=='four_state_reset' else check_first_arrival)(out)
            elif args.command=='continuation-replay':r=study.extended_train(out,args.state_dir,args.input_dir,args.run_id,args.stop,args.acknowledge_local_use_rights)
            else:raise ValueError('Unsupported command')
        r.update(command=args.command,wall_seconds=time.perf_counter()-start)
        dump(out/'RESULT.json',r);print(json.dumps(r,indent=2));return 0
    except Exception as exc:
        blocked=isinstance(exc,PermissionError) and 'environment changed' in str(exc)
        r={'status':'blocked_environment' if blocked else 'failed','error_class':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'partial_outputs_are_not_accepted_results':True}
        dump(out/'FAILED.json',r);print(json.dumps(r,indent=2));return 3 if blocked else 1
