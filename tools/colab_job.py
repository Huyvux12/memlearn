"""Colab VM driver: run several experiment stages in parallel on one GPU, then print JOB DONE.

Usage on the VM:  python colab_job.py job.json
The spec lists stages {label, module, preset, overrides}; `parallel` bounds concurrent processes. Each stage is
its own process with its own model copy (the GPU is far from saturated by batch-1 work), all reading one shared
HF cache filled before launch. Each stage self-stops at its budget; a hard kill follows after a grace period.
"""
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

GRACE=180

def run_job(spec,root,src_zip=None,install=True,tests=True,predownload=True,python=sys.executable):
    root=Path(root);runs=root/'runs';runs.mkdir(parents=True,exist_ok=True);started=time.monotonic();log=[]
    if src_zip and not (root/'memlearn_attention').exists():
        with zipfile.ZipFile(src_zip) as z:z.extractall(root)
    env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_XET='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def step(name,cmd,timeout):
        t=time.monotonic();r=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=timeout)
        (runs/f'{name}.log').write_text(r.stdout+r.stderr,encoding='utf-8');log.append(dict(stage=name,returncode=r.returncode,elapsed_s=time.monotonic()-t))
        print(f'STEP {name} rc={r.returncode} {time.monotonic()-t:.0f}s',flush=True);return r.returncode
    if install and step('dependencies',[python,'-m','pip','install','-q','-r','requirements.txt'],600):raise SystemExit('dependency install failed')
    if tests and step('software_tests',[python,'-m','pytest','-q','-p','no:cacheprovider'],600):raise SystemExit('software tests failed; no model run')
    if predownload:
        models=sorted({st['overrides'].get('model') for st in spec['stages'] if st['overrides'].get('model')}|set(spec.get('models',[])))
        code=('from huggingface_hub import snapshot_download\n'+''.join(
            f"snapshot_download({m!r},allow_patterns=['*.json','*.safetensors','*.model','*.jinja','merges.txt','vocab.json'],ignore_patterns=['consolidated*'],max_workers=8)\n" for m in models))
        if models and step('predownload',[python,'-c',code],1800):raise SystemExit('model download failed')
    budget_gb=spec.get('vram_budget_gb')
    def fits(st,active):
        # Optional VRAM bookkeeping: a stage starts only if the declared vram_gb of the running stages leaves room (or nothing runs).
        return budget_gb is None or not active or sum(a[0].get('vram_gb',0) for a in active.values())+st.get('vram_gb',0)<=budget_gb
    def drain(stages,parallel,attempt):
        pending=list(stages);active={};failed=[]
        while pending or active:
            while pending and len(active)<parallel:
                i=next((i for i,st in enumerate(pending) if fits(st,active)),None)
                if i is None:break
                st=pending.pop(i);out=runs/st['label'];out.mkdir(parents=True,exist_ok=True)
                module=importlib_preset(root,st['module'],st['preset'],python)
                cfg=dict(module,**st['overrides'],out=str(out),budget_seconds=st.get('budget_seconds',spec.get('budget_seconds',1800)))
                settings=out/'settings.json';settings.write_text(json.dumps(cfg,indent=2))
                handle=open(out/('console.log' if attempt==1 else f'console_attempt{attempt}.log'),'w',encoding='utf-8')
                proc=subprocess.Popen([python,'-u','-m',st['module'],'--preset',st['preset'],'--config',str(settings),'--out',str(out)],
                                      cwd=root,env=env,stdout=handle,stderr=subprocess.STDOUT)
                active[st['label']]=(st,proc,handle,time.monotonic(),cfg['budget_seconds'])
                print(f'START {st["label"]} pid={proc.pid} attempt={attempt}',flush=True)
            for label,(st,proc,handle,t0,budget) in list(active.items()):
                if proc.poll() is None and time.monotonic()-t0>budget+GRACE:proc.kill();print(f'KILLED {label} after budget+grace',flush=True)
                if proc.poll() is not None:
                    handle.close();del active[label];log.append(dict(stage=label,returncode=proc.returncode,elapsed_s=time.monotonic()-t0,attempt=attempt))
                    print(f'END {label} rc={proc.returncode} {time.monotonic()-t0:.0f}s',flush=True)
                    if proc.returncode!=0:failed.append(st)
            time.sleep(2)
        return failed
    failed=drain(spec['stages'],spec.get('parallel',1),1)
    # Failures under parallel load (typically CUDA OOM while several large models share the GPU) get one
    # sequential retry on an otherwise idle GPU; each stage resumes from its own receipts.
    if failed and spec.get('retry_failed',True):
        print('RETRY '+' '.join(st['label'] for st in failed),flush=True);drain(failed,1,2)
    (runs/'job_status.json').write_text(json.dumps(dict(spec=spec,steps=log,elapsed_s=time.monotonic()-started),indent=2))
    target=root.parent/f'results_{spec["name"]}.zip'
    # Only this job's stage folders and its own logs: --root may be a checkout whose runs/ holds unrelated results.
    mine=[runs/st['label'] for st in spec['stages']]
    files=[p for d in mine if d.exists() for p in d.rglob('*')]+[runs/f'{x["stage"]}.log' for x in log if (runs/f'{x["stage"]}.log').exists()]+[runs/'job_status.json']
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(set(files)):
            # Adapter weights are large and reproducible from seeds; keep them only when asked.
            if p.is_file() and not p.name.endswith('.tmp') and (spec.get('keep_adapters') or p.suffix!='.pt'):z.write(p,'runs/'+str(p.relative_to(runs)))
    return target,log

def importlib_preset(root,module,preset,python):
    """Resolve a module preset in a clean subprocess so the driver never imports torch itself."""
    code=f'import json,sys;sys.path.insert(0,{str(root)!r});import importlib;print(json.dumps(importlib.import_module({module!r}).preset({preset!r})))'
    return json.loads(subprocess.run([python,'-c',code],capture_output=True,text=True,check=True,cwd=root).stdout)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='Run a job spec (stages in parallel on one GPU).')
    p.add_argument('spec');p.add_argument('--root',default='/content/memlearn',help='working copy of the source (created from --src if missing)')
    p.add_argument('--src',default='/content/memlearn_src.zip',help='source archive; use "" when --root already holds the source')
    p.add_argument('--skip-install',action='store_true');p.add_argument('--skip-tests',action='store_true')
    a=p.parse_args();spec=json.loads(Path(a.spec).read_text())
    target,log=run_job(spec,a.root,a.src or None,install=not a.skip_install,tests=not a.skip_tests)
    print('RESULT',target,flush=True)
    print('JOB DONE',flush=True)
