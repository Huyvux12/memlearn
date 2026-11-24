"""Notebook-only process supervision and result export; no scientific scoring."""
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path


def safe_extract(z,path):
    root=Path(path).resolve();root.mkdir(parents=True,exist_ok=True)
    for member in z.infolist():
        target=(root/member.filename).resolve()
        if root!=target and root not in target.parents:raise ValueError('Unsafe archive member')
        if (member.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive links are not accepted')
    z.extractall(root)


def stop_group(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    if proc.poll() is None:
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:pass
    # A parent may exit on SIGTERM while its descendants ignore the signal and
    # keep stdout/CUDA alive. Reap the WHOLE job group, not just the parent.
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    if proc.poll() is None:proc.wait(timeout=3)


def stream(command,stage,cwd,timeout,log_path):
    if timeout<=0:return {'returncode':None,'timed_out':True,'stage':stage}
    print(f'[{stage}] bắt đầu; cap={int(timeout)}s',flush=True)
    env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_XET='1')
    proc=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                          encoding='utf-8',errors='replace',env=env,start_new_session=True)
    q=queue.Queue();started=time.monotonic();last=started;timed=False
    def reader():
        for line in proc.stdout:q.put(line)
    thread=threading.Thread(target=reader,daemon=True);thread.start()
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    try:
        with Path(log_path).open('a',encoding='utf-8') as log:
            while proc.poll() is None or thread.is_alive() or not q.empty():
                now=time.monotonic()
                if now-started>timeout and not timed:
                    timed=True;print(f'[{stage}] time cap; preserving completed receipts',flush=True);stop_group(proc)
                try:
                    line=q.get(timeout=.1);print(line,end='',flush=True);log.write(line);log.flush()
                except queue.Empty:pass
                if now-last>=15:
                    print(f'[{stage}] elapsed={int(now-started)}s',flush=True);last=now
            proc.wait();thread.join(timeout=1)
    finally:
        stop_group(proc);proc.stdout.close()
    return dict(returncode=proc.returncode,timed_out=timed,stage=stage,elapsed_s=time.monotonic()-started)


# Multi-stage notebook presets: (stage preset, share of the remaining time).
# Multi-stage notebook presets: (stage label, module preset, share of remaining time or None = even split, stage overrides).
_E3_COMMON={'methods':['LORA_QKVO','SEPARABLE','VECTOR_PAIR'],'layers':[6,12,18,24]}
_BOTH_ORDERS=[[f,d,o] for f in ('neutral','default') for d in ('none','q') for o in ('book_first','rule_first')]
STAGES={'e1':[('probe_v2','probe_v2',.62,{}),('mech_v2','mech_v2',1.,{})],
        # E3 (docs/PAPER_A_PLAN.md): 3 seeds x {book_first-only lr 5e-4, both orders lr 5e-4, book_first-only lr 2e-3}.
        # FULL is deterministic for fixed data, so it is scored once.
        'e3':[('full','screen_v2',None,{'methods':['FULL']})]+
             [(f'{mode}_s{seed}','screen_v2',None,dict(_E3_COMMON,seed=seed,lr=lr,**({'train_conditions':_BOTH_ORDERS} if mode=='both' else {})))
              for seed in (17,29,43) for mode,lr in (('bf',5e-4),('both',5e-4),('bfhi',2e-3))]}
# Same E3 plan split by seed for Kaggle sessions (FULL scored with seed 17).
for _seed in (17,29,43):
    STAGES[f'e3_s{_seed}']=[x for x in STAGES['e3'] if x[0].endswith(f'_s{_seed}') or (_seed==17 and x[0]=='full')]

def run_probe_notebook(root,source_zip,preset='probe_v2',install=True,total_seconds=1200,session_started=None,overrides=None):
    """Benchmark v2 probe or adapter screen (log-prob scoring). Same self-stop and always-export contract as run_notebook."""
    root=Path(root).resolve();session_started=session_started or time.monotonic();deadline=session_started+total_seconds
    run_root=root/('run_'+preset);run_root.mkdir(exist_ok=True);summary=[];error=None
    def remaining():return max(0,deadline-time.monotonic())
    try:
        if install:
            result=stream([sys.executable,'-m','pip','install','-q','-r',str(root/'requirements.txt')],'DEPENDENCIES',root,min(240,max(0,remaining()-90)),root/'dependencies.log')
            summary.append(result)
            if result['returncode']!=0:raise RuntimeError('Dependency installation failed; see dependencies.log')
        result=stream([sys.executable,'-m','pytest','-q'],'CPU SOFTWARE TESTS',root,min(150,max(0,remaining()-90)),root/'software_tests.log')
        summary.append(result)
        if result['returncode']!=0:raise RuntimeError('CPU software tests failed; model run was not started')
        import importlib
        stages=STAGES.get(preset,[(preset,preset,1.,{})])
        for index,(name,stage_preset,share,stage_overrides) in enumerate(stages):
            module='memlearn_attention.'+('screen_v2' if stage_preset.startswith('screen') else 'mech_v2' if stage_preset.startswith('mech') else 'probe_v2')
            stage_root=run_root if len(stages)==1 else run_root/name;stage_root.mkdir(exist_ok=True)
            cfg=importlib.import_module(module).preset(stage_preset)
            if overrides:cfg.update(overrides)
            cfg.update(stage_overrides)
            # `share` is the fraction of the time left when this stage starts; None splits it evenly over the stages left.
            share=share if share is not None else 1/(len(stages)-index)
            cfg.update(out=str(stage_root),budget_seconds=max(1,int((remaining()-60)*share)))
            settings=root/f'settings_{name}.json';settings.write_text(json.dumps(cfg,indent=2))
            result=stream([sys.executable,'-u','-m',module,'--preset',stage_preset,'--config',str(settings),'--out',str(stage_root)],
                          name.upper(),root,max(0,min(cfg['budget_seconds']+40,remaining()-30)),stage_root/'console.log')
            summary.append(result)
            if result['returncode']!=0 and not result['timed_out']:error=f'{name} failed; inspect ERROR.log and console.log'
    except Exception as e:error=repr(e);print('NOTEBOOK ERROR:',error,flush=True)
    finally:
        (root/'notebook_status.json').write_text(json.dumps(dict(error=error,stages=summary,elapsed_s=time.monotonic()-session_started),indent=2))
        target=root.parent/'MEMLEARN_PROBE_RESULTS.zip'
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for p in sorted(run_root.rglob('*')):
                if p.is_file() and not p.name.endswith('.tmp'):z.write(p,'run/'+str(p.relative_to(run_root)))
            for name in ('dependencies.log','software_tests.log','notebook_status.json'):
                p=root/name
                if p.exists():z.write(p,name)
            for p in sorted(root.glob('settings_*.json')):z.write(p,p.name)
            z.write(source_zip,'MEMLEARN_ATTENTION_SOURCE.zip')
        print('Saved:',target,flush=True)
        for path in sorted(run_root.rglob('*.json')):
            if path.name in ('run_status.json','probe_decision.json','method_availability.json'):print(path.relative_to(run_root),':',path.read_text()[:4000],flush=True)
    return target,error

def run_notebook(root,source_zip,preset='pilot30',run_llm=True,install=True,total_seconds=1800,
                 session_started=None,overrides=None,resume_zip=''):
    root=Path(root).resolve();session_started=session_started or time.monotonic();deadline=session_started+total_seconds
    kind=preset if run_llm else 'smoke';run_root=root/('run_'+kind);run_root.mkdir(exist_ok=True)
    summary=[];error=None
    def remaining():return max(0,deadline-time.monotonic())
    if resume_zip:
        with zipfile.ZipFile(resume_zip) as z:
            # Extract into a temporary child then move only the run subtree.
            temp=root/'restore';safe_extract(z,temp)
            from shutil import copytree
            if (run_root/'config.json').exists():raise ValueError('Choose an empty run directory before restoring')
            copytree(temp/'run',run_root,dirs_exist_ok=True)
    try:
        if install:
            result=stream([sys.executable,'-m','pip','install','-q','-r',str(root/'requirements.txt')],
                          'DEPENDENCIES',root,min(240,max(0,remaining()-90)),root/'dependencies.log')
            summary.append(result)
            if result['returncode']!=0:raise RuntimeError('Dependency installation failed; see dependencies.log')
        result=stream([sys.executable,'-m','pytest','-q'], 'CPU SOFTWARE TESTS',root,
                      min(120,max(0,remaining()-90)),root/'software_tests.log')
        summary.append(result)
        if result['returncode']!=0:raise RuntimeError('CPU software tests failed; model run was not started')
        from memlearn_attention.runner import preset as get_preset
        if (run_root/'config.json').exists():
            cfg=json.loads((run_root/'config.json').read_text())
            if cfg['name']!=kind:raise ValueError('Resume preset mismatch')
            if overrides and any(cfg.get(k)!=v for k,v in overrides.items()):raise ValueError('Overrides differ from saved config; start a new run')
        else:
            cfg=get_preset(kind)
            if overrides:cfg.update(overrides)
            cfg.update(out=str(run_root),budget_seconds=max(1,int(remaining()-65)))
        settings=root/'run_settings.json';settings.write_text(json.dumps(cfg,indent=2))
        result=stream([sys.executable,'-u','-m','memlearn_attention.runner','--preset',kind,'--config',str(settings),'--out',str(run_root)],
                      'QWEN TRAIN + EVAL' if run_llm else 'TEST ONLY RANDOM QWEN',root,max(0,remaining()-55),run_root/'console.log')
        summary.append(result)
        if result['returncode']!=0 and not result['timed_out']:error='Neural runner failed; inspect run/ERROR.log and console.log'
    except Exception as e:error=repr(e);print('NOTEBOOK ERROR:',error,flush=True)
    finally:
        if (run_root/'config.json').exists():
            for flag,stage in [('--export','EXPORT COMPLETED METRICS'),('--audit','OFFLINE SCORE REPLAY')]:
                if remaining()<5:break
                summary.append(stream([sys.executable,'-m','memlearn_attention.runner',flag,'--out',str(run_root)],
                                      stage,root,min(30,max(1,remaining()-5)),run_root/'recovery.log'))
        (root/'notebook_status.json').write_text(json.dumps(dict(error=error,stages=summary,elapsed_s=time.monotonic()-session_started),indent=2))
        target=root.parent/'MEMLEARN_ATTENTION_RESULTS.zip'
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for p in sorted(run_root.rglob('*')):
                if p.is_file() and not p.name.endswith('.tmp'):z.write(p,'run/'+str(p.relative_to(run_root)))
            for name in ('dependencies.log','software_tests.log','notebook_status.json','README_VI.md'):
                p=root/name
                if p.exists():z.write(p,name)
            z.write(source_zip,'MEMLEARN_ATTENTION_SOURCE.zip')
        print('Saved:',target,flush=True)
        for name in ('audit.json','run_status.json','validity_by_rq.json'):
            path=run_root/name
            if path.exists():print(name+':',path.read_text()[:6000],flush=True)
    return target,error
