import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_parallel_job_runs_stages_concurrently_and_zips_without_adapters(tmp_path):
    sys.path.insert(0,str(ROOT))
    from tools.colab_job import run_job
    root=tmp_path/'root'
    for folder in ('memlearn_attention','tools'):shutil.copytree(ROOT/folder,root/folder,ignore=shutil.ignore_patterns('__pycache__'))
    spec=dict(name='t',parallel=2,budget_seconds=300,stages=[
        dict(label='probe',module='memlearn_attention.probe_v2',preset='probe_smoke',overrides={}),
        dict(label='screen',module='memlearn_attention.screen_v2',preset='screen_smoke',overrides={'methods':['FULL','QUERY'],'chunk':64})])
    target,log=run_job(spec,root,install=False,tests=False,predownload=False)
    assert {x['stage'] for x in log}=={'probe','screen'} and all(x['returncode']==0 for x in log)
    with zipfile.ZipFile(target) as z:names=z.namelist()
    assert 'runs/probe/run_status.json' in names and 'runs/screen/run_status.json' in names
    assert 'runs/screen/settings.json' in names and not any(n.endswith('.pt') for n in names)
    settings=json.loads((root/'runs'/'screen'/'settings.json').read_text())
    assert settings['chunk']==64 and settings['methods']==['FULL','QUERY'] and settings['budget_seconds']==300
    assert json.loads((root/'runs'/'screen'/'run_status.json').read_text())['status']=='COMPLETE'

def test_reproduce_all_spec_is_resolvable_and_layout_points_at_its_stages():
    import importlib
    sys.path.insert(0,str(ROOT))
    from tools.make_job_specs import reproduce_all,layout_reproduce
    spec=reproduce_all();labels=[s['label'] for s in spec['stages']]
    assert len(labels)==len(set(labels))
    for s in spec['stages']:
        cfg=importlib.import_module(s['module']).preset(s['preset'])
        unknown=set(s['overrides'])-set(cfg)-{'allow_any_arch','style','conditions','modes','generate_every','train_conditions','model'}
        assert not unknown,(s['label'],unknown)
    layout=layout_reproduce()
    refs=[v for v in layout.values() if isinstance(v,dict) and 'label' in v]
    refs+=[x for v in layout.values() if isinstance(v,dict) for x in v.values() if isinstance(x,dict) and 'label' in x]
    assert refs and all(r['label'] in labels for r in refs)
    prefixes=layout['adaptation'][0]['prefixes']
    assert all(any(l.startswith(p) for l in labels) for p in prefixes)

def test_job_zip_excludes_unrelated_runs(tmp_path):
    sys.path.insert(0,str(ROOT))
    from tools.colab_job import run_job
    root=tmp_path/'root'
    for folder in ('memlearn_attention','tools'):shutil.copytree(ROOT/folder,root/folder,ignore=shutil.ignore_patterns('__pycache__'))
    (root/'runs'/'old_unrelated').mkdir(parents=True);(root/'runs'/'old_unrelated'/'secret.json').write_text('{}')
    spec=dict(name='x',parallel=1,budget_seconds=200,stages=[dict(label='p',module='memlearn_attention.probe_v2',preset='probe_smoke',overrides={})])
    target,_=run_job(spec,root,install=False,tests=False,predownload=False)
    with zipfile.ZipFile(target) as z:names=z.namelist()
    assert 'runs/p/run_status.json' in names and 'runs/job_status.json' in names and not any('old_unrelated' in n for n in names)

def test_failed_stage_is_retried_once_sequentially(tmp_path):
    sys.path.insert(0,str(ROOT))
    from tools.colab_job import run_job
    root=tmp_path/'root'
    for folder in ('memlearn_attention','tools'):shutil.copytree(ROOT/folder,root/folder,ignore=shutil.ignore_patterns('__pycache__'))
    # A stage that fails on the first attempt only (a marker file flips it), like an OOM under parallel load.
    (root/'flaky.py').write_text("import sys,pathlib\nm=pathlib.Path('flaky_seen')\nif not m.exists():m.write_text('1');sys.exit(3)\n"
                                 "import json;p=pathlib.Path(sys.argv[sys.argv.index('--out')+1]);(p/'run_status.json').write_text(json.dumps({'status':'COMPLETE'}))\n")
    (root/'flaky_mod.py').write_text("def preset(name):return {'name':name}\n")
    spec=dict(name='r',parallel=2,budget_seconds=60,stages=[dict(label='f',module='flaky',preset='x',overrides={})])
    import tools.colab_job as cj
    orig=cj.importlib_preset;cj.importlib_preset=lambda root,module,preset,python:{'name':preset}
    try:target,log=run_job(spec,root,install=False,tests=False,predownload=False)
    finally:cj.importlib_preset=orig
    assert [(x['stage'],x['returncode'],x['attempt']) for x in log]==[('f',3,1),('f',0,2)]
    with zipfile.ZipFile(target) as z:assert 'runs/f/run_status.json' in z.namelist()

def test_vram_budget_keeps_large_stages_apart(tmp_path):
    sys.path.insert(0,str(ROOT))
    from tools.colab_job import run_job
    root=tmp_path/'root'
    for folder in ('memlearn_attention','tools'):shutil.copytree(ROOT/folder,root/folder,ignore=shutil.ignore_patterns('__pycache__'))
    (root/'sleeper.py').write_text("import sys,time,json,pathlib\np=pathlib.Path(sys.argv[sys.argv.index('--out')+1]);t=time.time();time.sleep(1.5)\n"
                                   "(p/'span.json').write_text(json.dumps([t,time.time()]))\n")
    spec=dict(name='v',parallel=3,budget_seconds=60,vram_budget_gb=100,stages=[dict(label=l,module='sleeper',preset='x',overrides={},vram_gb=g)
                                                                             for l,g in (('big1',60),('big2',60),('small',30))])
    import tools.colab_job as cj
    orig=cj.importlib_preset;cj.importlib_preset=lambda root,module,preset,python:{'name':preset}
    try:run_job(spec,root,install=False,tests=False,predownload=False)
    finally:cj.importlib_preset=orig
    span={l:json.loads((root/'runs'/l/'span.json').read_text()) for l in ('big1','big2','small')}
    overlap=lambda a,b:span[a][0]<span[b][1] and span[b][0]<span[a][1]
    assert not overlap('big1','big2') and overlap('big1','small')  # the small stage fills the gap next to a big one
