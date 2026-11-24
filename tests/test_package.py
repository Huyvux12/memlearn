import importlib.util
import json
import zipfile
import base64
import hashlib
import pytest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_standalone_notebook_embeds_exact_source_and_small_compiling_cells(tmp_path):
    assert (ROOT/'tools'/'package.py').exists(), 'notebook packager missing'
    import runpy
    package=runpy.run_path(str(ROOT/'tools'/'package.py'))
    artifacts=package['build'](tmp_path)
    nb=json.loads(artifacts['notebook'].read_text());ns={}
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source']);compile(source,'notebook-cell','exec')
        assert len(source)<15000
        if cell.get('metadata',{}).get('payload'):exec(source,ns)
    payload=base64.b64decode(''.join(ns['PAYLOAD_PARTS']))
    assert hashlib.sha256(payload).hexdigest()==artifacts['source_sha256']
    assert payload==artifacts['source_zip'].read_bytes()
    with zipfile.ZipFile(artifacts['source_zip']) as z:
        assert z.read('memlearn_attention/attention.py')==(ROOT/'memlearn_attention'/'attention.py').read_bytes()
        assert z.read('tests/test_attention.py')==(ROOT/'tests'/'test_attention.py').read_bytes()
        assert not any('test_runtime' in n or '__pycache__' in n or '.work/' in n for n in z.namelist())

def test_plain_notebook_writes_exact_readable_source_without_payload(tmp_path):
    import runpy,sys
    sys.path.insert(0,str(ROOT))
    package=runpy.run_path(str(ROOT/'tools'/'package.py'))
    artifacts=package['build_plain'](tmp_path)
    nb=json.loads(artifacts['notebook'].read_text(encoding='utf-8'));written={}
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        assert not cell['metadata'].get('payload')
        if source.startswith('%%writefile '):
            head,body=source.split('\n',1);written[head.split(' ',1)[1]]=body
        else:
            # Executable (non-source) cells must not decode hidden payloads.
            assert 'PAYLOAD_PARTS' not in source and 'b64decode' not in source
            compile(source,'notebook-cell','exec')
    assert written==artifacts['files']
    assert written['memlearn_attention/attention.py']==(ROOT/'memlearn_attention'/'attention.py').read_text(encoding='utf-8').rstrip('\n')+'\n'
    assert 'tests/test_runner.py' in written and not any('.env' in k or '__pycache__' in k for k in written)
    assert package['plain_source_sha256'](written)==artifacts['source_sha256']

def test_plain_probe_notebook_launches_probe_runtime(tmp_path):
    import runpy,sys
    sys.path.insert(0,str(ROOT))
    artifacts=runpy.run_path(str(ROOT/'tools'/'package.py'))['build_plain'](tmp_path,preset='probe_v2',total_seconds=1200)
    cells=[''.join(c['source']) for c in json.loads(artifacts['notebook'].read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code']
    assert 'run_probe_notebook' in cells[-1] and "PRESET = 'probe_v2'" in cells[0]
    compile(cells[-1],'launch','exec')

def test_pilot_v11_preset_is_valid_and_shuffle_is_seeded():
    from memlearn_attention import runner
    cfg=runner.preset('pilot_v11');runner.validate_config(cfg)
    assert cfg['shuffle_examples'] and cfg['max_train_steps']==0 and 'LORA_V' in cfg['methods']
    assert not runner.preset('pilot30')['shuffle_examples']

def test_notebook_supervisor_stops_blocking_child(tmp_path):
    import runpy,sys,time
    path=ROOT/'tools'/'notebook_runtime.py'
    assert path.exists()
    runtime=runpy.run_path(str(path));start=time.monotonic()
    result=runtime['stream']([sys.executable,'-c','import time;time.sleep(30)'],'TEST',ROOT,.3,tmp_path/'log')
    assert result['timed_out'] and result['returncode']!=0
    assert time.monotonic()-start<7

def test_archive_traversal_rejected(tmp_path):
    import io,runpy
    runtime=runpy.run_path(str(ROOT/'tools'/'notebook_runtime.py'))
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,'w') as z:z.writestr('../escape','bad')
    with zipfile.ZipFile(io.BytesIO(raw.getvalue())) as z:
        with pytest.raises(ValueError):runtime['safe_extract'](z,tmp_path)

def test_supervisor_kills_descendant_ignoring_sigterm(tmp_path):
    import runpy,sys,os,signal,time
    runtime=runpy.run_path(str(ROOT/'tools'/'notebook_runtime.py'))
    pidfile=tmp_path/'pid'
    child="import signal,time,os;signal.signal(signal.SIGTERM,signal.SIG_IGN);open("+repr(str(pidfile))+",'w').write(str(os.getpid()));time.sleep(30)"
    parent='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(child)+']);time.sleep(30)'
    started=time.monotonic()
    runtime['stream']([sys.executable,'-c',parent],'DESCENDANT',ROOT,.6,tmp_path/'log')
    assert time.monotonic()-started<7, 'descendant kept supervisor blocked after cap'
    pid=int(pidfile.read_text())
    try:
        proc=Path('/proc')/str(pid)/'stat'
        assert not proc.exists() or proc.read_text().split()[2]=='Z', 'orphan child still alive'
    finally:
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass

def test_multistage_e1_runs_probe_then_mech_with_shared_overrides(tmp_path,monkeypatch):
    import runpy,sys
    sys.path.insert(0,str(ROOT))
    from tools import notebook_runtime as rt
    calls=[]
    def fake_stream(command,stage,cwd,timeout,log_path):
        calls.append((stage,command,timeout))
        if '--out' in command:
            out=Path(command[command.index('--out')+1]);out.mkdir(parents=True,exist_ok=True);(out/'run_status.json').write_text('{"status":"COMPLETE"}')
        return dict(returncode=0,timed_out=False,stage=stage,elapsed_s=0.)
    monkeypatch.setattr(rt,'stream',fake_stream)
    root=tmp_path/'root';root.mkdir();src=tmp_path/'src.zip'
    with zipfile.ZipFile(src,'w') as z:z.writestr('x.txt','x')
    target,error=rt.run_probe_notebook(root,src,preset='e1',install=False,total_seconds=1000,overrides={'model':'Qwen/Qwen3-1.7B','variants':['M0','M1']})
    assert error is None
    stages=[c[0] for c in calls]
    assert stages==['CPU SOFTWARE TESTS','PROBE_V2','MECH_V2']
    for name in ('probe_v2','mech_v2'):
        cfg=json.loads((root/f'settings_{name}.json').read_text())
        assert cfg['model']=='Qwen/Qwen3-1.7B' and cfg['allow_any_arch'] and cfg['out'].endswith(name)
    assert json.loads((root/'settings_mech_v2.json').read_text())['variants']==['M0','M1']
    with zipfile.ZipFile(target) as z:
        names=z.namelist()
    assert 'run/probe_v2/run_status.json' in names and 'run/mech_v2/run_status.json' in names and 'settings_probe_v2.json' in names

def test_e3_stage_plan_covers_seeds_modes_and_scores_full_once(tmp_path,monkeypatch):
    import sys
    sys.path.insert(0,str(ROOT))
    from tools import notebook_runtime as rt
    seen=[]
    def fake_stream(command,stage,cwd,timeout,log_path):
        if '--config' in command:seen.append(json.loads(Path(command[command.index('--config')+1]).read_text()))
        return dict(returncode=0,timed_out=False,stage=stage,elapsed_s=0.)
    monkeypatch.setattr(rt,'stream',fake_stream)
    root=tmp_path/'root';root.mkdir();src=tmp_path/'src.zip'
    with zipfile.ZipFile(src,'w') as z:z.writestr('x.txt','x')
    rt.run_probe_notebook(root,src,preset='e3',install=False,total_seconds=10000)
    assert len(seen)==10 and seen[0]['methods']==['FULL']
    trained=seen[1:]
    assert all('FULL' not in c['methods'] and c['layers']==[6,12,18,24] for c in trained)
    assert sorted({c['seed'] for c in trained})==[17,29,43]
    both=[c for c in trained if any(t[2]=='rule_first' for t in c['train_conditions'])]
    assert len(both)==3 and all(c['lr']==5e-4 for c in both)
    assert sorted(c['lr'] for c in trained if c not in both)==[5e-4]*3+[2e-3]*3
    assert len({c['out'] for c in seen})==10 and all(c['n_test']==12 for c in seen)

def test_e3_split_by_seed_is_a_partition_of_e3():
    import sys
    sys.path.insert(0,str(ROOT))
    from tools import notebook_runtime as rt
    parts=[rt.STAGES[f'e3_s{s}'] for s in (17,29,43)]
    assert sorted(x[0] for p in parts for x in p)==sorted(x[0] for x in rt.STAGES['e3'])
    assert parts[0][0][0]=='full' and all(len(p)==3 for p in parts[1:]) and len(parts[0])==4
