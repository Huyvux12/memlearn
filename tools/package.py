"""Build a deterministic source archive and a standalone small-cell notebook."""
import base64
import hashlib
import json
import textwrap
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def build(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    source=destination/'MEMLEARN_ATTENTION_SOURCE.zip'
    files=[]
    for folder in ('memlearn_attention','tests','tools','docs','configs'):
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.json')]
    files += [ROOT/x for x in ('requirements.txt','pyproject.toml','README.md','README_VI.md','LICENSE') if (ROOT/x).exists()]
    with zipfile.ZipFile(source,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(files):
            info=zipfile.ZipInfo(str(path.relative_to(ROOT)),(2026,10,17,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,path.read_bytes())
    raw=source.read_bytes();sha=hashlib.sha256(raw).hexdigest();encoded=base64.b64encode(raw).decode()
    cells=[]
    def markdown(text):cells.append({'cell_type':'markdown','metadata':{},'source':[text]})
    def code(text,payload=False):cells.append({'cell_type':'code','execution_count':None,'metadata':{'payload':True} if payload else {},'outputs':[],'source':[textwrap.dedent(text).lstrip()]})
    markdown('# MEMLEARN Attention Lab\n\nCan thiệp attention và train adapter thật trên frozen Qwen3. Bật **GPU + Internet**, Run All. Chỉ cần notebook này. `pilot30` là phép sàng lọc khoảng 30 phút, không phải toàn bộ bằng chứng paper. Kết quả âm vẫn hợp lệ; phần thiếu được ghi NOT_RUN. Đọc README_VI trong ZIP để hiểu RQ/metric/baseline.')
    code('''
    import time
    SESSION_STARTED = time.monotonic()
    PRESET = 'pilot30'  # 'research' needs a much longer budget
    RUN_LLM = True     # False => tiny random Qwen SOFTWARE TEST ONLY
    INSTALL_DEPENDENCIES = True
    TOTAL_BUDGET_SECONDS = 1800
    OVERRIDES = {}    # e.g. {'rank': 4}; a changed config requires a new run
    RESUME_ZIP = ''   # optional path to a prior results ZIP from this exact code/config
    ''')
    code('PAYLOAD_PARTS = []',payload=True)
    for start in range(0,len(encoded),6000):code('PAYLOAD_PARTS.append('+repr(encoded[start:start+6000])+')',payload=True)
    code(f'''
    import base64, hashlib, io, json, sys, zipfile
    from pathlib import Path
    payload = base64.b64decode(''.join(PAYLOAD_PARTS))
    assert hashlib.sha256(payload).hexdigest() == {sha!r}, 'Source payload changed'
    base = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
    ROOT = base / 'MEMLEARN_ATTENTION_{sha[:10]}'
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            target = (ROOT / member.filename).resolve()
            assert ROOT.resolve() in target.parents, 'Unsafe payload path'
        archive.extractall(ROOT)
    SOURCE_ZIP = ROOT / 'source_payload.zip'
    SOURCE_ZIP.write_bytes(payload)
    sys.path.insert(0, str(ROOT))
    del payload, PAYLOAD_PARTS
    print('Source ready:', ROOT)
    ''')
    code('''
    from tools.notebook_runtime import run_notebook
    result_zip, error = run_notebook(
        ROOT, SOURCE_ZIP, preset=PRESET, run_llm=RUN_LLM,
        install=INSTALL_DEPENDENCIES, total_seconds=TOTAL_BUDGET_SECONDS,
        session_started=SESSION_STARTED, overrides=OVERRIDES, resume_zip=RESUME_ZIP)
    try:
        from IPython.display import display, FileLink
        display(FileLink(str(result_zip)))
    except ImportError:
        print('Result ZIP:', result_zip)
    if error:
        print('Run has an error; the ZIP contains diagnostics and completed measurements:', error)
    else:
        print('Download ZIP. Read run/report.md and run/coverage.csv first.')
    ''')
    notebook=destination/'MEMLEARN_ATTENTION_KAGGLE.ipynb'
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
    # Stable cell IDs satisfy nbformat 4.5 and make payload cells inspectable.
    for i,cell in enumerate(cells):cell['id']=f'attention-{i:03d}'
    notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return {'notebook':notebook,'source_zip':source,'source_sha256':sha}

def _plain_files():
    files=[]
    for folder in ('memlearn_attention','tests','tools','docs','configs'):
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.json')]
    files += [ROOT/x for x in ('requirements.txt','pyproject.toml','README.md','README_VI.md') if (ROOT/x).exists()]
    # LF and exactly one final newline: %%writefile cannot represent a missing final newline.
    return {str(p.relative_to(ROOT)).replace('\\','/'):p.read_text(encoding='utf-8').replace('\r\n','\n').rstrip('\n')+'\n' for p in sorted(files)}

def plain_source_sha256(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def build_plain(destination,preset='pilot_v11',total_seconds=1500,name='MEMLEARN_ATTENTION_KAGGLE_PLAIN.ipynb',overrides=None,colab=False):
    """Readable notebook: every source file is a visible %%writefile cell; no encoded payload."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    files=_plain_files();sha=plain_source_sha256(files)
    from memlearn_attention import __version__ as version
    cells=[]
    def markdown(text):cells.append({'cell_type':'markdown','metadata':{},'source':[text]})
    def code(text,meta=None):cells.append({'cell_type':'code','execution_count':None,'metadata':meta or {},'outputs':[],'source':[text]})
    markdown(f'# MEMLEARN Attention Lab {version} (mã nguồn đọc được)\n\nMỗi file nguồn nằm trong một cell `%%writefile` bên dưới, không mã hóa. '
             f'Cell kiểm tra so SHA256 của toàn bộ file đã ghi với `{sha[:16]}…`. Bật **GPU + Internet**, Run All. '
             f'Notebook tự dừng theo `TOTAL_BUDGET_SECONDS` và luôn xuất ZIP kết quả.')
    code(textwrap.dedent(f'''
    import os, time
    from pathlib import Path
    SESSION_STARTED = time.monotonic()
    PRESET = {preset!r}
    RUN_LLM = True     # False => tiny random Qwen SOFTWARE TEST ONLY
    INSTALL_DEPENDENCIES = True
    TOTAL_BUDGET_SECONDS = {int(total_seconds)}  # hard self-stop incl. install, tests, model download, export
    OVERRIDES = {overrides or {}!r}
    RESUME_ZIP = ''
    EXPECTED_SOURCE_SHA256 = {sha!r}
    base = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
    ROOT = base / 'MEMLEARN_ATTENTION_{version}'
    for folder in {sorted({str(Path(k).parent).replace(chr(92),'/') for k in files} - {'.'})!r}:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    print('Writing source into', ROOT)
    ''').lstrip())
    for rel,text in files.items():
        markdown(f'### `{rel}`')
        code(f'%%writefile {rel}\n'+text,{'source_file':rel})
    code(textwrap.dedent('''
    import hashlib, json, sys, zipfile
    written = {}
    for path in sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.md', '.json', '.txt', '.toml')):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(('run_', 'restore/')) or rel in ('notebook_status.json', 'run_settings.json'): continue
        text = path.read_text(encoding='utf-8').rstrip('\\n') + '\\n'
        path.write_text(text, encoding='utf-8'); written[rel] = text
    actual = hashlib.sha256(json.dumps(written, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert actual == EXPECTED_SOURCE_SHA256, f'Source mismatch: {actual}'
    SOURCE_ZIP = ROOT / 'source_payload.zip'
    with zipfile.ZipFile(SOURCE_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for rel, text in written.items(): z.writestr(zipfile.ZipInfo(rel, (2026, 10, 22, 0, 0, 0)), text)
    sys.path.insert(0, str(ROOT))
    print('Source verified:', actual, len(written), 'files')
    ''').lstrip())
    if preset.startswith(('probe','screen','mech','e1','e3')):
        launch='''
    from tools.notebook_runtime import run_probe_notebook
    result_zip, error = run_probe_notebook(
        ROOT, SOURCE_ZIP, preset=PRESET, install=INSTALL_DEPENDENCIES,
        total_seconds=TOTAL_BUDGET_SECONDS, session_started=SESSION_STARTED, overrides=OVERRIDES)'''
    else:
        launch='''
    from tools.notebook_runtime import run_notebook
    result_zip, error = run_notebook(
        ROOT, SOURCE_ZIP, preset=PRESET, run_llm=RUN_LLM,
        install=INSTALL_DEPENDENCIES, total_seconds=TOTAL_BUDGET_SECONDS,
        session_started=SESSION_STARTED, overrides=OVERRIDES, resume_zip=RESUME_ZIP)'''
    code(textwrap.dedent(launch+'''
    print('Result ZIP:', result_zip)
    if error:
        print('Run has an error; the ZIP contains diagnostics and completed measurements:', error)
    else:
        print('Download ZIP. Read run/report.md and run/coverage.csv first.')
    ''').lstrip())
    if colab:
        # Drive is mounted first so the user authorizes at the start; results are copied there at the end.
        cells.insert(1,{'cell_type':'markdown','metadata':{},'source':['## Colab\n\nRuntime → Change runtime type → GPU (A100 nếu có). '
                        'Cell dưới mount Google Drive để lưu ZIP kết quả vào `MyDrive/MEMLEARN/`; giữ tab mở trong lúc chạy.']})
        cells.insert(2,{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[textwrap.dedent('''
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            print('Not running on Colab; results stay in the working directory.')
        ''').lstrip()]})
        code(textwrap.dedent('''
        import shutil
        from pathlib import Path
        target = Path('/content/drive/MyDrive/MEMLEARN')
        if Path('/content/drive/MyDrive').exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy(result_zip, target / f'{PRESET}_{Path(result_zip).name}')
            print('Copied to', target)
        ''').lstrip())
    for i,cell in enumerate(cells):cell['id']=f'plain-{i:03d}'
    notebook=destination/name
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
    notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return {'notebook':notebook,'source_sha256':sha,'files':files}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',default='dist');p.add_argument('--plain',action='store_true')
    p.add_argument('--preset',default='pilot_v11');p.add_argument('--seconds',type=int,default=1500);p.add_argument('--overrides',default='{}');p.add_argument('--colab',action='store_true');a=p.parse_args()
    if a.plain:
        import sys;sys.path.insert(0,str(ROOT));r=build_plain(a.out,a.preset,a.seconds,overrides=json.loads(a.overrides),colab=a.colab);print({'notebook':str(r['notebook']),'source_sha256':r['source_sha256'],'files':len(r['files'])})
    else:print({k:str(v) for k,v in build(a.out).items()})
